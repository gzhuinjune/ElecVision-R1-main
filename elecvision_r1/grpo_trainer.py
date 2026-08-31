from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - training extras are optional for local checks.
    torch = None
    F = None


def _require_torch() -> None:
    if torch is None or F is None:
        raise ImportError("PyTorch is required for ElecVisionGRPOTorchTrainer")


@dataclass
class GRPOLossConfig:
    clip_range: float = 0.2
    kl_beta: float = 0.04
    eps: float = 1e-8


@dataclass
class GRPOLossOutput:
    loss: Any
    policy_loss: Any
    kl_loss: Any
    mean_ratio: Any
    mean_kl: Any


def masked_mean(values: Any, mask: Any, eps: float = 1e-8) -> Any:
    _require_torch()
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(eps)


def masked_sequence_mean(values: Any, mask: Any, eps: float = 1e-8) -> Any:
    _require_torch()
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(eps)


def token_logprobs_from_logits(logits: Any, labels: Any) -> Any:
    _require_torch()
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def causal_response_logprobs(logits: Any, input_ids: Any, response_mask: Any | None = None) -> tuple[Any, Any]:
    _require_torch()
    shifted_logits = logits[:, :-1, :]
    shifted_labels = input_ids[:, 1:]
    logprobs = token_logprobs_from_logits(shifted_logits, shifted_labels)
    if response_mask is None:
        mask = torch.ones_like(logprobs, dtype=torch.bool)
    else:
        mask = response_mask[:, 1:].to(dtype=torch.bool)
    return logprobs, mask


def grpo_token_kl(policy_logprobs: Any, reference_logprobs: Any) -> Any:
    _require_torch()
    diff = reference_logprobs - policy_logprobs
    return torch.exp(diff) - diff - 1.0


def grpo_policy_loss(
    *,
    policy_logprobs: Any,
    old_logprobs: Any,
    reference_logprobs: Any,
    advantages: Any,
    action_mask: Any,
    config: GRPOLossConfig | None = None,
) -> GRPOLossOutput:
    _require_torch()
    cfg = config or GRPOLossConfig()
    if advantages.ndim == 1:
        advantages = advantages[:, None]
    advantages = advantages.to(dtype=policy_logprobs.dtype)

    ratio = torch.exp(policy_logprobs - old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    policy_loss = -masked_sequence_mean(surrogate, action_mask, eps=cfg.eps).mean()

    token_kl = grpo_token_kl(policy_logprobs, reference_logprobs)
    kl_loss = masked_sequence_mean(token_kl, action_mask, eps=cfg.eps).mean()
    loss = policy_loss + cfg.kl_beta * kl_loss
    return GRPOLossOutput(
        loss=loss,
        policy_loss=policy_loss.detach(),
        kl_loss=kl_loss.detach(),
        mean_ratio=masked_mean(ratio.detach(), action_mask, eps=cfg.eps),
        mean_kl=kl_loss.detach(),
    )


class ElecVisionGRPOTorchTrainer:
    def __init__(
        self,
        policy_model: Any,
        reference_model: Any,
        optimizer: Any,
        *,
        loss_config: GRPOLossConfig | None = None,
    ) -> None:
        _require_torch()
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.optimizer = optimizer
        self.loss_config = loss_config or GRPOLossConfig()
        self.reference_model.eval()
        for parameter in self.reference_model.parameters():
            parameter.requires_grad_(False)

    def sequence_logprobs(self, model: Any, batch: Mapping[str, Any]) -> tuple[Any, Any]:
        excluded = {"response_mask", "advantages", "old_logprobs"}
        model_inputs = {key: value for key, value in batch.items() if key not in excluded}
        outputs = model(**model_inputs)
        response_mask = batch.get("response_mask")
        return causal_response_logprobs(outputs.logits, batch["input_ids"], response_mask=response_mask)

    def train_step(
        self,
        batch: Mapping[str, Any],
        *,
        advantages: Any,
        old_logprobs: Any | None = None,
    ) -> GRPOLossOutput:
        self.policy_model.train()
        policy_logprobs, action_mask = self.sequence_logprobs(self.policy_model, batch)
        with torch.no_grad():
            if old_logprobs is None:
                old_logprobs = policy_logprobs.detach()
            reference_logprobs, _ = self.sequence_logprobs(self.reference_model, batch)

        output = grpo_policy_loss(
            policy_logprobs=policy_logprobs,
            old_logprobs=old_logprobs,
            reference_logprobs=reference_logprobs,
            advantages=advantages,
            action_mask=action_mask,
            config=self.loss_config,
        )
        if hasattr(self.policy_model, "backward") and hasattr(self.policy_model, "step"):
            self.policy_model.backward(output.loss)
            self.policy_model.step()
        else:
            self.optimizer.zero_grad(set_to_none=True)
            output.loss.backward()
            self.optimizer.step()
        return output
