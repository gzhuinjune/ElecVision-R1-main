import unittest


try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from elecvision_r1.grpo_trainer import (
    ElecVisionGRPOTorchTrainer,
    GRPOLossConfig,
    causal_response_logprobs,
    grpo_policy_loss,
)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class GRPOTorchTrainerTests(unittest.TestCase):
    def test_grpo_policy_loss_is_finite(self):
        policy = torch.log_softmax(torch.randn(2, 4), dim=-1)
        old = policy.detach()
        reference = policy.detach() - 0.01
        advantages = torch.tensor([1.0, -1.0])
        mask = torch.ones_like(policy, dtype=torch.bool)
        output = grpo_policy_loss(
            policy_logprobs=policy,
            old_logprobs=old,
            reference_logprobs=reference,
            advantages=advantages,
            action_mask=mask,
            config=GRPOLossConfig(),
        )
        self.assertTrue(torch.isfinite(output.loss))

    def test_grpo_policy_loss_uses_sequence_level_weighting(self):
        policy = torch.zeros(2, 3)
        old = torch.zeros(2, 3)
        reference = torch.zeros(2, 3)
        advantages = torch.tensor([1.0, 3.0])
        mask = torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.bool)
        output = grpo_policy_loss(
            policy_logprobs=policy,
            old_logprobs=old,
            reference_logprobs=reference,
            advantages=advantages,
            action_mask=mask,
            config=GRPOLossConfig(kl_beta=0.0),
        )
        self.assertAlmostEqual(float(output.loss), -2.0)

    def test_trainer_updates_policy_parameters(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(16, 8)
                self.head = torch.nn.Linear(8, 16)

            def forward(self, input_ids, attention_mask=None, pixel_values=None):
                if pixel_values is not None:
                    self._received_pixel_values = True
                return type("Output", (), {"logits": self.head(self.embed(input_ids))})

        policy = TinyModel()
        reference = TinyModel()
        reference.load_state_dict(policy.state_dict())
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
        trainer = ElecVisionGRPOTorchTrainer(policy, reference, optimizer)
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4], [1, 2, 5, 6]], dtype=torch.long),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
            "response_mask": torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.bool),
            "pixel_values": torch.zeros(2, 3, 4, 4),
        }
        with torch.no_grad():
            old_logprobs, _ = trainer.sequence_logprobs(policy, batch)
        before = policy.head.weight.detach().clone()
        output = trainer.train_step(batch, advantages=torch.tensor([1.0, -1.0]), old_logprobs=old_logprobs)
        after = policy.head.weight.detach()
        self.assertTrue(torch.isfinite(output.loss))
        self.assertFalse(torch.equal(before, after))
        self.assertTrue(getattr(policy, "_received_pixel_values", False))

    def test_causal_response_mask_is_shifted(self):
        logits = torch.randn(1, 5, 16)
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        response_mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.bool)
        _, mask = causal_response_logprobs(logits, input_ids, response_mask=response_mask)
        self.assertEqual(mask.tolist(), [[False, True, True, True]])


if __name__ == "__main__":
    unittest.main()
