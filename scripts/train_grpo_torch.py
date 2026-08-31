from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elecvision_r1.grpo import ElecVisionGRPOScorer, load_jsonl
from elecvision_r1.grpo_trainer import ElecVisionGRPOTorchTrainer, GRPOLossConfig


def _freeze_vision_encoder(model: Any) -> None:
    vision_markers = ("vision", "visual", "vit", "vision_tower", "image_tower")
    for name, parameter in model.named_parameters():
        if any(marker in name.lower() for marker in vision_markers):
            parameter.requires_grad_(False)


def _apply_lora(model: Any, *, r: int, alpha: int, dropout: float, target_modules: str) -> Any:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:  # pragma: no cover - optional paper-scale dependency.
        raise ImportError("Install peft to use --lora for the paper SFT/GRPO setup.") from exc

    targets = [item.strip() for item in target_modules.split(",") if item.strip()] or None
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=targets,
    )
    return get_peft_model(model, config)


def _unwrap_model(model: Any) -> Any:
    return getattr(model, "module", model)


def _refresh_reference_model(torch: Any, policy_model: Any, reference_model: Any, device: Any) -> None:
    source = _unwrap_model(policy_model)
    target = _unwrap_model(reference_model)
    target.load_state_dict(source.state_dict(), strict=True)
    target.to(device)
    target.eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)


def _maybe_initialize_deepspeed(policy_model: Any, optimizer: Any, config_path: str):
    if not config_path:
        return policy_model, optimizer
    try:
        import deepspeed
    except ImportError as exc:  # pragma: no cover - optional paper-scale dependency.
        raise ImportError("Install deepspeed to use --deepspeed and ZeRO training.") from exc

    with Path(config_path).open("r", encoding="utf-8") as handle:
        ds_config = json.load(handle)
    engine, optimizer, _, _ = deepspeed.initialize(model=policy_model, optimizer=optimizer, config=ds_config)
    return engine, optimizer


def _load_training_stack(
    model_name_or_path: str,
    learning_rate: float,
    model_class: str,
    *,
    bf16: bool = False,
    lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "",
    freeze_vision_encoder: bool = False,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    processor = None
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    except Exception:
        if model_class == "vision-language":
            raise

    tokenizer = getattr(processor, "tokenizer", None) if processor is not None else None
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {}
    if bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16

    if model_class == "vision-language":
        try:
            from transformers import AutoModelForVision2Seq
        except ImportError as exc:  # pragma: no cover - depends on transformers version.
            raise ImportError("AutoModelForVision2Seq is required for --model-class vision-language") from exc
        policy_model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, trust_remote_code=True, **model_kwargs)
    elif model_class == "causal-lm":
        policy_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, **model_kwargs)
    else:
        try:
            from transformers import AutoModelForVision2Seq

            policy_model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, trust_remote_code=True, **model_kwargs)
        except Exception:
            policy_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, **model_kwargs)

    if freeze_vision_encoder:
        _freeze_vision_encoder(policy_model)
    if lora:
        policy_model = _apply_lora(
            policy_model,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=lora_target_modules,
        )

    reference_model = copy.deepcopy(policy_model)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
    return torch, processor, tokenizer, policy_model, reference_model, optimizer


def _messages_to_text(messages: list[Any]) -> str:
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            parts.append(str(message))
            continue
        role = str(message.get("role", "")).strip()
        content = message.get("content", "")
        if isinstance(content, list):
            content_text = " ".join(str(item.get("text", item)) if isinstance(item, Mapping) else str(item) for item in content)
        else:
            content_text = str(content)
        parts.append(f"{role}: {content_text}" if role else content_text)
    return "\n".join(part for part in parts if part.strip()).strip()


def _row_prompt(row: Mapping[str, Any], processor: Any | None = None, tokenizer: Any | None = None) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        templater = processor if hasattr(processor, "apply_chat_template") else tokenizer
        if templater is not None and hasattr(templater, "apply_chat_template"):
            return templater.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return _messages_to_text(messages)
    return str(row.get("prompt", row.get("instruction", row.get("query", "")))).strip()


def _row_image_path(row: Mapping[str, Any], base_dir: Path) -> Path | None:
    raw = row.get("image", row.get("image_path", row.get("image_file", "")))
    if not raw:
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = base_dir / path
    return path


def _load_image(path: Path | None):
    if path is None:
        return None
    from PIL import Image

    return Image.open(path).convert("RGB")


def _processor_call(processor: Any, tokenizer: Any, *, texts: list[str], images: list[Any], max_length: int):
    active_images = [image for image in images if image is not None]
    if active_images:
        if processor is None:
            raise ValueError("A Hugging Face processor is required for image batches.")
        if len(active_images) != len(images):
            raise ValueError("A multimodal batch must provide an image for every record.")
        return processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def _prompt_length(processor: Any, tokenizer: Any, prompt: str, image: Any, max_length: int) -> int:
    if image is not None:
        encoded = processor(text=[prompt], images=[image], truncation=True, max_length=max_length, return_tensors="pt")
        return int(encoded["input_ids"].shape[-1])
    return len(tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"])


def _decode_generated_completions(torch, tokenizer: Any, generated: Any, input_ids: Any) -> list[str]:
    prompt_tokens = int(input_ids.shape[-1])
    if generated.shape[-1] > prompt_tokens:
        repeated_input_ids = input_ids.repeat_interleave(max(1, generated.shape[0] // input_ids.shape[0]), dim=0)
        if repeated_input_ids.shape[0] == generated.shape[0] and torch.equal(generated[:, :prompt_tokens], repeated_input_ids):
            generated = generated[:, prompt_tokens:]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def _batch_encode(torch, processor, tokenizer, items: list[Mapping[str, Any]], *, max_length: int, device):
    texts = [str(item["text"]) for item in items]
    prompts = [str(item["prompt"]) for item in items]
    images = [item.get("image") for item in items]
    encoded = _processor_call(processor, tokenizer, texts=texts, images=images, max_length=max_length)
    encoded = {key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()}
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for index, (prompt, image) in enumerate(zip(prompts, images)):
        prompt_length = _prompt_length(processor, tokenizer, prompt, image, max_length)
        response_mask[index, prompt_length:] = attention_mask[index, prompt_length:].to(dtype=torch.bool)
    encoded["response_mask"] = response_mask
    return encoded


def _generate_completions(
    *,
    torch,
    processor,
    tokenizer,
    model,
    row: Mapping[str, Any],
    image,
    max_length: int,
    max_new_tokens: int,
    num_generations: int,
    temperature: float,
    device,
) -> list[str]:
    prompt = _row_prompt(row, processor=processor, tokenizer=tokenizer)
    encoded = _processor_call(processor, tokenizer, texts=[prompt], images=[image], max_length=max_length)
    encoded = {key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()}
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=num_generations,
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    return _decode_generated_completions(torch, tokenizer, generated, encoded["input_ids"])


def _group_items(
    *,
    torch,
    processor,
    tokenizer,
    model,
    rows: list[Mapping[str, Any]],
    base_dir: Path,
    scorer: ElecVisionGRPOScorer,
    max_length: int,
    max_new_tokens: int,
    num_generations: int,
    temperature: float,
    device,
) -> list[dict[str, Any]]:
    prepared_rows: list[dict[str, Any]] = []
    for row in rows:
        prepared = dict(row)
        image = _load_image(_row_image_path(row, base_dir))
        if not prepared.get("completions"):
            prepared["completions"] = _generate_completions(
                torch=torch,
                processor=processor,
                tokenizer=tokenizer,
                model=model,
                row=row,
                image=image,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                num_generations=num_generations,
                temperature=temperature,
                device=device,
            )
        prepared["_image_obj"] = image
        prepared_rows.append(prepared)

    groups = scorer.score_records(prepared_rows)
    items: list[dict[str, Any]] = []
    for row, group in zip(prepared_rows, groups):
        prompt = _row_prompt(row, processor=processor, tokenizer=tokenizer)
        image = row.get("_image_obj")
        for score in group.scores:
            completion = score.completion.strip()
            items.append(
                {
                    "prompt": prompt,
                    "text": (prompt + "\n" + completion).strip(),
                    "image": image,
                    "advantage": float(score.advantage),
                }
            )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ElecVision-GRPO with the PyTorch objective.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-class", choices=["auto", "causal-lm", "vision-language"], default="auto")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--optimization-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.04)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--deepspeed", default="", help="Path to a DeepSpeed config, e.g. configs/deepspeed_zero3_bf16.json.")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="")
    parser.add_argument("--freeze-vision-encoder", action="store_true")
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    torch, processor, tokenizer, policy_model, reference_model, optimizer = _load_training_stack(
        args.model,
        args.learning_rate,
        args.model_class,
        bf16=args.bf16,
        lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        freeze_vision_encoder=args.freeze_vision_encoder,
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy_model.to(device)
    reference_model.to(device)
    policy_model, optimizer = _maybe_initialize_deepspeed(policy_model, optimizer, args.deepspeed)

    scorer = ElecVisionGRPOScorer()
    train_path = Path(args.train_jsonl)
    rows = load_jsonl(train_path)
    trainer = ElecVisionGRPOTorchTrainer(
        policy_model,
        reference_model,
        optimizer,
        loss_config=GRPOLossConfig(clip_range=args.clip_range, kl_beta=args.kl_beta),
    )

    for _ in range(args.iterations):
        _refresh_reference_model(torch, policy_model, reference_model, device)
        items = _group_items(
            torch=torch,
            processor=processor,
            tokenizer=tokenizer,
            model=_unwrap_model(policy_model),
            rows=rows,
            base_dir=train_path.resolve().parent,
            scorer=scorer,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            num_generations=args.num_generations,
            temperature=args.temperature,
            device=device,
        )
        for start in range(0, len(items), args.batch_size):
            batch_items = items[start : start + args.batch_size]
            batch = _batch_encode(torch, processor, tokenizer, batch_items, max_length=args.max_length, device=device)
            advantages = torch.tensor([item["advantage"] for item in batch_items], dtype=torch.float32, device=device)
            with torch.no_grad():
                old_logprobs, _ = trainer.sequence_logprobs(policy_model, batch)
                old_logprobs = old_logprobs.detach()
            for _ in range(args.optimization_steps):
                output = trainer.train_step(batch, advantages=advantages, old_logprobs=old_logprobs)
                print(
                    {
                        "loss": round(float(output.loss.detach().cpu()), 6),
                        "policy_loss": round(float(output.policy_loss.cpu()), 6),
                        "kl_loss": round(float(output.kl_loss.cpu()), 6),
                        "mean_ratio": round(float(output.mean_ratio.cpu()), 6),
                    }
                )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _unwrap_model(policy_model).save_pretrained(output_dir)
    if processor is not None and hasattr(processor, "save_pretrained"):
        processor.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
