from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_grpo_torch import _apply_lora, _freeze_vision_encoder, _row_image_path, _row_prompt


class SFTJsonlDataset:
    def __init__(
        self,
        *,
        torch: Any,
        processor: Any | None,
        tokenizer: Any,
        rows: list[Mapping[str, Any]],
        base_dir: Path,
        max_length: int,
        mask_prompt_labels: bool,
    ) -> None:
        self.torch = torch
        self.processor = processor
        self.tokenizer = tokenizer
        self.rows = rows
        self.base_dir = base_dir
        self.max_length = max_length
        self.mask_prompt_labels = mask_prompt_labels

    def __len__(self) -> int:
        return len(self.rows)

    def _response_text(self, row: Mapping[str, Any]) -> str:
        response = row.get("response", row.get("completion", row.get("answer", row.get("solution", ""))))
        return str(response).strip()

    def _load_image(self, row: Mapping[str, Any]):
        image_path = _row_image_path(row, self.base_dir)
        if image_path is None:
            return None
        from PIL import Image

        return Image.open(image_path).convert("RGB")

    def _encode(self, text: str, image: Any | None = None) -> dict[str, Any]:
        kwargs = {
            "padding": "max_length",
            "truncation": True,
            "max_length": self.max_length,
            "return_tensors": "pt",
        }
        if image is not None:
            if self.processor is None:
                raise ValueError("A Hugging Face processor is required when SFT rows contain images.")
            return self.processor(text=[text], images=[image], **kwargs)
        return self.tokenizer([text], **kwargs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        prompt = _row_prompt(row, processor=self.processor, tokenizer=self.tokenizer)
        response = self._response_text(row)
        text = (prompt + "\n" + response).strip()
        image = self._load_image(row)
        encoded = self._encode(text, image)
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        labels = item["input_ids"].clone()

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            labels = labels.masked_fill(item["input_ids"] == pad_token_id, -100)
        if self.mask_prompt_labels:
            prompt_encoded = self._encode(prompt, image)
            prompt_len = int(prompt_encoded["attention_mask"].sum().item())
            labels[:prompt_len] = -100

        item["labels"] = labels
        return item


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append(payload)
    return rows


def _load_sft_stack(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    processor = None
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    except Exception:
        if args.model_class == "vision-language":
            raise

    tokenizer = getattr(processor, "tokenizer", None) if processor is not None else None
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {}
    if args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16

    if args.model_class == "vision-language":
        from transformers import AutoModelForVision2Seq

        model = AutoModelForVision2Seq.from_pretrained(args.model, trust_remote_code=True, **model_kwargs)
    elif args.model_class == "causal-lm":
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, **model_kwargs)
    else:
        try:
            from transformers import AutoModelForVision2Seq

            model = AutoModelForVision2Seq.from_pretrained(args.model, trust_remote_code=True, **model_kwargs)
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, **model_kwargs)

    if args.freeze_vision_encoder:
        _freeze_vision_encoder(model)
    if args.lora:
        model = _apply_lora(
            model,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )
    return torch, processor, tokenizer, model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ElecVision-R1 supervised fine-tuning stage.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-class", choices=["auto", "causal-lm", "vision-language"], default="auto")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--deepspeed", default="", help="Path to a DeepSpeed config, e.g. configs/deepspeed_zero3_bf16.json.")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="")
    parser.add_argument("--freeze-vision-encoder", action="store_true")
    parser.add_argument("--mask-prompt-labels", action="store_true", default=True)
    args = parser.parse_args()

    torch, processor, tokenizer, model = _load_sft_stack(args)
    train_path = Path(args.train_jsonl)
    rows = _load_jsonl(train_path)
    dataset = SFTJsonlDataset(
        torch=torch,
        processor=processor,
        tokenizer=tokenizer,
        rows=rows,
        base_dir=train_path.resolve().parent,
        max_length=args.max_length,
        mask_prompt_labels=args.mask_prompt_labels,
    )

    from transformers import Trainer, TrainingArguments, default_data_collator

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        deepspeed=args.deepspeed or None,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    if processor is not None and hasattr(processor, "save_pretrained"):
        processor.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
