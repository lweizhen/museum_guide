import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import argparse
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from scripts.finetune.common import read_jsonl, to_qwen25vl_messages


TARGET_MODULE_PRESETS: dict[str, list[str]] = {
    "qv": ["q_proj", "v_proj"],
    "qkvo": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qkvo_ffn": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def resolve_target_modules(target_mode: str, custom_modules: str) -> list[str]:
    if target_mode == "custom":
        modules = [item.strip() for item in custom_modules.split(",") if item.strip()]
        if not modules:
            raise RuntimeError("当 --target-mode custom 时，必须通过 --target-modules 提供至少一个模块名。")
        return modules
    if target_mode not in TARGET_MODULE_PRESETS:
        raise RuntimeError(f"不支持的 target_mode: {target_mode}")
    return TARGET_MODULE_PRESETS[target_mode]


try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )
    from transformers import Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
except ImportError as exc:  # pragma: no cover - dependency guard for local lightweight use
    raise RuntimeError(
        "Missing fine-tuning dependencies. Install: transformers peft accelerate "
        "bitsandbytes qwen-vl-utils torch"
    ) from exc


class LoraJsonlDataset(Dataset):
    def __init__(self, path: str, limit: int | None = None) -> None:
        self.rows = read_jsonl(path, limit=limit)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


@dataclass
class Qwen25VLCollator:
    processor: Any
    max_length: int
    max_pixels: int | None = None

    def _text_and_images(self, row: dict[str, Any], include_answer: bool) -> tuple[str, list[Any], list[Any] | None]:
        messages = to_qwen25vl_messages(row, include_answer=include_answer)
        if self.max_pixels:
            # qwen-vl-utils accepts image dicts with extra pixel constraints.
            image_item = messages[0]["content"][0]
            image_item["max_pixels"] = self.max_pixels
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not include_answer,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        return text, image_inputs or [], video_inputs

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        all_images: list[Any] = []
        all_videos: list[Any] = []
        prompt_lengths: list[int] = []

        for row in features:
            full_text, images, videos = self._text_and_images(row, include_answer=True)
            prompt_text, prompt_images, prompt_videos = self._text_and_images(row, include_answer=False)
            texts.append(full_text)
            all_images.extend(images)
            if videos:
                all_videos.extend(videos)

            prompt_inputs = self.processor(
                text=[prompt_text],
                images=prompt_images,
                videos=prompt_videos,
                return_tensors="pt",
                padding=False,
            )
            prompt_lengths.append(int(prompt_inputs["input_ids"].shape[1]))

        inputs = self.processor(
            text=texts,
            images=all_images or None,
            videos=all_videos or None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        labels = inputs["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        for row_idx, prompt_len in enumerate(prompt_lengths):
            labels[row_idx, : min(prompt_len, labels.shape[1])] = -100
        inputs["labels"] = labels
        return inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL with LoRA/QLoRA on museum guide data.")
    parser.add_argument("--model-path", required=True, help="HF model id or local path, e.g. /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--train-file", default="data/multimodal_eval/train_lora.jsonl")
    parser.add_argument("--val-file", default="data/multimodal_eval/val_lora.jsonl")
    parser.add_argument("--output-dir", default="outputs/lora/qwen25vl3b_museum")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-pixels", type=int, default=512 * 512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-mode", default="qkvo_ffn", choices=["qv", "qkvo", "qkvo_ffn", "custom"], help="Choose which projection modules receive LoRA adapters.")
    parser.add_argument("--target-modules", default="", help="Comma-separated module names when --target-mode custom is used.")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit QLoRA loading.")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dataset = LoraJsonlDataset(args.train_file, limit=args.max_train_samples)
    val_dataset = None
    if Path(args.val_file).exists() and Path(args.val_file).stat().st_size > 0:
        val_rows = read_jsonl(args.val_file, limit=args.max_val_samples)
        if val_rows:
            val_dataset = LoraJsonlDataset(args.val_file, limit=args.max_val_samples)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    quantization_config = None
    if not args.no_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    target_modules = resolve_target_modules(args.target_mode, args.target_modules)
    print(f"LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_config = {
        "model_path": args.model_path,
        "train_file": args.train_file,
        "val_file": args.val_file,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "max_pixels": args.max_pixels,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_mode": args.target_mode,
        "target_modules": target_modules,
        "use_4bit": not args.no_4bit,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "lora_train_config.json").write_text(
        __import__("json").dumps(train_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    training_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=args.report_to,
        remove_unused_columns=False,
        seed=args.seed,
    )
    strategy_key = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    training_kwargs[strategy_key] = "steps" if val_dataset is not None else "no"
    if val_dataset is not None:
        training_kwargs["eval_steps"] = args.save_steps
    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=Qwen25VLCollator(processor=processor, max_length=args.max_length, max_pixels=args.max_pixels),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter and processor to {args.output_dir}")


if __name__ == "__main__":
    main()
