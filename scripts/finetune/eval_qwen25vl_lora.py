import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from scripts.finetune.common import read_jsonl, to_qwen25vl_messages


try:
    from peft import PeftModel
    from transformers import AutoProcessor
    from transformers import Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing inference dependencies. Install: transformers peft accelerate qwen-vl-utils torch"
    ) from exc


def normalize_text(text: str) -> str:
    return "".join(str(text).lower().split())


def lcs_len(a: str, b: str) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        curr = [0]
        for j, cb in enumerate(b, 1):
            curr.append(prev[j - 1] + 1 if ca == cb else max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(reference: str, prediction: str) -> float:
    ref = normalize_text(reference)
    pred = normalize_text(prediction)
    if not ref or not pred:
        return 0.0
    lcs = lcs_len(ref, pred)
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def generate_one(
    model: Any,
    processor: Any,
    row: dict[str, Any],
    max_new_tokens: int,
    max_pixels: int | None,
) -> str:
    messages = to_qwen25vl_messages(row, include_answer=False)
    if max_pixels:
        messages[0]["content"][0]["max_pixels"] = max_pixels
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    output_ids = generated[:, prompt_len:]
    return processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Qwen2.5-VL LoRA adapter on closed-set museum guide data.")
    parser.add_argument("--model-path", required=True, help="Base HF model path or id.")
    parser.add_argument("--adapter-path", default=None, help="LoRA adapter path. Omit to evaluate base model.")
    parser.add_argument("--test-file", default="data/multimodal_eval/test_lora.jsonl")
    parser.add_argument("--output", default="outputs/finetune/eval_qwen25vl_lora_results.csv")
    parser.add_argument("--summary", default="outputs/finetune/eval_qwen25vl_lora_summary.txt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-pixels", type=int, default=512 * 512)
    parser.add_argument("--bf16", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.test_file, limit=args.limit)
    processor_path = args.adapter_path or args.model_path
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    counters: Counter[str] = Counter()
    task_counters: dict[str, Counter[str]] = defaultdict(Counter)
    rouge_sum = 0.0
    task_rouge_sum: dict[str, float] = defaultdict(float)

    fieldnames = [
        "artifact_id",
        "artifact_name",
        "image_id",
        "task",
        "instruction",
        "reference",
        "prediction",
        "name_mentioned",
        "rouge_l_f1",
    ]
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows, 1):
            prediction = generate_one(model, processor, row, args.max_new_tokens, args.max_pixels)
            reference = str(row.get("output", ""))
            artifact_name = str(row.get("artifact_name", ""))
            task = str(row.get("task", ""))
            name_mentioned = int(bool(artifact_name) and artifact_name in prediction)
            rouge = rouge_l_f1(reference, prediction)

            counters["total"] += 1
            counters["name_mentioned"] += name_mentioned
            rouge_sum += rouge
            task_counters[task]["total"] += 1
            task_counters[task]["name_mentioned"] += name_mentioned
            task_rouge_sum[task] += rouge

            writer.writerow(
                {
                    "artifact_id": row.get("artifact_id", ""),
                    "artifact_name": artifact_name,
                    "image_id": row.get("image_id", ""),
                    "task": task,
                    "instruction": row.get("instruction", ""),
                    "reference": reference,
                    "prediction": prediction,
                    "name_mentioned": name_mentioned,
                    "rouge_l_f1": f"{rouge:.4f}",
                }
            )
            print(f"{idx}/{len(rows)} task={task} name_mentioned={name_mentioned} rouge_l={rouge:.4f}")

    total = counters["total"]
    lines = [
        f"rows: {total}",
        f"name mention rate: {counters['name_mentioned'] / total if total else 0:.4f}",
        f"avg rouge-l f1: {rouge_sum / total if total else 0:.4f}",
        "",
        "by task:",
    ]
    for task in sorted(task_counters):
        task_total = task_counters[task]["total"]
        lines.append(
            f"- {task}: rows={task_total}, "
            f"name_mention={task_counters[task]['name_mentioned'] / task_total if task_total else 0:.4f}, "
            f"rouge_l={task_rouge_sum[task] / task_total if task_total else 0:.4f}"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Results written to: {out_path}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
