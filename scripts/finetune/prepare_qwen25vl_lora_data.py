import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import argparse

from scripts.finetune.common import convert_lora_row, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert project LoRA JSONL into Qwen2.5-VL messages JSONL.")
    parser.add_argument("--input", default="data/multimodal_eval/train_lora.jsonl")
    parser.add_argument("--output", default="data/multimodal_eval/train_qwen25vl_messages.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-answer", action="store_true", help="Omit assistant answers for inference-style prompts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input, limit=args.limit)
    converted = [convert_lora_row(row, include_answer=not args.no_answer) for row in rows]
    write_jsonl(args.output, converted)
    print(f"Converted {len(converted)} rows -> {args.output}")


if __name__ == "__main__":
    main()
