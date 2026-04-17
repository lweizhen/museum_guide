
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_image_path(image_path: str | Path) -> str:
    path = Path(image_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return str(path)


def to_qwen25vl_messages(row: dict[str, Any], include_answer: bool = True) -> list[dict[str, Any]]:
    image_path = resolve_image_path(str(row["image_path"]))
    instruction = str(row.get("instruction", ""))
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": str(row.get("output", ""))})
    return messages


def convert_lora_row(row: dict[str, Any], include_answer: bool = True) -> dict[str, Any]:
    return {
        "artifact_id": row.get("artifact_id", ""),
        "artifact_name": row.get("artifact_name", ""),
        "image_id": row.get("image_id", ""),
        "task": row.get("task", ""),
        "split": row.get("split", ""),
        "messages": to_qwen25vl_messages(row, include_answer=include_answer),
    }
