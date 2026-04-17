from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_INPUTS = [
    Path("data/exhibits.txt"),
    Path("data/exhibits_museumschina.txt"),
]
DEFAULT_OUTPUT = Path("data/exhibits_combined.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge the curated small KB and MuseumsChina KB into one text KB."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[str(path) for path in DEFAULT_INPUTS],
        help="Input KB text files. Artifacts must be separated by blank lines.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output merged KB path. Default: data/exhibits_combined.txt",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Deduplicate exact normalized blocks. Disabled by default to preserve same-name artifacts.",
    )
    return parser.parse_args()


def _load_blocks(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"知识库文件不存在：{path}")

    raw = path.read_text(encoding="utf-8").strip()
    blocks = [block.strip().replace("\n", " ") for block in raw.split("\n\n") if block.strip()]
    if not blocks:
        raise RuntimeError(f"知识库文件为空或格式不正确：{path}")
    return blocks


def _normalize_block(block: str) -> str:
    return re.sub(r"\s+", "", block)


def main() -> None:
    args = parse_args()

    merged: list[str] = []
    seen: set[str] = set()
    stats: list[tuple[str, int]] = []

    for input_path in [Path(item) for item in args.inputs]:
        blocks = _load_blocks(input_path)
        stats.append((str(input_path), len(blocks)))
        for block in blocks:
            if args.dedupe:
                key = _normalize_block(block)
                if key in seen:
                    continue
                seen.add(key)
            merged.append(block)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(merged) + "\n", encoding="utf-8")

    print("OK，已生成统一知识库：")
    for path, count in stats:
        print(f"- {path}: {count} 条")
    print(f"- output: {output_path}: {len(merged)} 条")


if __name__ == "__main__":
    main()
