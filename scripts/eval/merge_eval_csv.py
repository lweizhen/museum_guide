from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge evaluation CSV shard files with identical headers.")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input CSV files or glob patterns, for example outputs/raw/eval_*_s*.csv.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Merged output CSV path.",
    )
    parser.add_argument(
        "--dedupe-key",
        default="chain,image_id,question_idx",
        help="Comma-separated columns used to remove duplicate rows. Empty string disables de-duplication.",
    )
    return parser.parse_args()


def _expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(item) for item in glob.glob(pattern)]
        if matches:
            paths.extend(matches)
        else:
            paths.append(Path(pattern))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def main() -> None:
    args = parse_args()
    input_paths = _expand_inputs(args.inputs)
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input CSV files: " + ", ".join(missing))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] | None = None
    rows: list[dict[str, str]] = []
    dedupe_columns = [item.strip() for item in args.dedupe_key.split(",") if item.strip()]
    seen_keys: set[tuple[str, ...]] = set()

    for path in input_paths:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                continue
            if fieldnames is None:
                fieldnames = list(reader.fieldnames)
            elif list(reader.fieldnames) != fieldnames:
                raise ValueError(f"Header mismatch in {path}")

            for row in reader:
                if dedupe_columns:
                    key = tuple(row.get(column, "") for column in dedupe_columns)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                rows.append(row)

    if fieldnames is None:
        raise RuntimeError("No rows or headers found in input CSV files")

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Merged {len(input_paths)} files, {len(rows)} rows -> {output_path}")


if __name__ == "__main__":
    main()
