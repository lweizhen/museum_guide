from __future__ import annotations

import csv
from pathlib import Path

from crawl_names import OUTPUT_FIELDS, dedupe_items, strip_era_prefix

INPUT_CSV = Path("bowuguozhongguo_names_filtered.csv")


def main() -> None:
    with open(INPUT_CSV, "r", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    normalized_rows: list[dict[str, str]] = []
    for row in rows:
        normalized = {field: (row.get(field) or "").strip() for field in OUTPUT_FIELDS}
        normalized["name"] = strip_era_prefix(normalized["name"], normalized["era"])
        normalized_rows.append(normalized)

    normalized_rows = dedupe_items(normalized_rows)

    with open(INPUT_CSV, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(normalized_rows)

    print(f"已规范化 {len(normalized_rows)} 条展品名称 -> {INPUT_CSV}")


if __name__ == "__main__":
    main()
