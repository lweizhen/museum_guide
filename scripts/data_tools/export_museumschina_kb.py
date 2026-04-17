from __future__ import annotations

import csv
import json
from pathlib import Path

INPUT_CSV = Path("bowuguozhongguo_names_filtered.csv")
SUPPLEMENT_JSON = Path("data/museumschina_manual_supplements.json")
OUTPUT_TXT = Path("data/exhibits_museumschina.txt")

STANDARD_FIELDS = [
    ("name", "展品名称"),
    ("era", "所属时代"),
    ("find_spot", "出土地"),
    ("historical_significance", "历史意义"),
    ("supplement_info", "补充信息"),
    ("shape_and_style", "纹饰与造型"),
    ("historical_background", "历史背景"),
    ("story", "故事传说"),
]


def normalize_value(value: str) -> str:
    return " ".join((value or "").strip().split())


def load_rows() -> list[dict[str, str]]:
    with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_supplements() -> dict[str, dict[str, object]]:
    if not SUPPLEMENT_JSON.exists():
        return {}
    return json.loads(SUPPLEMENT_JSON.read_text(encoding="utf-8"))


def build_supplement_info(row: dict[str, str], supplement: dict[str, object]) -> str:
    parts: list[str] = []

    existing = normalize_value(str(supplement.get("supplement_info", "")))
    if existing:
        parts.append(existing)

    metadata_pairs = [
        ("类别", row.get("category", "")),
        ("收藏单位", row.get("museum", "")),
        ("级别", row.get("level", "")),
        ("入藏年度", row.get("accession_year", "")),
        ("质地", row.get("material", "")),
    ]
    meta = [f"{label}：{normalize_value(value)}" for label, value in metadata_pairs if normalize_value(value)]
    if meta:
        parts.append("馆藏元数据：" + "；".join(meta) + "。")

    verification_note = normalize_value(str(supplement.get("verification_note", "")))
    if verification_note:
        parts.append("校验说明：" + verification_note)

    return " ".join(parts)


def build_story(supplement: dict[str, object]) -> str:
    return normalize_value(str(supplement.get("story", "")))


def format_sources(row: dict[str, str], supplement: dict[str, object]) -> list[str]:
    lines: list[str] = []

    detail_url = normalize_value(row.get("detail_url", ""))
    source = normalize_value(row.get("source", ""))
    sources = [item for item in [detail_url, source] if item]
    if sources:
        lines.append(f"【数据来源】{' | '.join(sources)}")

    supplement_sources = supplement.get("sources", [])
    if isinstance(supplement_sources, list):
        cleaned: list[str] = []
        for item in supplement_sources:
            if not isinstance(item, dict):
                continue
            title = normalize_value(str(item.get("title", "")))
            url = normalize_value(str(item.get("url", "")))
            if title and url:
                cleaned.append(f"{title}：{url}")
            elif url:
                cleaned.append(url)
        if cleaned:
            lines.append(f"【补充来源】{' | '.join(cleaned)}")

    return lines


def format_record(row: dict[str, str], supplements: dict[str, dict[str, object]]) -> str:
    detail_url = normalize_value(row.get("detail_url", ""))
    supplement = supplements.get(detail_url, {})

    field_values = {
        "name": row.get("name", ""),
        "era": row.get("era", ""),
        "find_spot": supplement.get("find_spot", ""),
        "historical_significance": supplement.get("historical_significance", ""),
        "supplement_info": build_supplement_info(row, supplement),
        "shape_and_style": supplement.get("shape_and_style", ""),
        "historical_background": supplement.get("historical_background", ""),
        "story": build_story(supplement),
    }

    lines: list[str] = []
    for field, label in STANDARD_FIELDS:
        value = normalize_value(str(field_values.get(field, "")))
        if value:
            lines.append(f"【{label}】{value}")

    lines.extend(format_sources(row, supplement))
    return "\n".join(lines)


def main() -> None:
    rows = load_rows()
    supplements = load_supplements()

    records = [format_record(row, supplements) for row in rows]
    records = [record for record in records if record.strip()]

    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TXT.write_text("\n\n".join(records), encoding="utf-8")

    print(
        f"已导出 {len(records)} 条知识库记录 -> {OUTPUT_TXT} "
        f"(人工校验补充 {len(supplements)} 条)"
    )


if __name__ == "__main__":
    main()
