from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.config import DATA_PATH, IMAGE_META_PATH


OUT_DIR = Path("data/multimodal_eval")
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.1

FACT_FIELD_ORDER = [
    "展品名称",
    "类别",
    "所属时代",
    "馆藏单位",
    "材质",
    "出土地",
    "功能用途",
    "历史意义",
    "文化价值",
    "纹饰与造型",
    "历史背景",
    "补充信息",
    "故事传说",
    "数据来源",
    "补充来源",
]

DESCRIPTION_FIELDS = [
    "功能用途",
    "历史意义",
    "文化价值",
    "纹饰与造型",
    "历史背景",
    "补充信息",
    "故事传说",
]

QA_FIELD_SPECS = [
    ("展品名称", "这件文物叫什么名字？"),
    ("所属时代", "这件文物属于什么时代？"),
    ("类别", "这件文物属于什么类别？"),
    ("馆藏单位", "这件文物收藏于哪里？"),
    ("材质", "这件文物主要材质是什么？"),
    ("功能用途", "这件文物的主要功能或用途是什么？"),
    ("历史意义", "这件文物有什么历史意义？"),
    ("文化价值", "这件文物体现了怎样的文化价值？"),
    ("纹饰与造型", "这件文物在纹饰或造型上有什么特点？"),
    ("历史背景", "这件文物反映了怎样的历史背景？"),
    ("故事传说", "这件文物相关的故事传说是什么？"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a unified multimodal evaluation dataset from the current museum "
            "text KB and image metadata."
        )
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="Directory to write multimodal dataset files.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Artifact-level train split ratio. Default: 0.7",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Artifact-level validation split ratio. Default: 0.1",
    )
    parser.add_argument(
        "--limit-artifacts",
        type=int,
        default=0,
        help="Only export the first N matched artifacts after sorting. 0 means all.",
    )
    return parser.parse_args()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip()).lower()


def _artifact_key(name: str, era: str, museum: str) -> str:
    return "|".join([_normalize(name), _normalize(era), _normalize(museum)])


def _split_ratio_key(key: str) -> float:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _parse_kb_blocks(path: str) -> list[dict[str, object]]:
    raw = Path(path).read_text(encoding="utf-8").strip()
    if not raw:
        raise RuntimeError(f"知识库文件为空: {path}")

    blocks = [block.strip() for block in re.split(r"\n\s*\n", raw) if block.strip()]
    parsed: list[dict[str, object]] = []
    for index, block in enumerate(blocks, start=1):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^【(.+?)】\s*(.*)$", line)
            if match:
                fields[match.group(1).strip()] = match.group(2).strip()
        parsed.append(
            {
                "block_id": index,
                "raw_text": block,
                "fields": fields,
            }
        )
    return parsed


def _load_image_meta(path: str) -> list[dict[str, str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_reference_description(fields: dict[str, str]) -> str:
    lead_parts = [
        fields.get("展品名称", "").strip(),
        fields.get("所属时代", "").strip(),
        fields.get("类别", "").strip(),
        fields.get("馆藏单位", "").strip(),
    ]
    lead_parts = [part for part in lead_parts if part]
    lead = "，".join(lead_parts)

    detail_sentences: list[str] = []
    for label in DESCRIPTION_FIELDS:
        value = fields.get(label, "").strip()
        if value:
            detail_sentences.append(value)

    if lead and detail_sentences:
        return f"{lead}。{' '.join(detail_sentences)}"
    if lead:
        return lead
    return " ".join(detail_sentences)


def _build_qa_pairs(fields: dict[str, str]) -> list[dict[str, str]]:
    qa_pairs: list[dict[str, str]] = []
    for field_name, question in QA_FIELD_SPECS:
        answer = fields.get(field_name, "").strip()
        if not answer:
            continue
        qa_pairs.append(
            {
                "task": "qa",
                "question": question,
                "answer": answer,
                "answer_field": field_name,
                "question_type": "open",
            }
        )
    return qa_pairs


def _build_fact_dict(fields: dict[str, str]) -> dict[str, str]:
    ordered: dict[str, str] = {}
    for label in FACT_FIELD_ORDER:
        value = fields.get(label, "").strip()
        if value:
            ordered[label] = value

    for label, value in fields.items():
        if label not in ordered and value.strip():
            ordered[label] = value.strip()
    return ordered


def _group_images_by_artifact(image_meta: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in image_meta:
        detail_url = (item.get("detail_url") or "").strip()
        name = (item.get("name") or "").strip()
        era = (item.get("era") or "").strip()
        museum = (item.get("museum") or "").strip()
        key = detail_url or _artifact_key(name, era, museum)
        grouped[key].append(item)
    return grouped


def _index_kb_blocks(
    kb_blocks: list[dict[str, object]],
) -> tuple[dict[str, list[dict[str, object]]], dict[str, list[dict[str, object]]], dict[str, list[dict[str, object]]]]:
    by_triple: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_name_era: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_name: dict[str, list[dict[str, object]]] = defaultdict(list)

    for block in kb_blocks:
        fields = block["fields"]  # type: ignore[index]
        name = str(fields.get("展品名称", "")).strip()
        era = str(fields.get("所属时代", "")).strip()
        museum = str(fields.get("馆藏单位", "")).strip()

        by_triple[_artifact_key(name, era, museum)].append(block)
        by_name_era["|".join([_normalize(name), _normalize(era)])].append(block)
        by_name[_normalize(name)].append(block)

    return by_triple, by_name_era, by_name


def _choose_best_block(
    candidates: list[dict[str, object]],
    museum: str,
) -> dict[str, object]:
    normalized_museum = _normalize(museum)

    def sort_key(block: dict[str, object]) -> tuple[int, int]:
        fields = block["fields"]  # type: ignore[index]
        block_museum = _normalize(str(fields.get("馆藏单位", "")))
        museum_match = 1 if normalized_museum and block_museum == normalized_museum else 0
        filled_fields = sum(1 for value in fields.values() if str(value).strip())
        return museum_match, filled_fields

    return max(candidates, key=sort_key)


def _match_kb_block(
    first_image: dict[str, str],
    by_triple: dict[str, list[dict[str, object]]],
    by_name_era: dict[str, list[dict[str, object]]],
    by_name: dict[str, list[dict[str, object]]],
) -> tuple[dict[str, object] | None, str]:
    name = (first_image.get("name") or "").strip()
    era = (first_image.get("era") or "").strip()
    museum = (first_image.get("museum") or "").strip()

    triple_key = _artifact_key(name, era, museum)
    triple_candidates = by_triple.get(triple_key, [])
    if triple_candidates:
        return _choose_best_block(triple_candidates, museum), "name+era+museum"

    name_era_key = "|".join([_normalize(name), _normalize(era)])
    name_era_candidates = by_name_era.get(name_era_key, [])
    if name_era_candidates:
        return _choose_best_block(name_era_candidates, museum), "name+era"

    name_candidates = by_name.get(_normalize(name), [])
    if name_candidates:
        return _choose_best_block(name_candidates, museum), "name"

    return None, ""


def _determine_split(key: str, train_ratio: float, val_ratio: float) -> str:
    value = _split_ratio_key(key)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def _build_samples(
    image_groups: dict[str, list[dict[str, str]]],
    kb_blocks: list[dict[str, object]],
    train_ratio: float,
    val_ratio: float,
    limit_artifacts: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_triple, by_name_era, by_name = _index_kb_blocks(kb_blocks)

    samples: list[dict[str, object]] = []
    unmatched: list[dict[str, str]] = []
    split_counter: Counter[str] = Counter()
    match_counter: Counter[str] = Counter()

    items = sorted(image_groups.items(), key=lambda pair: pair[0])
    if limit_artifacts > 0:
        items = items[:limit_artifacts]

    for artifact_idx, (artifact_group_key, images) in enumerate(items, start=1):
        first_image = images[0]
        block, match_strategy = _match_kb_block(first_image, by_triple, by_name_era, by_name)
        if block is None:
            unmatched.append(
                {
                    "group_key": artifact_group_key,
                    "name": first_image.get("name", ""),
                    "era": first_image.get("era", ""),
                    "museum": first_image.get("museum", ""),
                    "detail_url": first_image.get("detail_url", ""),
                }
            )
            continue

        fields = dict(block["fields"])  # type: ignore[arg-type]
        reference_description = _build_reference_description(fields)
        qa_pairs = _build_qa_pairs(fields)
        facts = _build_fact_dict(fields)
        artifact_key = artifact_group_key or _artifact_key(
            first_image.get("name", ""),
            first_image.get("era", ""),
            first_image.get("museum", ""),
        )
        split = _determine_split(artifact_key, train_ratio, val_ratio)
        split_counter[split] += 1
        match_counter[match_strategy] += 1

        image_samples: list[dict[str, object]] = []
        for image_idx, image_item in enumerate(images, start=1):
            local_path = str(image_item.get("local_path", "")).replace("\\", "/")
            image_samples.append(
                {
                    "image_id": f"{artifact_idx:04d}_{image_idx:02d}",
                    "image_path": local_path,
                    "image_url": image_item.get("image_url", ""),
                }
            )

        samples.append(
            {
                "artifact_id": f"artifact_{artifact_idx:04d}",
                "artifact_key": artifact_key,
                "split": split,
                "match_strategy": match_strategy,
                "artifact_name": fields.get("展品名称", first_image.get("name", "")),
                "era": fields.get("所属时代", first_image.get("era", "")),
                "museum": fields.get("馆藏单位", first_image.get("museum", "")),
                "category": fields.get("类别", first_image.get("category", "")),
                "detail_url": first_image.get("detail_url", ""),
                "source": first_image.get("source", ""),
                "source_urls": [
                    url
                    for url in [first_image.get("detail_url", ""), fields.get("数据来源", ""), fields.get("补充来源", "")]
                    if url
                ],
                "reference_description": reference_description,
                "reference_facts": facts,
                "qa_pairs": qa_pairs,
                "images": image_samples,
                "kb_block_text": str(block["raw_text"]),
            }
        )

    summary = {
        "artifact_count": len(samples),
        "image_count": sum(len(sample["images"]) for sample in samples),
        "split_counts": dict(split_counter),
        "match_strategy_counts": dict(match_counter),
        "unmatched_count": len(unmatched),
        "unmatched_examples": unmatched[:20],
    }
    return samples, summary


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_split_files(out_dir: Path, samples: list[dict[str, object]]) -> None:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    image_level_grouped: dict[str, list[dict[str, object]]] = defaultdict(list)

    for sample in samples:
        split = str(sample["split"])
        grouped[split].append(sample)

        for image in sample["images"]:  # type: ignore[index]
            image_level_grouped[split].append(
                {
                    "artifact_id": sample["artifact_id"],
                    "artifact_name": sample["artifact_name"],
                    "era": sample["era"],
                    "museum": sample["museum"],
                    "category": sample["category"],
                    "detail_url": sample["detail_url"],
                    "source": sample["source"],
                    "source_urls": sample["source_urls"],
                    "split": split,
                    "match_strategy": sample["match_strategy"],
                    "image_id": image["image_id"],
                    "image_path": image["image_path"],
                    "image_url": image["image_url"],
                    "reference_description": sample["reference_description"],
                    "reference_facts": sample["reference_facts"],
                    "qa_pairs": sample["qa_pairs"],
                }
            )

    for split_name in ("train", "val", "test"):
        _write_jsonl(out_dir / f"{split_name}.jsonl", grouped.get(split_name, []))
        _write_jsonl(
            out_dir / f"{split_name}_images.jsonl",
            image_level_grouped.get(split_name, []),
        )


def main() -> None:
    args = parse_args()
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise RuntimeError("划分比例非法，请确保 train_ratio > 0、val_ratio >= 0 且 train_ratio + val_ratio < 1。")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kb_blocks = _parse_kb_blocks(DATA_PATH)
    image_meta = _load_image_meta(IMAGE_META_PATH)
    image_groups = _group_images_by_artifact(image_meta)

    samples, summary = _build_samples(
        image_groups=image_groups,
        kb_blocks=kb_blocks,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        limit_artifacts=args.limit_artifacts,
    )

    if not samples:
        raise RuntimeError("没有生成任何多模态样本，请检查知识库路径、图片索引元数据和字段匹配逻辑。")

    _write_jsonl(out_dir / "artifacts.jsonl", samples)
    _write_split_files(out_dir, samples)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"已生成多模态评测数据集：{out_dir}")
    print(f"文物样本数：{summary['artifact_count']}")
    print(f"图片样本数：{summary['image_count']}")
    print(f"划分统计：{summary['split_counts']}")
    print(f"匹配策略统计：{summary['match_strategy_counts']}")
    print(f"未匹配文物数：{summary['unmatched_count']}")


if __name__ == "__main__":
    main()
