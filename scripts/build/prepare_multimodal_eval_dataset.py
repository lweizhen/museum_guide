import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.config import DATA_PATH, IMAGE_META_PATH


OUT_DIR = Path("data/multimodal_eval")
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15

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
            "Build a closed-set multimodal dataset for LoRA fine-tuning and "
            "closed-set image QA evaluation. Images of the same artifact are "
            "split across train/val/test whenever possible."
        )
    )
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output dataset directory.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Per-artifact image train ratio for artifacts with 4+ images.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Per-artifact image validation ratio for artifacts with 4+ images.",
    )
    parser.add_argument(
        "--min-images-per-artifact",
        type=int,
        default=1,
        help="Only keep artifacts with at least this many local images. Single-image artifacts are duplicated into train and test.",
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


def _stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _parse_kb_blocks(path: str) -> list[dict[str, object]]:
    raw = Path(path).read_text(encoding="utf-8").strip()
    if not raw:
        raise RuntimeError(f"知识库文件为空: {path}")

    blocks = [block.strip() for block in re.split(r"\n\s*\n", raw) if block.strip()]
    parsed: list[dict[str, object]] = []
    for index, block in enumerate(blocks, start=1):
        fields: dict[str, str] = {}
        for match in re.finditer(r"【([^】]+)】\s*([^【]*)", block):
            label = match.group(1).strip()
            value = " ".join(match.group(2).strip().split())
            if label and value:
                fields[label] = value
        parsed.append({"block_id": index, "raw_text": block, "fields": fields})
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
    lead = "，".join(part for part in lead_parts if part)
    details = [fields.get(label, "").strip() for label in DESCRIPTION_FIELDS]
    details = [detail for detail in details if detail]
    if lead and details:
        return f"{lead}。{' '.join(details)}"
    return lead or " ".join(details)


def _build_qa_pairs(fields: dict[str, str]) -> list[dict[str, str]]:
    qa_pairs: list[dict[str, str]] = []
    for field_name, question in QA_FIELD_SPECS:
        answer = fields.get(field_name, "").strip()
        if answer:
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
        local_path = str(item.get("local_path", "")).strip()
        if local_path and not Path(local_path).exists():
            continue
        detail_url = (item.get("detail_url") or "").strip()
        name = (item.get("name") or "").strip()
        era = (item.get("era") or "").strip()
        museum = (item.get("museum") or "").strip()
        key = detail_url or _artifact_key(name, era, museum)
        grouped[key].append(item)
    return grouped


def _index_kb_blocks(
    kb_blocks: list[dict[str, object]],
) -> tuple[
    dict[str, list[dict[str, object]]],
    dict[str, list[dict[str, object]]],
    dict[str, list[dict[str, object]]],
]:
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


def _choose_best_block(candidates: list[dict[str, object]], museum: str) -> dict[str, object]:
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

    triple_candidates = by_triple.get(_artifact_key(name, era, museum), [])
    if triple_candidates:
        return _choose_best_block(triple_candidates, museum), "name+era+museum"

    name_era_candidates = by_name_era.get("|".join([_normalize(name), _normalize(era)]), [])
    if name_era_candidates:
        return _choose_best_block(name_era_candidates, museum), "name+era"

    name_candidates = by_name.get(_normalize(name), [])
    if name_candidates:
        return _choose_best_block(name_candidates, museum), "name"
    return None, ""


def _split_images_within_artifact(
    artifact_key: str,
    images: list[dict[str, str]],
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[dict[str, str]]]:
    sorted_images = sorted(
        images,
        key=lambda item: _stable_hash(
            "|".join([artifact_key, str(item.get("local_path", "")), str(item.get("image_url", ""))])
        ),
    )
    total = len(sorted_images)
    if total == 1:
        return {"train": sorted_images, "val": [], "test": sorted_images}
    if total < 1:
        return {"train": [], "val": [], "test": []}
    if total == 2:
        return {"train": sorted_images[:1], "val": [], "test": sorted_images[1:]}
    if total == 3:
        return {"train": sorted_images[:2], "val": [], "test": sorted_images[2:]}

    test_count = max(1, round(total * max(0.0, 1.0 - train_ratio - val_ratio)))
    val_count = max(1, round(total * val_ratio))
    if test_count + val_count >= total:
        test_count = 1
        val_count = 1
    train_count = total - val_count - test_count
    if train_count <= test_count:
        train_count = total - 2
        val_count = 1
        test_count = 1

    val_start = train_count
    test_start = train_count + val_count
    return {
        "train": sorted_images[:train_count],
        "val": sorted_images[val_start:test_start],
        "test": sorted_images[test_start:],
    }


def _identification_output(fields: dict[str, str]) -> str:
    name = fields.get("展品名称", "")
    era = fields.get("所属时代", "")
    category = fields.get("类别", "")
    museum = fields.get("馆藏单位", "")
    parts = [f"这件文物是{name}" if name else "这是一件文物"]
    if era:
        parts.append(f"属于{era}")
    if category:
        parts.append(f"类别为{category}")
    if museum:
        parts.append(f"馆藏单位为{museum}")
    return "，".join(parts) + "。"


def _build_lora_samples_for_image(
    image_row: dict[str, object],
    fields: dict[str, str],
    reference_description: str,
    qa_pairs: list[dict[str, str]],
) -> list[dict[str, object]]:
    base = {
        "artifact_id": image_row["artifact_id"],
        "artifact_name": image_row["artifact_name"],
        "image_id": image_row["image_id"],
        "image_path": image_row["image_path"],
        "image_url": image_row["image_url"],
        "split": image_row["split"],
    }
    samples: list[dict[str, object]] = [
        {
            **base,
            "task": "identify",
            "instruction": "请识别图片中的文物，并说明名称、时代、类别和馆藏单位。",
            "output": _identification_output(fields),
        },
        {
            **base,
            "task": "caption",
            "instruction": "请以博物馆导览员口吻介绍图片中的文物。",
            "output": reference_description,
        },
    ]
    guide_parts = [
        fields.get("历史背景", ""),
        fields.get("历史意义", ""),
        fields.get("文化价值", ""),
        fields.get("故事传说", ""),
    ]
    guide_text = " ".join(part.strip() for part in guide_parts if part.strip())
    if guide_text:
        samples.append(
            {
                **base,
                "task": "guide_style",
                "instruction": "如果你是博物馆讲解员，请讲解这件文物的历史背景和文化价值。",
                "output": guide_text,
            }
        )
    for qa in qa_pairs:
        samples.append(
            {
                **base,
                "task": "qa",
                "instruction": qa["question"],
                "output": qa["answer"],
                "answer_field": qa["answer_field"],
            }
        )
    return samples


def _build_samples(
    image_groups: dict[str, list[dict[str, str]]],
    kb_blocks: list[dict[str, object]],
    train_ratio: float,
    val_ratio: float,
    min_images_per_artifact: int,
    limit_artifacts: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_triple, by_name_era, by_name = _index_kb_blocks(kb_blocks)
    samples: list[dict[str, object]] = []
    unmatched: list[dict[str, str]] = []
    too_few_images: list[dict[str, str]] = []
    image_split_counter: Counter[str] = Counter()
    artifact_split_presence: Counter[str] = Counter()
    match_counter: Counter[str] = Counter()

    items = sorted(image_groups.items(), key=lambda pair: pair[0])
    if limit_artifacts > 0:
        items = items[:limit_artifacts]

    artifact_idx = 0
    for artifact_group_key, images in items:
        if len(images) < min_images_per_artifact:
            first = images[0] if images else {}
            too_few_images.append(
                {
                    "group_key": artifact_group_key,
                    "name": first.get("name", ""),
                    "era": first.get("era", ""),
                    "museum": first.get("museum", ""),
                    "image_count": str(len(images)),
                }
            )
            continue

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

        artifact_idx += 1
        fields = dict(block["fields"])  # type: ignore[arg-type]
        reference_description = _build_reference_description(fields)
        qa_pairs = _build_qa_pairs(fields)
        facts = _build_fact_dict(fields)
        artifact_key = artifact_group_key or _artifact_key(
            first_image.get("name", ""),
            first_image.get("era", ""),
            first_image.get("museum", ""),
        )
        split_images = _split_images_within_artifact(artifact_key, images, train_ratio, val_ratio)
        match_counter[match_strategy] += 1

        all_image_samples: list[dict[str, object]] = []
        split_image_counts: dict[str, int] = {}
        image_idx = 0
        for split_name in ("train", "val", "test"):
            split_items = split_images.get(split_name, [])
            split_image_counts[split_name] = len(split_items)
            if split_items:
                artifact_split_presence[split_name] += 1
            for image_item in split_items:
                image_idx += 1
                local_path = str(image_item.get("local_path", "")).replace("\\", "/")
                all_image_samples.append(
                    {
                        "image_id": f"{artifact_idx:04d}_{image_idx:02d}",
                        "split": split_name,
                        "image_path": local_path,
                        "image_url": image_item.get("image_url", ""),
                    }
                )
                image_split_counter[split_name] += 1

        samples.append(
            {
                "artifact_id": f"artifact_{artifact_idx:04d}",
                "artifact_key": artifact_key,
                "dataset_type": "closed_set_lora",
                "match_strategy": match_strategy,
                "artifact_name": fields.get("展品名称", first_image.get("name", "")),
                "era": fields.get("所属时代", first_image.get("era", "")),
                "museum": fields.get("馆藏单位", first_image.get("museum", "")),
                "category": fields.get("类别", first_image.get("category", "")),
                "detail_url": first_image.get("detail_url", ""),
                "source": first_image.get("source", ""),
                "source_urls": [
                    url
                    for url in [
                        first_image.get("detail_url", ""),
                        fields.get("数据来源", ""),
                        fields.get("补充来源", ""),
                    ]
                    if url
                ],
                "reference_description": reference_description,
                "reference_facts": facts,
                "qa_pairs": qa_pairs,
                "split_image_counts": split_image_counts,
                "images": all_image_samples,
                "kb_block_text": str(block["raw_text"]),
            }
        )

    image_assignment_count = sum(len(sample["images"]) for sample in samples)
    unique_image_keys = {
        str(image.get("image_path") or image.get("image_url") or image.get("image_id"))
        for sample in samples
        for image in sample["images"]
    }

    summary = {
        "dataset_type": "closed_set_lora",
        "split_policy": "Images are split within each artifact. Single-image artifacts are duplicated into train and test; extra images are assigned to train first.",
        "artifact_count": len(samples),
        "image_count": len(unique_image_keys),
        "image_assignment_count": image_assignment_count,
        "image_split_counts": {split: image_split_counter.get(split, 0) for split in ("train", "val", "test")},
        "artifact_split_presence": {split: artifact_split_presence.get(split, 0) for split in ("train", "val", "test")},
        "match_strategy_counts": dict(match_counter),
        "min_images_per_artifact": min_images_per_artifact,
        "excluded_too_few_images_count": len(too_few_images),
        "excluded_too_few_images_examples": too_few_images[:20],
        "unmatched_count": len(unmatched),
        "unmatched_examples": unmatched[:20],
    }
    return samples, summary


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_split_files(out_dir: Path, samples: list[dict[str, object]]) -> dict[str, int]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    image_level_grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    lora_grouped: dict[str, list[dict[str, object]]] = defaultdict(list)

    for sample in samples:
        images_by_split: dict[str, list[dict[str, object]]] = defaultdict(list)
        for image in sample["images"]:  # type: ignore[index]
            split = str(image["split"])
            images_by_split[split].append(image)
            image_row = {
                "artifact_id": sample["artifact_id"],
                "artifact_name": sample["artifact_name"],
                "era": sample["era"],
                "museum": sample["museum"],
                "category": sample["category"],
                "detail_url": sample["detail_url"],
                "source": sample["source"],
                "source_urls": sample["source_urls"],
                "split": split,
                "dataset_type": "closed_set_lora",
                "match_strategy": sample["match_strategy"],
                "image_id": image["image_id"],
                "image_path": image["image_path"],
                "image_url": image["image_url"],
                "reference_description": sample["reference_description"],
                "reference_facts": sample["reference_facts"],
                "qa_pairs": sample["qa_pairs"],
            }
            image_level_grouped[split].append(image_row)
            lora_grouped[split].extend(
                _build_lora_samples_for_image(
                    image_row,
                    sample["reference_facts"],  # type: ignore[arg-type]
                    str(sample["reference_description"]),
                    sample["qa_pairs"],  # type: ignore[arg-type]
                )
            )

        for split_name, split_images in images_by_split.items():
            split_sample = dict(sample)
            split_sample["split"] = split_name
            split_sample["images"] = split_images
            grouped[split_name].append(split_sample)

    lora_counts: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        _write_jsonl(out_dir / f"{split_name}.jsonl", grouped.get(split_name, []))
        _write_jsonl(out_dir / f"{split_name}_images.jsonl", image_level_grouped.get(split_name, []))
        lora_rows = lora_grouped.get(split_name, [])
        _write_jsonl(out_dir / f"{split_name}_lora.jsonl", lora_rows)
        lora_counts[split_name] = len(lora_rows)
    return lora_counts


def main() -> None:
    args = parse_args()
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise RuntimeError(
            "划分比例非法，请确保 train_ratio > 0、val_ratio >= 0 且 train_ratio + val_ratio < 1。"
        )
    if args.min_images_per_artifact < 1:
        raise RuntimeError("闭集跨图测试至少要求每件文物有 2 张图片。")

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
        min_images_per_artifact=args.min_images_per_artifact,
        limit_artifacts=args.limit_artifacts,
    )
    if not samples:
        raise RuntimeError("没有生成任何闭集多模态样本，请检查知识库、图片元数据和最小图片数限制。")

    _write_jsonl(out_dir / "artifacts.jsonl", samples)
    lora_counts = _write_split_files(out_dir, samples)
    summary["lora_sample_counts"] = lora_counts
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"已生成闭集 LoRA / 多模态评测数据集：{out_dir}")
    print(f"文物数：{summary['artifact_count']}")
    print(f"唯一图片数：{summary['image_count']}")
    print(f"图片记录数：{summary['image_assignment_count']}")
    print(f"图片划分：{summary['image_split_counts']}")
    print(f"LoRA 指令样本：{summary['lora_sample_counts']}")
    print(f"匹配策略：{summary['match_strategy_counts']}")
    print(f"因图片数不足排除：{summary['excluded_too_few_images_count']}")
    print(f"未匹配知识库：{summary['unmatched_count']}")


if __name__ == "__main__":
    main()
