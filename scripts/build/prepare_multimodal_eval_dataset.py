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
from src.llm import call_llm


OUT_DIR = Path("data/multimodal_eval")
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15

ANSWER_GENERATOR = "template"
LLM_CACHE: dict[str, str] = {}
LLM_CACHE_PATH: Path | None = None
LLM_MAX_CALLS = 0
LLM_CALL_COUNT = 0

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

SHORT_ANSWER_FIELDS = {
    "展品名称",
    "所属时代",
    "类别",
    "馆藏单位",
    "材质",
    "出土地",
}

GUIDE_ANSWER_PREFIX = {
    "功能用途": "它的用途可以这样理解：",
    "历史意义": "它的重要之处在于，",
    "文化价值": "它的文化价值主要体现在，",
    "纹饰与造型": "参观时可以重点留意它的造型和纹饰：",
    "历史背景": "把它放回当时的历史环境中看，",
    "故事传说": "关于它的故事，可以这样讲给观众听：",
}

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
    parser.add_argument(
        "--answer-generator",
        choices=["template", "llm"],
        default="template",
        help="How to generate LoRA target answers. template=rule-based templates; llm=use the configured text LLM to generate grounded museum-guide answers.",
    )
    parser.add_argument(
        "--llm-cache",
        default="data/multimodal_eval/llm_generation_cache.jsonl",
        help="Append-only cache file for LLM-generated LoRA answers. Used when --answer-generator llm.",
    )
    parser.add_argument(
        "--llm-max-calls",
        type=int,
        default=0,
        help="Maximum new LLM generations to make in this run. 0 means no cap. Useful for staged generation and resume.",
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


def _format_grounding_context(fields: dict[str, str]) -> str:
    lines: list[str] = []
    for label in FACT_FIELD_ORDER:
        value = fields.get(label, "").strip()
        if value:
            lines.append(f"【{label}】{value}")
    return "\n".join(lines)


def _with_grounding_context(question: str, context: str) -> str:
    return f"""你是一名博物馆导览员。请结合图片和下方参考资料回答观众问题。

要求：
1. 事实以参考资料为准，不要编造资料中没有的信息。
2. 如果问题询问名称、时代、类别、馆藏单位、材质等字段，请直接准确回答。
3. 如果问题询问历史背景、文化价值、功能用途或故事，请像导览员一样自然讲解，先点明文物，再说明看点或价值。
4. 回答控制在180字以内，不要输出参考资料标题或推理过程。

【参考资料】
{context}

【观众问题】
{question}"""

def _compact_join(parts: list[str]) -> str:
    cleaned = [part.strip().rstrip("。") for part in parts if part and part.strip()]
    if not cleaned:
        return ""
    return "。".join(cleaned) + "。"


def _artifact_name(fields: dict[str, str]) -> str:
    return fields.get("展品名称", "").strip() or "这件文物"


def _artifact_intro(fields: dict[str, str]) -> str:
    name = _artifact_name(fields)
    era = fields.get("所属时代", "").strip()
    category = fields.get("类别", "").strip()
    if era and category:
        return f"{name}是一件{era}时期的{category}"
    if era:
        return f"{name}属于{era}时期"
    if category:
        return f"{name}是一件{category}"
    return name


def _artifact_identity_clause(fields: dict[str, str]) -> str:
    era = fields.get("所属时代", "").strip()
    if not era:
        return ""
    dynastic_eras = {
        "夏", "商", "西周", "东周", "春秋", "战国", "秦", "西汉", "东汉", "汉", "三国",
        "西晋", "东晋", "晋", "南北朝", "隋", "唐", "五代", "北宋", "南宋", "宋", "辽",
        "金", "元", "明", "清", "民国"
    }
    if era in dynastic_eras:
        return f"它出自{era}代"
    if era.endswith(("代", "朝")):
        return f"它出自{era}"
    return f"它出自{era}时期"


def _category_clause(fields: dict[str, str]) -> str:
    return ""


def _museum_clause(fields: dict[str, str]) -> str:
    museum = fields.get("馆藏单位", "").strip()
    if museum:
        return f"现在收藏在{museum}"
    return ""


def _value_to_spoken_clause(value: str) -> str:
    value = value.strip()
    replacements = (
        ("反映了", "它反映出"),
        ("体现了", "它体现出"),
        ("展现了", "它展现出"),
        ("折射出", "它折射出"),
        ("说明了", "它说明了"),
        ("见证了", "它见证了"),
    )
    for prefix, replacement in replacements:
        if value.startswith(prefix):
            return replacement + value[len(prefix):]
    return value


def _value_to_seen_clause(value: str) -> str:
    value = value.strip()
    replacements = (
        "反映了",
        "体现了",
        "展现了",
        "折射出",
        "说明了",
        "见证了",
    )
    for prefix in replacements:
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def _guide_caption_output(fields: dict[str, str]) -> str:
    name = _artifact_name(fields)
    identity_clause = _artifact_identity_clause(fields)
    category_clause = _category_clause(fields)
    museum_clause = _museum_clause(fields)
    function = fields.get("功能用途", "").strip()
    value = fields.get("文化价值", "").strip() or fields.get("历史意义", "").strip()
    background = fields.get("历史背景", "").strip()
    parts: list[str] = [f"大家现在看到的是{name}"]
    if identity_clause:
        parts.append(identity_clause)
    if category_clause:
        parts.append(category_clause)
    if museum_clause:
        parts.append(museum_clause)
    if function:
        parts.append(f"在当时，这类器物并不只是摆设，{function}")
    if value:
        if value.startswith(("反映", "体现", "展现", "折射", "说明", "见证", "对研究", "为研究")):
            parts.append(f"透过它，我们也能看到{_value_to_seen_clause(value)}")
        else:
            parts.append(f"透过它，我们也能感受到{value}")
    if background:
        parts.append(f"把它放回当时的生活里看，{background}")
    return _compact_join(parts)


def _guide_style_output(fields: dict[str, str]) -> str:
    name = _artifact_name(fields)
    identity_clause = _artifact_identity_clause(fields)
    background = fields.get("历史背景", "").strip()
    value = fields.get("文化价值", "").strip() or fields.get("历史意义", "").strip()
    story = fields.get("故事传说", "").strip()
    parts = ["这件文物真正吸引人的地方，不只在器物本身，也在它背后连着的时代生活和文化信息"]
    if identity_clause:
        parts.append(identity_clause)
    if background:
        parts.append(f"放在当时的时代里看，{background}")
    if value:
        if value.startswith(("反映", "体现", "展现", "折射", "说明", "见证", "对研究", "为研究")):
            parts.append(f"所以说，今天我们还能从中看到{_value_to_seen_clause(value)}")
        else:
            parts.append(f"也正因为这样，它在今天依然有价值，因为{value}")
    if story:
        parts.append(f"要是再联系它背后的故事来看，{story}")
    return _compact_join(parts)


def _guide_answer_for_qa(answer_field: str, answer: str, fields: dict[str, str]) -> str:
    answer = answer.strip()
    if not answer:
        return answer
    if answer_field in SHORT_ANSWER_FIELDS:
        return answer
    name = fields.get("展品名称", "").strip() or "这件文物"
    if answer_field == "功能用途":
        return f"在当时，人们会这样使用{name}：{answer}"
    if answer_field == "文化价值":
        if answer.startswith(("反映", "体现", "展现", "折射", "说明", "见证", "对研究", "为研究")):
            return f"它让我们看到{_value_to_seen_clause(answer)}"
        return f"这件文物真正值得关注的地方在于，{answer}"
    if answer_field == "历史背景":
        return f"要理解{name}，得先把它放回当时的时代环境里。{answer}"
    if answer_field == "历史意义":
        return f"{name}之所以重要，正在于{answer}"
    if answer_field == "纹饰与造型":
        return f"看{name}的时候，可以先注意这些细节：{answer}"
    if answer_field == "故事传说":
        return f"如果顺着它背后的故事往下看，{answer}"
    prefix = GUIDE_ANSWER_PREFIX.get(answer_field, "")
    if prefix and not answer.startswith(prefix):
        return f"{prefix}{answer}"
    return answer


def _guide_overview_output(fields: dict[str, str]) -> str:
    name = _artifact_name(fields)
    identity_clause = _artifact_identity_clause(fields)
    function = fields.get("功能用途", "").strip()
    value = fields.get("文化价值", "").strip() or fields.get("历史意义", "").strip()
    background = fields.get("历史背景", "").strip()
    parts = [f"大家现在看到的是{name}"]
    if identity_clause:
        parts.append(identity_clause)
    if function:
        parts.append(f"它和当时人们的日常生活联系很紧，{function}")
    if value:
        if value.startswith(("反映", "体现", "展现", "折射", "说明", "见证", "对研究", "为研究")):
            parts.append(f"今天再看它，我们不只是看一件器物，也是在读它背后的文化信息，比如{_value_to_seen_clause(value)}")
        else:
            parts.append(f"今天再看它，我们不只是看一件器物，也是在读它背后的文化信息，因为{value}")
    if background and len(parts) < 4:
        parts.append(f"放在当时的时代里看，{background}")
    return _compact_join(parts)


def _guide_highlight_output(fields: dict[str, str]) -> str:
    name = _artifact_name(fields)
    shape = fields.get("纹饰与造型", "").strip()
    function = fields.get("功能用途", "").strip()
    value = fields.get("文化价值", "").strip() or fields.get("历史意义", "").strip()
    parts = [f"看{name}的时候，不妨先注意它最有代表性的几个细节"]
    if shape:
        parts.append(f"先看这里，{shape}")
    if function:
        parts.append(f"这些地方不只是好看，也和实际用途连在一起，{function}")
    if value:
        if value.startswith(("反映", "体现", "展现", "折射", "说明", "见证", "对研究", "为研究")):
            parts.append(f"也正因为这样，我们还能从中看到{_value_to_seen_clause(value)}")
        else:
            parts.append(f"也正因为这样，它也能帮助我们理解相关历史文化，因为{value}")
    return _compact_join(parts)

def _guide_story_output(fields: dict[str, str]) -> str:
    name = _artifact_name(fields)
    identity_clause = _artifact_identity_clause(fields)
    story = fields.get("故事传说", "").strip()
    background = fields.get("历史背景", "").strip()
    value = fields.get("历史意义", "").strip() or fields.get("文化价值", "").strip()
    parts = [f"如果把时间拨回它所属的年代，{name}就不只是一件陈列在展柜里的文物"]
    if identity_clause:
        parts.append(identity_clause)
    if story:
        parts.append(story)
    elif background:
        parts.append(f"放在当时来看，{background}")
    if value:
        if value.startswith(("反映", "体现", "展现", "折射", "说明", "见证", "对研究", "为研究")):
            parts.append(f"所以说，今天我们还能从中看到{_value_to_seen_clause(value)}")
        else:
            parts.append(f"所以说，它留给今天的重要意义就在于{value}")
    return _compact_join(parts)

def _load_llm_cache(path: Path) -> dict[str, str]:
    cache: dict[str, str] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(row.get("key", "")).strip()
            value = str(row.get("value", "")).strip()
            if key and value:
                cache[key] = value
    return cache


def _append_llm_cache(path: Path, key: str, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


def _should_use_llm_generation(task: str, answer_field: str | None = None) -> bool:
    if ANSWER_GENERATOR != "llm":
        return False
    if task == "identify":
        return False
    if task == "grounded_qa" and answer_field in SHORT_ANSWER_FIELDS:
        return False
    return True


def _build_llm_target_prompt(
    task: str,
    instruction: str,
    template_output: str,
    fields: dict[str, str],
    answer_field: str | None = None,
) -> str:
    name = _artifact_name(fields)
    task_hint_map = {
        "grounded_caption": "请生成一段自然、准确、适合普通观众收听的文物导览介绍。",
        "grounded_guide_style": "请生成一段更像博物馆讲解员现场讲解的回答，突出历史背景、文化信息和讲解感。",
        "grounded_overview_guide": "请生成一段约30秒的导览开场讲解。",
        "grounded_highlight_guide": "请生成一段提醒观众重点观察看点的讲解。",
        "grounded_story_guide": "请生成一段更有画面感和故事感的讲解。",
        "grounded_qa": "请针对观众问题生成自然、准确的中文回答。",
    }
    task_hint = task_hint_map.get(task, "请生成自然、准确的中文导览回答。")
    if task == "grounded_qa" and answer_field:
        task_hint += f" 当前问题字段为：{answer_field}。"
    return (
        "你是一名经验丰富的中文博物馆导览员，现在要为多模态文物问答模型撰写训练回答。\n"
        "你的目标不是复述资料字段，而是把可靠资料转化成自然、可信、适合观众收听的讲解语言。\n\n"
        "请严格遵守以下要求：\n"
        "1. 只能依据给定资料作答，不得编造名称、年代、用途、馆藏单位、历史背景等事实。\n"
        "2. 回答要像导览员面对观众讲话，口语自然、通顺，有讲解感，但不要过度夸张。\n"
        "3. 不要使用‘从导览的角度’‘从研究价值来看’‘如果从用途来看’这类生硬元话语。\n"
        "4. 不要写成资料摘抄，不要机械罗列字段，不要照抄模板参考答案的句式。\n"
        "5. 可以使用自然导览表达，如‘大家现在看到的是……’‘先看这里……’‘把它放回当时看……’。\n"
        "6. 如果是问答型任务，先直接回答问题，再用一两句自然补充；如果是讲解型任务，写成完整连贯的一小段讲解。\n"
        "7. 不要输出标题、标签、字段名、括号说明、分点或额外解释，只输出最终回答。\n\n"
        f"文物名称：{name}\n"
        f"任务类型：{task}\n"
        f"任务要求：{task_hint}\n"
        f"用户指令：{instruction}\n"
        f"模板参考答案（仅供理解任务，不要照抄句式）：{template_output}\n"
    )

def _generate_output_with_llm(
    task: str,
    instruction: str,
    template_output: str,
    fields: dict[str, str],
    answer_field: str | None = None,
) -> str:
    global LLM_CALL_COUNT
    cache_key = _stable_hash(
        json.dumps(
            {
                "task": task,
                "instruction": instruction,
                "template_output": template_output,
                "artifact": _artifact_name(fields),
                "answer_field": answer_field or "",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    cached = LLM_CACHE.get(cache_key, "").strip()
    if cached:
        return cached
    if LLM_MAX_CALLS > 0 and LLM_CALL_COUNT >= LLM_MAX_CALLS:
        raise RuntimeError(
            "已达到本次运行允许的新 LLM 生成调用上限。可以直接重新运行同一命令继续，已缓存结果会自动复用。"
        )
    prompt = _build_llm_target_prompt(task, instruction, template_output, fields, answer_field)
    output = call_llm(prompt).strip()
    if not output:
        raise RuntimeError(f"大模型未能为任务 {task} 生成有效训练回答。")
    LLM_CACHE[cache_key] = output
    LLM_CALL_COUNT += 1
    if LLM_CACHE_PATH is not None:
        _append_llm_cache(LLM_CACHE_PATH, cache_key, output)
    return output




def _build_lora_samples_for_image(
    image_row: dict[str, object],
    fields: dict[str, str],
    reference_description: str,
    qa_pairs: list[dict[str, str]],
) -> list[dict[str, object]]:
    del reference_description
    base = {
        "artifact_id": image_row["artifact_id"],
        "artifact_name": image_row["artifact_name"],
        "image_id": image_row["image_id"],
        "image_path": image_row["image_path"],
        "image_url": image_row["image_url"],
        "split": image_row["split"],
    }
    grounding_context = _format_grounding_context(fields)
    overview_text = _guide_overview_output(fields)
    highlight_text = _guide_highlight_output(fields)
    story_text = _guide_story_output(fields)
    caption_text = _guide_caption_output(fields)
    guide_style_text = _guide_style_output(fields)

    def build_sample(
        task: str,
        instruction: str,
        template_output: str,
        answer_field: str | None = None,
    ) -> dict[str, object]:
        output = template_output
        if _should_use_llm_generation(task, answer_field):
            output = _generate_output_with_llm(task, instruction, template_output, fields, answer_field)
        sample = {
            **base,
            "task": task,
            "instruction": instruction,
            "output": output,
        }
        if task != "identify":
            sample["grounding_context"] = grounding_context
        if answer_field:
            sample["answer_field"] = answer_field
        return sample

    samples: list[dict[str, object]] = [
        {
            **base,
            "task": "identify",
            "instruction": "请识别图片中的文物，并说明名称、时代、类别和馆藏单位。",
            "output": _identification_output(fields),
        },
        build_sample(
            "grounded_caption",
            _with_grounding_context(
                "请以博物馆导览员口吻介绍图片中的文物，重点说明名称、时代、类别、用途和文化价值。",
                grounding_context,
            ),
            caption_text,
        ),
    ]

    guide_parts = [
        fields.get("历史背景", ""),
        fields.get("历史意义", ""),
        fields.get("文化价值", ""),
        fields.get("故事传说", ""),
    ]
    guide_text = " ".join(part.strip() for part in guide_parts if part.strip())
    if guide_text and guide_style_text:
        samples.append(
            build_sample(
                "grounded_guide_style",
                _with_grounding_context("请讲解这件文物的历史背景、历史意义和文化价值。", grounding_context),
                guide_style_text,
            )
        )

    for task_name, question, output in [
        ("grounded_overview_guide", "请用约30秒的导览口吻，为普通观众介绍这件文物。", overview_text),
        ("grounded_highlight_guide", "请告诉观众参观这件文物时最值得注意的看点。", highlight_text),
        ("grounded_story_guide", "请用更有画面感的方式讲讲这件文物背后的历史或故事。", story_text),
    ]:
        if output:
            samples.append(build_sample(task_name, _with_grounding_context(question, grounding_context), output))

    for qa in qa_pairs:
        answer_field = qa["answer_field"]
        instruction = _with_grounding_context(qa["question"], grounding_context)
        template_output = _guide_answer_for_qa(answer_field, qa["answer"], fields)
        samples.append(build_sample("grounded_qa", instruction, template_output, answer_field=answer_field))
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
        "lora_sample_policy": "LoRA samples are aligned with vl_rag_lora and enhanced for guide style. Except for identify, samples include grounding_context; additional overview/highlight/story guide tasks teach museum-guide tone and richer explanations.",
        "answer_generator": ANSWER_GENERATOR,
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
    global ANSWER_GENERATOR, LLM_CACHE, LLM_CACHE_PATH, LLM_MAX_CALLS, LLM_CALL_COUNT
    args = parse_args()
    ANSWER_GENERATOR = args.answer_generator
    LLM_MAX_CALLS = args.llm_max_calls
    LLM_CALL_COUNT = 0
    if ANSWER_GENERATOR == "llm":
        LLM_CACHE_PATH = Path(args.llm_cache)
        LLM_CACHE = _load_llm_cache(LLM_CACHE_PATH)
        print(f"已加载 LLM 训练回答缓存：{len(LLM_CACHE)} 条")
    else:
        LLM_CACHE_PATH = None
        LLM_CACHE = {}
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
