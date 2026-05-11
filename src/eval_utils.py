"""评测工具函数模块。

这里集中放置统一评测脚本会复用的小工具，例如 JSONL 读取、
文本归一化、字段抽取、自动打分、JSON 解析等。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

LB = "\u3010"
RB = "\u3011"
QUESTION_PREFIX = "\u3002\u95ee\u9898\uff1a"


def read_jsonl(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    """读取 JSONL 文件，`limit` 大于 0 时只读取前若干条。"""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def normalize_text(text: str) -> str:
    """去掉空白并转小写，便于自动指标做宽松匹配。"""
    return re.sub(r"\s+", "", (text or "").strip()).lower()


def strip_model_artifacts(text: str) -> str:
    """移除模型可能输出的 `<think>...</think>` 推理片段。"""
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    return cleaned.strip()


def extract_field(doc: str, label: str) -> str:
    """从半结构化知识块中提取指定字段内容。"""
    pattern = rf"{LB}{re.escape(label)}{RB}\s*(.*?)(?={LB}|$)"
    match = re.search(pattern, doc or "", flags=re.S)
    return match.group(1).strip() if match else ""


def build_text_query(name: str, question: str, era: str = "", museum: str = "") -> str:
    """把文物名称、时代、馆藏单位和用户问题拼成文本检索查询。"""
    prefix_parts = [
        part.strip()
        for part in [name, era, museum]
        if part and part.strip() and part.strip() != "-"
    ]
    prefix = " ".join(prefix_parts)
    if prefix:
        return f"{prefix}{QUESTION_PREFIX}{question.strip()}"
    return question.strip()


def safe_div(num: int | float, den: int | float) -> float:
    """安全除法，分母为 0 时返回 0。"""
    return float(num) / float(den) if den else 0.0


def summarize_counter(counter: Counter[str], limit: int = 10) -> str:
    """把 Counter 的高频项格式化为摘要字符串。"""
    if not counter:
        return "none"
    return "; ".join(f"{key}: {count}" for key, count in counter.most_common(limit))


def _answer_units(text: str) -> list[str]:
    """把答案拆成中文单字和英文/数字片段，用于宽松重合度计算。"""
    text = normalize_text(text)
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text)


def _overlap_f1(predicted: str, expected: str) -> tuple[float, float, float]:
    """计算预测答案与参考答案的字符/词片段级 precision、recall、F1。"""
    pred_units = _answer_units(predicted)
    gold_units = _answer_units(expected)
    if not pred_units or not gold_units:
        return 0.0, 0.0, 0.0

    pred_counter = Counter(pred_units)
    gold_counter = Counter(gold_units)
    overlap = sum(min(pred_counter[token], gold_counter[token]) for token in gold_counter)
    precision = overlap / max(len(pred_units), 1)
    recall = overlap / max(len(gold_units), 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score_answer(predicted: str, expected: str) -> tuple[bool, float]:
    """对单条答案做任务型自动评分。

    返回 `(是否命中, 分数)`。该指标不是通用语义评价，而是为了快速判断
    模型回答是否覆盖参考答案中的核心事实。
    """
    pred = strip_model_artifacts(predicted)
    gold = (expected or "").strip()
    if not pred or not gold:
        return False, 0.0

    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)
    if pred_norm == gold_norm:
        return True, 1.0

    if gold_norm and gold_norm in pred_norm:
        return True, 0.95 if len(gold_norm) <= 24 else 0.85

    if pred_norm and pred_norm in gold_norm and len(pred_norm) >= 4:
        return True, 0.8

    _precision, recall, f1 = _overlap_f1(pred, gold)
    score = max(f1, recall * 0.85)
    return (recall >= 0.6 and f1 >= 0.35), score


def contains_expected(predicted: str, expected: str) -> int:
    """判断预测文本中是否直接包含参考答案。"""
    pred_norm = normalize_text(strip_model_artifacts(predicted))
    expected_norm = normalize_text(expected)
    return int(bool(expected_norm and expected_norm in pred_norm))


def parse_json_object(text: str) -> dict[str, Any] | None:
    """从模型输出中解析 JSON 对象。

    如果输出前后夹杂说明文字，会尝试提取第一个 `{...}` 片段。
    """
    text = (text or "").strip()
    if not text:
        return None

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None
