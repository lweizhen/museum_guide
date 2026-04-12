from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
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
    return re.sub(r"\s+", "", (text or "").strip()).lower()


def strip_model_artifacts(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    return cleaned.strip()


def extract_field(doc: str, label: str) -> str:
    pattern = rf"【{re.escape(label)}】\s*(.*?)(?=【|$)"
    match = re.search(pattern, doc or "", flags=re.S)
    return match.group(1).strip() if match else ""


def build_text_query(name: str, question: str, era: str = "", museum: str = "") -> str:
    prefix_parts = [
        part.strip()
        for part in [name, era, museum]
        if part and part.strip() and part.strip() != "-"
    ]
    prefix = " ".join(prefix_parts)
    if prefix:
        return f"{prefix}。问题：{question.strip()}"
    return question.strip()


def safe_div(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def summarize_counter(counter: Counter[str], limit: int = 10) -> str:
    if not counter:
        return "none"
    return "; ".join(f"{key}: {count}" for key, count in counter.most_common(limit))


def _answer_units(text: str) -> list[str]:
    text = normalize_text(text)
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text)


def _overlap_f1(predicted: str, expected: str) -> tuple[float, float, float]:
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
    pred_norm = normalize_text(strip_model_artifacts(predicted))
    expected_norm = normalize_text(expected)
    return int(bool(expected_norm and expected_norm in pred_norm))


def parse_json_object(text: str) -> dict[str, Any] | None:
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
