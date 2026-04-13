from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.eval_utils import normalize_text, safe_div, strip_model_artifacts
from src.progress import Progress


METRICS_DIR = Path("outputs/metrics")
DEFAULT_INPUT = "outputs/judged/eval_scheme_b_judged_results.csv"
PREDICTION_COLUMNS = (
    "llm_answer",
    "generated_description",
    "answer",
    "fact_answer",
)
REFERENCE_COLUMNS = (
    "gold_answer",
    "reference_description",
    "target",
)
DEFAULT_GROUP_COLUMNS = ("mode", "answer_field")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute text-generation metrics for existing evaluation CSV files. "
            "Supports Scheme B QA, Scheme A QA/caption, and basic RAG outputs."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input evaluation CSV.")
    parser.add_argument(
        "--output",
        default="",
        help="Output CSV. Defaults to '<input_stem>_metrics.csv' in the same folder.",
    )
    parser.add_argument(
        "--summary",
        default="",
        help="Output summary txt. Defaults to '<input_stem>_metrics_summary.txt'.",
    )
    parser.add_argument(
        "--breakdown",
        default="",
        help="Output breakdown json. Defaults to '<input_stem>_metrics_breakdown.json'.",
    )
    parser.add_argument(
        "--prediction-col",
        default="",
        help="Prediction column. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--reference-col",
        default="",
        help="Reference/ground-truth column. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--group-cols",
        default=",".join(DEFAULT_GROUP_COLUMNS),
        help="Comma-separated columns used for grouped summary.",
    )
    parser.add_argument(
        "--mode",
        choices=["direct", "grounded", "both"],
        default="both",
        help="Filter rows by mode when a mode column exists.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit rows for smoke tests.")
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip Sentence-BERT semantic similarity.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable terminal progress display.",
    )
    return parser.parse_args()


def _default_path(input_path: Path, suffix: str) -> Path:
    return METRICS_DIR / f"{input_path.stem}{suffix}"


def _detect_column(fieldnames: list[str], candidates: tuple[str, ...], label: str) -> str:
    for name in candidates:
        if name in fieldnames:
            return name
    raise RuntimeError(
        f"Cannot detect {label} column. Available columns: {', '.join(fieldnames)}"
    )


def _text_units(text: str) -> list[str]:
    text = normalize_text(strip_model_artifacts(text))
    # Chinese is evaluated character-wise; alphanumeric spans are kept as tokens.
    import re

    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text)


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_l(prediction: str, reference: str) -> tuple[float, float, float]:
    pred_tokens = _text_units(prediction)
    ref_tokens = _text_units(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0

    prev = [0] * (len(ref_tokens) + 1)
    for pred_token in pred_tokens:
        curr = [0]
        for j, ref_token in enumerate(ref_tokens, start=1):
            if pred_token == ref_token:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr

    lcs = prev[-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def bleu(prediction: str, reference: str, max_order: int) -> float:
    pred_tokens = _text_units(prediction)
    ref_tokens = _text_units(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    precisions: list[float] = []
    for order in range(1, max_order + 1):
        pred_counts = _ngrams(pred_tokens, order)
        ref_counts = _ngrams(ref_tokens, order)
        possible = sum(pred_counts.values())
        if possible == 0:
            precisions.append(0.0)
            continue

        overlap = sum(
            min(count, ref_counts.get(ngram, 0)) for ngram, count in pred_counts.items()
        )
        # Add-one smoothing keeps sentence-level BLEU from collapsing too harshly.
        precisions.append((overlap + 1.0) / (possible + 1.0))

    if any(score <= 0 for score in precisions):
        return 0.0

    log_precision = sum(math.log(score) for score in precisions) / max_order
    brevity_penalty = (
        1.0
        if len(pred_tokens) > len(ref_tokens)
        else math.exp(1.0 - len(ref_tokens) / max(len(pred_tokens), 1))
    )
    return brevity_penalty * math.exp(log_precision)


def _safe_float(value: str) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _round(value: float) -> float:
    return round(float(value), 4)


def add_embedding_similarity(
    rows: list[dict[str, str]],
    prediction_col: str,
    reference_col: str,
) -> tuple[bool, str]:
    pairs = [
        (
            strip_model_artifacts(row.get(prediction_col, "")),
            row.get(reference_col, ""),
        )
        for row in rows
    ]
    texts: list[str] = []
    for prediction, reference in pairs:
        texts.append(prediction)
        texts.append(reference)

    try:
        from src.embedder import encode_texts

        embeddings = encode_texts(texts)
    except Exception as exc:  # noqa: BLE001 - metrics should still be usable.
        for row in rows:
            row["metric_semantic_similarity"] = ""
        return False, str(exc)

    for idx, row in enumerate(rows):
        pred_vec = embeddings[idx * 2]
        ref_vec = embeddings[idx * 2 + 1]
        if not pairs[idx][0].strip() or not pairs[idx][1].strip():
            similarity = 0.0
        else:
            similarity = float(np.dot(pred_vec, ref_vec))
        row["metric_semantic_similarity"] = f"{similarity:.4f}"

    return True, ""


def _filter_rows(rows: list[dict[str, str]], mode: str) -> list[dict[str, str]]:
    if mode == "both" or not rows or "mode" not in rows[0]:
        return rows
    return [row for row in rows if row.get("mode") == mode]


def compute_metrics(
    rows: list[dict[str, str]],
    prediction_col: str,
    reference_col: str,
    *,
    progress_enabled: bool,
) -> None:
    progress = Progress(len(rows), "Text metrics", enabled=progress_enabled)
    try:
        for row in rows:
            prediction = row.get(prediction_col, "")
            reference = row.get(reference_col, "")
            rouge_p, rouge_r, rouge_f1 = rouge_l(prediction, reference)
            row["metric_rouge_l_precision"] = f"{rouge_p:.4f}"
            row["metric_rouge_l_recall"] = f"{rouge_r:.4f}"
            row["metric_rouge_l_f1"] = f"{rouge_f1:.4f}"
            row["metric_bleu_1"] = f"{bleu(prediction, reference, 1):.4f}"
            row["metric_bleu_2"] = f"{bleu(prediction, reference, 2):.4f}"
            row["metric_bleu_4"] = f"{bleu(prediction, reference, 4):.4f}"
            progress.advance()
    finally:
        progress.close()


def _group_key(row: dict[str, str], group_cols: list[str]) -> str:
    if not group_cols:
        return "overall"
    parts = [f"{col}={row.get(col, '') or '<empty>'}" for col in group_cols]
    return " | ".join(parts)


def summarize(
    rows: list[dict[str, str]],
    group_cols: list[str],
    *,
    input_path: Path,
    output_path: Path,
    prediction_col: str,
    reference_col: str,
    embedding_enabled: bool,
    embedding_error: str,
) -> dict[str, Any]:
    metric_names = [
        "metric_rouge_l_f1",
        "metric_bleu_1",
        "metric_bleu_2",
        "metric_bleu_4",
        "metric_semantic_similarity",
    ]
    groups: dict[str, list[dict[str, str]]] = {"overall": rows}
    for row in rows:
        key = _group_key(row, group_cols)
        groups.setdefault(key, []).append(row)

    group_stats: dict[str, dict[str, Any]] = {}
    for key, group_rows in groups.items():
        evaluated = [
            row
            for row in group_rows
            if row.get(prediction_col, "").strip() and row.get(reference_col, "").strip()
        ]
        stats: dict[str, Any] = {
            "rows": len(group_rows),
            "evaluated_rows": len(evaluated),
        }

        for metric_name in metric_names:
            values = [
                value
                for row in group_rows
                if (value := _safe_float(row.get(metric_name, ""))) is not None
            ]
            stats[f"avg_{metric_name.removeprefix('metric_')}"] = _round(_avg(values))

        if "auto_correct" in group_rows[0]:
            values = [
                value
                for row in group_rows
                if (value := _safe_float(row.get("auto_correct", ""))) is not None
            ]
            stats["auto_accuracy"] = _round(_avg(values))

        if "auto_score" in group_rows[0]:
            values = [
                value
                for row in group_rows
                if (value := _safe_float(row.get("auto_score", ""))) is not None
            ]
            stats["avg_auto_score"] = _round(_avg(values))

        if "judge_pass" in group_rows[0]:
            judge_values = [
                1.0 if row.get("judge_pass") == "true" else 0.0
                for row in group_rows
                if row.get("judge_pass") in {"true", "false"}
            ]
            stats["judge_pass_rate"] = _round(_avg(judge_values))

        if "judge_score" in group_rows[0]:
            values = [
                value
                for row in group_rows
                if (value := _safe_float(row.get("judge_score", ""))) is not None
            ]
            stats["avg_judge_score"] = _round(_avg(values))

        group_stats[key] = stats

    return {
        "input": str(input_path),
        "output": str(output_path),
        "prediction_col": prediction_col,
        "reference_col": reference_col,
        "group_cols": group_cols,
        "rows": len(rows),
        "embedding_enabled": embedding_enabled,
        "embedding_error": embedding_error,
        "groups": group_stats,
    }


def write_summary(summary_path: Path, breakdown: dict[str, Any]) -> None:
    lines = [
        f"Input: {breakdown['input']}",
        f"Output: {breakdown['output']}",
        f"Prediction column: {breakdown['prediction_col']}",
        f"Reference column: {breakdown['reference_col']}",
        f"Rows: {breakdown['rows']}",
        f"Embedding enabled: {breakdown['embedding_enabled']}",
    ]
    if breakdown["embedding_error"]:
        lines.append(f"Embedding warning: {breakdown['embedding_error']}")

    for key, stats in breakdown["groups"].items():
        lines.append(f"[{key}]")
        lines.append(f"rows: {stats['rows']}")
        lines.append(f"evaluated rows: {stats['evaluated_rows']}")
        for metric_key, metric_value in stats.items():
            if metric_key in {"rows", "evaluated_rows"}:
                continue
            lines.append(f"{metric_key}: {metric_value:.4f}")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    metric_fields = [
        "metric_rouge_l_precision",
        "metric_rouge_l_recall",
        "metric_rouge_l_f1",
        "metric_bleu_1",
        "metric_bleu_2",
        "metric_bleu_4",
        "metric_semantic_similarity",
    ]
    output_fields = list(fieldnames)
    for field in metric_fields:
        if field not in output_fields:
            output_fields.append(field)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_path = Path(args.output) if args.output else _default_path(input_path, "_metrics.csv")
    summary_path = (
        Path(args.summary) if args.summary else _default_path(input_path, "_metrics_summary.txt")
    )
    breakdown_path = (
        Path(args.breakdown)
        if args.breakdown
        else _default_path(input_path, "_metrics_breakdown.json")
    )

    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        raise RuntimeError(f"Input CSV has no header: {input_path}")

    rows = _filter_rows(rows, args.mode)
    if args.limit > 0:
        rows = rows[: args.limit]

    prediction_col = args.prediction_col or _detect_column(
        fieldnames, PREDICTION_COLUMNS, "prediction"
    )
    reference_col = args.reference_col or _detect_column(
        fieldnames, REFERENCE_COLUMNS, "reference"
    )
    group_cols = [
        col.strip()
        for col in args.group_cols.split(",")
        if col.strip() and col.strip() in fieldnames
    ]

    compute_metrics(
        rows,
        prediction_col,
        reference_col,
        progress_enabled=not args.no_progress,
    )

    embedding_enabled = False
    embedding_error = ""
    if not args.skip_embedding:
        embedding_enabled, embedding_error = add_embedding_similarity(
            rows,
            prediction_col,
            reference_col,
        )
    else:
        for row in rows:
            row["metric_semantic_similarity"] = ""

    write_csv(output_path, rows, fieldnames)
    breakdown = summarize(
        rows,
        group_cols,
        input_path=input_path,
        output_path=output_path,
        prediction_col=prediction_col,
        reference_col=reference_col,
        embedding_enabled=embedding_enabled,
        embedding_error=embedding_error,
    )
    breakdown_path.parent.mkdir(parents=True, exist_ok=True)
    breakdown_path.write_text(
        json.dumps(breakdown, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary(summary_path, breakdown)

    print(f"OK: wrote metrics CSV: {output_path}")
    print(f"OK: wrote summary: {summary_path}")
    print(f"OK: wrote breakdown: {breakdown_path}")
    if embedding_error:
        print(f"WARNING: semantic similarity skipped: {embedding_error}")


if __name__ == "__main__":
    main()
