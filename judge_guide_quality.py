from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.eval_utils import parse_json_object, safe_div, summarize_counter
from src.llm import call_judge_llm
from src.progress import Progress


DEFAULT_INPUT = Path("outputs/raw/eval_scheme_a_qa_results.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/judged")

ANSWER_COLUMNS = (
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
CONTEXT_COLUMNS = (
    "question",
    "query",
    "target",
    "target_name",
    "retrieved_top1",
    "retrieved_doc_name",
    "retrieved_names",
    "recognized_name",
    "recognized_era",
    "recognized_museum",
    "gold_answer",
    "reference_description",
)
GUIDE_SCORE_COLUMNS = (
    "guide_factuality",
    "guide_groundedness",
    "guide_style",
    "guide_clarity",
    "guide_completeness",
    "guide_fluency",
    "guide_engagement",
    "guide_overall",
)
GUIDE_COLUMNS = GUIDE_SCORE_COLUMNS + (
    "guide_pass",
    "guide_reason",
    "guide_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge whether generated museum-guide answers sound like a good guide."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input result CSV. Default: outputs/raw/eval_scheme_a_qa_results.csv",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output CSV. Default: outputs/judged/<input_stem>_guide_quality.csv",
    )
    parser.add_argument(
        "--summary",
        default="",
        help="Summary TXT. Default: outputs/judged/<input_stem>_guide_quality_summary.txt",
    )
    parser.add_argument(
        "--breakdown",
        default="",
        help="Breakdown JSON. Default: outputs/judged/<input_stem>_guide_quality_breakdown.json",
    )
    parser.add_argument(
        "--answer-col",
        default="",
        help="Prediction column to judge. Default: auto-detect.",
    )
    parser.add_argument(
        "--reference-col",
        default="",
        help="Reference / gold-answer column. Default: auto-detect.",
    )
    parser.add_argument(
        "--group-cols",
        default="mode",
        help="Comma-separated columns for grouped summaries. Missing columns are ignored.",
    )
    parser.add_argument(
        "--mode",
        choices=["direct", "grounded", "both"],
        default="both",
        help="Optional filter when the CSV has a mode column. Default: both.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum rows to judge. 0 means all eligible rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-judge rows even when guide quality fields already exist.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when a judge model call fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate input and print how many rows would be judged.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable terminal progress output.",
    )
    return parser.parse_args()


def _default_paths(input_path: Path) -> tuple[Path, Path, Path]:
    stem = input_path.stem
    output = DEFAULT_OUTPUT_DIR / f"{stem}_guide_quality.csv"
    summary = DEFAULT_OUTPUT_DIR / f"{stem}_guide_quality_summary.txt"
    breakdown = DEFAULT_OUTPUT_DIR / f"{stem}_guide_quality_breakdown.json"
    return output, summary, breakdown


def _first_existing_column(
    fieldnames: list[str],
    preferred: str,
    candidates: tuple[str, ...],
) -> str:
    if preferred:
        if preferred not in fieldnames:
            raise ValueError(f"Column not found: {preferred}")
        return preferred

    for column in candidates:
        if column in fieldnames:
            return column

    raise ValueError(f"None of these columns exist: {', '.join(candidates)}")


def _selected_modes(mode: str) -> set[str]:
    return {"direct", "grounded"} if mode == "both" else {mode}


def _has_existing_guide_judge(row: dict[str, str]) -> bool:
    return any((row.get(column) or "").strip() for column in GUIDE_COLUMNS)


def _row_mode_allowed(row: dict[str, str], modes: set[str]) -> bool:
    if "mode" not in row:
        return True
    return (row.get("mode") or "").strip() in modes


def _is_eligible(
    row: dict[str, str],
    *,
    answer_col: str,
    reference_col: str,
    modes: set[str],
    overwrite: bool,
) -> bool:
    if not _row_mode_allowed(row, modes):
        return False
    if not (row.get(answer_col) or "").strip():
        return False
    if not (row.get(reference_col) or "").strip():
        return False
    if not overwrite and _has_existing_guide_judge(row):
        return False
    return True


def _ensure_fieldnames(fieldnames: list[str] | None) -> list[str]:
    base = list(fieldnames or [])
    for column in GUIDE_COLUMNS:
        if column not in base:
            base.append(column)
    return base


def _short_context(row: dict[str, str]) -> str:
    pieces: list[str] = []
    for column in CONTEXT_COLUMNS:
        value = (row.get(column) or "").strip()
        if not value:
            continue
        value = value.replace("\r", " ").replace("\n", " ")
        if len(value) > 900:
            value = value[:900] + "..."
        pieces.append(f"{column}: {value}")
    return "\n".join(pieces)


def _build_guide_judge_prompt(
    *,
    row: dict[str, str],
    answer_col: str,
    reference_col: str,
) -> str:
    answer = (row.get(answer_col) or "").strip()
    reference = (row.get(reference_col) or "").strip()
    question = (row.get("question") or row.get("query") or "").strip()
    context = _short_context(row)

    return f"""
你是一名严格但公平的博物馆导览讲解质量评审。请判断“模型生成的讲解”是否像一名合格导览员给观众讲解文物。

请只依据给出的题目、参考答案/参考描述、可用上下文和模型生成内容评分；不要因为参考答案措辞不同就机械扣分，但发现事实错误、无依据扩展、跑题、英语夹杂、语气生硬时要明确扣分。

评分维度均为 1-5 分：
- factuality：事实正确性，是否与参考信息一致。
- groundedness：证据约束，是否避免编造和无依据扩展。
- guide_style：导览员风格，是否有讲解感、面向观众、自然亲切。
- clarity：清晰度，是否结构清楚、重点明确。
- completeness：完整度，是否回答了用户问题的关键点。
- fluency：语言流畅度，中文是否自然，无明显乱码或中英混杂。
- engagement：吸引力，是否适合展厅讲解，有一定故事性或感染力。
- overall：综合质量。

pass 为 true 的建议标准：overall >= 4，并且 factuality、groundedness、fluency 都不低于 4。

题目：
{question}

参考答案/参考描述：
{reference}

可用上下文：
{context}

模型生成的讲解：
{answer}

请输出严格 JSON，不要输出 Markdown，不要添加解释性前后缀：
{{
  "factuality": 1-5,
  "groundedness": 1-5,
  "guide_style": 1-5,
  "clarity": 1-5,
  "completeness": 1-5,
  "fluency": 1-5,
  "engagement": 1-5,
  "overall": 1-5,
  "pass": true/false,
  "reason": "用中文简要说明主要优点和主要扣分点"
}}
""".strip()


def _normalize_score(value: object) -> str:
    try:
        score = float(str(value).strip())
    except (TypeError, ValueError):
        return ""
    score = min(max(score, 1.0), 5.0)
    if score.is_integer():
        return str(int(score))
    return f"{score:.2f}".rstrip("0").rstrip(".")


def _parse_score(value: str) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


def _is_pass(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass", "passed"}


def _normalize_bool(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "pass", "passed"}:
        return "true"
    if text in {"false", "0", "no", "fail", "failed"}:
        return "false"
    return "false"


def _judge_row(
    row: dict[str, str],
    *,
    answer_col: str,
    reference_col: str,
) -> dict[str, str]:
    prompt = _build_guide_judge_prompt(
        row=row,
        answer_col=answer_col,
        reference_col=reference_col,
    )
    raw = call_judge_llm(prompt)
    obj = parse_json_object(raw)
    if not obj:
        return {"guide_error": f"invalid_json: {raw[:300]}"}

    result: dict[str, str] = {}
    for key in GUIDE_SCORE_COLUMNS:
        json_key = key.replace("guide_", "")
        result[key] = _normalize_score(obj.get(json_key))

    result["guide_pass"] = _normalize_bool(obj.get("pass", False))
    result["guide_reason"] = str(obj.get("reason", "")).strip()
    result["guide_error"] = ""
    return result


def _group_key(row: dict[str, str], group_cols: list[str]) -> str:
    if not group_cols:
        return "all"
    values = []
    for column in group_cols:
        values.append(f"{column}={(row.get(column) or '').strip() or '<empty>'}")
    return " | ".join(values)


def _init_stats() -> dict[str, Any]:
    return {
        "count": 0,
        "pass": 0,
        "score_sums": defaultdict(float),
    }


def _add_stats(stats: dict[str, Any], row: dict[str, str]) -> None:
    stats["count"] += 1
    stats["pass"] += int(_is_pass(row.get("guide_pass", "")))
    for column in GUIDE_SCORE_COLUMNS:
        stats["score_sums"][column] += _parse_score(row.get(column, ""))


def _stats_to_dict(stats: dict[str, Any]) -> dict[str, Any]:
    count = int(stats["count"])
    return {
        "count": count,
        "pass_rate": round(safe_div(int(stats["pass"]), count), 4),
        **{
            f"avg_{column}": round(
                safe_div(float(stats["score_sums"][column]), count),
                4,
            )
            for column in GUIDE_SCORE_COLUMNS
        },
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    default_output, default_summary, default_breakdown = _default_paths(input_path)
    output_path = Path(args.output) if args.output else default_output
    summary_path = Path(args.summary) if args.summary else default_summary
    breakdown_path = Path(args.breakdown) if args.breakdown else default_breakdown

    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        original_fieldnames = list(reader.fieldnames or [])

    answer_col = _first_existing_column(
        original_fieldnames,
        args.answer_col,
        ANSWER_COLUMNS,
    )
    reference_col = _first_existing_column(
        original_fieldnames,
        args.reference_col,
        REFERENCE_COLUMNS,
    )
    group_cols = [
        column.strip()
        for column in args.group_cols.split(",")
        if column.strip() and column.strip() in original_fieldnames
    ]

    modes = _selected_modes(args.mode)
    eligible_indexes = [
        idx
        for idx, row in enumerate(rows)
        if _is_eligible(
            row,
            answer_col=answer_col,
            reference_col=reference_col,
            modes=modes,
            overwrite=args.overwrite,
        )
    ]
    if args.limit > 0:
        eligible_indexes = eligible_indexes[: args.limit]
    eligible_set = set(eligible_indexes)

    if args.dry_run:
        print(f"Input: {input_path}")
        print(f"Rows in input: {len(rows)}")
        print(f"Answer column: {answer_col}")
        print(f"Reference column: {reference_col}")
        print(f"Group columns: {', '.join(group_cols) if group_cols else '<none>'}")
        print(f"Rows that would be judged: {len(eligible_indexes)}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    breakdown_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = _ensure_fieldnames(original_fieldnames)
    progress = Progress(
        len(eligible_indexes),
        "Guide Quality Judge",
        enabled=not args.no_progress,
    )

    judged = 0
    judge_errors = 0
    skipped_mode = 0
    skipped_empty_answer = 0
    skipped_empty_reference = 0
    skipped_existing = 0
    error_reasons: Counter[str] = Counter()
    overall_stats = _init_stats()
    group_stats: defaultdict[str, dict[str, Any]] = defaultdict(_init_stats)

    try:
        with output_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for idx, row in enumerate(rows):
                if not _row_mode_allowed(row, modes):
                    skipped_mode += 1
                    writer.writerow(row)
                    continue

                if not (row.get(answer_col) or "").strip():
                    skipped_empty_answer += 1
                    writer.writerow(row)
                    continue

                if not (row.get(reference_col) or "").strip():
                    skipped_empty_reference += 1
                    writer.writerow(row)
                    continue

                if idx not in eligible_set:
                    if _has_existing_guide_judge(row) and not args.overwrite:
                        skipped_existing += 1
                    writer.writerow(row)
                    continue

                try:
                    row.update(
                        _judge_row(
                            row,
                            answer_col=answer_col,
                            reference_col=reference_col,
                        )
                    )
                    if row.get("guide_error"):
                        judge_errors += 1
                        error_reasons[row["guide_error"][:160]] += 1
                    else:
                        judged += 1
                except Exception as exc:
                    judge_errors += 1
                    row["guide_error"] = f"judge_error: {exc}"
                    error_reasons[str(exc)[:160]] += 1
                    if args.stop_on_error:
                        raise
                finally:
                    progress.advance()

                writer.writerow(row)

                if not row.get("guide_error"):
                    _add_stats(overall_stats, row)
                    _add_stats(group_stats[_group_key(row, group_cols)], row)
    finally:
        progress.close()

    summary_lines = [
        f"Input: {input_path}",
        f"Output: {output_path}",
        f"Answer column: {answer_col}",
        f"Reference column: {reference_col}",
        f"Group columns: {', '.join(group_cols) if group_cols else '<none>'}",
        f"Rows in input: {len(rows)}",
        f"Rows judged successfully this run: {judged}",
        f"Judge errors: {judge_errors}",
        f"Skipped by mode: {skipped_mode}",
        f"Skipped empty answers: {skipped_empty_answer}",
        f"Skipped empty references: {skipped_empty_reference}",
        f"Skipped existing guide fields: {skipped_existing}",
    ]

    overall_dict = _stats_to_dict(overall_stats)
    summary_lines.append("[overall]")
    for key, value in overall_dict.items():
        summary_lines.append(f"{key}: {value}")

    for group, stats in sorted(group_stats.items()):
        summary_lines.append(f"[{group}]")
        for key, value in _stats_to_dict(stats).items():
            summary_lines.append(f"{key}: {value}")

    summary_lines.append(f"Top judge errors: {summarize_counter(error_reasons)}")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    breakdown = {
        "input": str(input_path),
        "output": str(output_path),
        "answer_column": answer_col,
        "reference_column": reference_col,
        "group_columns": group_cols,
        "rows_in_input": len(rows),
        "rows_judged_successfully_this_run": judged,
        "judge_errors": judge_errors,
        "skipped_by_mode": skipped_mode,
        "skipped_empty_answers": skipped_empty_answer,
        "skipped_empty_references": skipped_empty_reference,
        "skipped_existing_guide_fields": skipped_existing,
        "overall": overall_dict,
        "groups": {
            group: _stats_to_dict(stats)
            for group, stats in sorted(group_stats.items())
        },
        "top_judge_errors": dict(error_reasons.most_common(20)),
    }
    breakdown_path.write_text(
        json.dumps(breakdown, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
