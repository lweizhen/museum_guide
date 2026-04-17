from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.eval_utils import parse_json_object, safe_div, summarize_counter
from src.llm import call_judge_llm
from src.progress import Progress


DEFAULT_INPUT = Path("outputs/raw/eval_scheme_b_results.csv")
DEFAULT_OUTPUT = Path("outputs/judged/eval_scheme_b_judged_results.csv")
DEFAULT_SUMMARY = Path("outputs/judged/eval_scheme_b_judged_summary.txt")
DEFAULT_BREAKDOWN = Path("outputs/judged/eval_scheme_b_judged_breakdown.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run semantic judge scoring on existing Scheme B CSV results."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Existing Scheme B result CSV. Default: outputs/raw/eval_scheme_b_results.csv",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Judged result CSV. Default: outputs/judged/eval_scheme_b_judged_results.csv",
    )
    parser.add_argument(
        "--summary",
        default=str(DEFAULT_SUMMARY),
        help="Judged summary TXT. Default: outputs/judged/eval_scheme_b_judged_summary.txt",
    )
    parser.add_argument(
        "--breakdown",
        default=str(DEFAULT_BREAKDOWN),
        help="Judged breakdown JSON. Default: outputs/judged/eval_scheme_b_judged_breakdown.json",
    )
    parser.add_argument(
        "--mode",
        choices=["direct", "grounded", "both"],
        default="both",
        help="Rows to judge by Scheme B mode. Default: both.",
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
        help="Re-judge rows even when judge_pass or judge_score already exists.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when a judge model call fails.",
    )
    return parser.parse_args()


def _selected_modes(mode: str) -> set[str]:
    return {"direct", "grounded"} if mode == "both" else {mode}


def _build_judge_prompt(question: str, gold_answer: str, model_answer: str) -> str:
    return f"""
你是一名严格但公平的博物馆问答评测员。请判断模型回答是否正确覆盖标准答案。

评分要求：
1. 只评估事实是否对齐，不因为表达方式不同而扣分。
2. 如果模型回答包含与标准答案冲突的事实，应该判为不通过。
3. 如果模型回答只是更详细，但核心事实与标准答案一致，可以判为通过。
4. 如果标准答案较短，模型回答只要明确包含该关键信息即可通过。
5. 请只输出 JSON，不要输出其他文字。

【问题】
{question}

【标准答案】
{gold_answer}

【模型回答】
{model_answer}

请输出：
{{"pass": true/false, "score": 0到1之间的小数, "reason": "简短原因"}}
""".strip()


def _judge_answer(question: str, gold_answer: str, model_answer: str) -> tuple[str, str, str]:
    judge_raw = call_judge_llm(_build_judge_prompt(question, gold_answer, model_answer))
    obj = parse_json_object(judge_raw)
    if not obj:
        return "", "", judge_raw[:300]

    judge_pass = str(obj.get("pass", "")).lower()
    judge_score = str(obj.get("score", "")).strip()
    judge_reason = str(obj.get("reason", "")).strip()
    return judge_pass, judge_score, judge_reason


def _has_existing_judge(row: dict[str, str]) -> bool:
    return bool((row.get("judge_pass") or "").strip() or (row.get("judge_score") or "").strip())


def _is_eligible(row: dict[str, str], modes: set[str], overwrite: bool) -> bool:
    if (row.get("mode") or "").strip() not in modes:
        return False
    if not (row.get("llm_answer") or "").strip():
        return False
    if not overwrite and _has_existing_judge(row):
        return False
    return True


def _parse_score(value: str) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


def _is_pass(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass", "passed"}


def _ensure_fieldnames(fieldnames: list[str] | None) -> list[str]:
    base = list(fieldnames or [])
    for name in ["judge_pass", "judge_score", "judge_reason"]:
        if name not in base:
            base.append(name)
    return base


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    breakdown_path = Path(args.breakdown)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    breakdown_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = _ensure_fieldnames(reader.fieldnames)

    modes = _selected_modes(args.mode)
    eligible_indexes = [
        idx for idx, row in enumerate(rows) if _is_eligible(row, modes, args.overwrite)
    ]
    if args.limit > 0:
        eligible_indexes = eligible_indexes[: args.limit]
    eligible_set = set(eligible_indexes)

    progress = Progress(len(eligible_indexes), "Judge Scheme B")
    judged = 0
    judge_errors = 0
    skipped_existing = 0
    skipped_empty = 0
    skipped_mode = 0
    pass_counts: Counter[str] = Counter()
    total_counts: Counter[str] = Counter()
    score_sums: defaultdict[str, float] = defaultdict(float)
    error_reasons: Counter[str] = Counter()

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(rows):
            mode = (row.get("mode") or "").strip()
            if mode not in modes:
                skipped_mode += 1
                writer.writerow(row)
                continue

            if not (row.get("llm_answer") or "").strip():
                skipped_empty += 1
                writer.writerow(row)
                continue

            if idx not in eligible_set:
                if _has_existing_judge(row) and not args.overwrite:
                    skipped_existing += 1
                writer.writerow(row)
                continue

            try:
                judge_pass, judge_score, judge_reason = _judge_answer(
                    row.get("question", ""),
                    row.get("gold_answer", ""),
                    row.get("llm_answer", ""),
                )
                row["judge_pass"] = judge_pass
                row["judge_score"] = judge_score
                row["judge_reason"] = judge_reason
                judged += 1
            except Exception as exc:
                judge_errors += 1
                row["judge_reason"] = f"judge_error: {exc}"
                error_reasons[str(exc)[:160]] += 1
                if args.stop_on_error:
                    raise
            finally:
                progress.advance()

            writer.writerow(row)

            if row.get("judge_score") or row.get("judge_pass"):
                total_counts[mode] += 1
                pass_counts[mode] += int(_is_pass(row.get("judge_pass", "")))
                score_sums[mode] += _parse_score(row.get("judge_score", ""))

    progress.close()

    summary_lines = [
        f"Input: {input_path}",
        f"Output: {output_path}",
        f"Mode: {args.mode}",
        f"Rows in input: {len(rows)}",
        f"Rows judged this run: {judged}",
        f"Judge errors: {judge_errors}",
        f"Skipped by mode: {skipped_mode}",
        f"Skipped empty answers: {skipped_empty}",
        f"Skipped existing judge fields: {skipped_existing}",
    ]
    for mode in sorted(modes):
        total = total_counts[mode]
        summary_lines.extend(
            [
                f"[{mode}] judged rows: {total}",
                f"[{mode}] judge pass rate: {safe_div(pass_counts[mode], total):.3f}",
                f"[{mode}] avg judge score: {safe_div(score_sums[mode], total):.3f}",
            ]
        )
    summary_lines.append(f"Top judge errors: {summarize_counter(error_reasons)}")

    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    breakdown: dict[str, Any] = {
        "input": str(input_path),
        "output": str(output_path),
        "mode": args.mode,
        "rows_in_input": len(rows),
        "rows_judged_this_run": judged,
        "judge_errors": judge_errors,
        "skipped_by_mode": skipped_mode,
        "skipped_empty_answers": skipped_empty,
        "skipped_existing_judge_fields": skipped_existing,
        "modes": {
            mode: {
                "judged_rows": total_counts[mode],
                "judge_pass_rate": round(safe_div(pass_counts[mode], total_counts[mode]), 4),
                "avg_judge_score": round(safe_div(score_sums[mode], total_counts[mode]), 4),
            }
            for mode in sorted(modes)
        },
        "top_judge_errors": error_reasons.most_common(20),
    }
    breakdown_path.write_text(
        json.dumps(breakdown, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n".join(summary_lines))
    print(f"Judged CSV written to: {output_path}")
    print(f"Summary written to: {summary_path}")
    print(f"Breakdown written to: {breakdown_path}")


if __name__ == "__main__":
    main()
