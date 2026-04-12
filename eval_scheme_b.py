from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.eval_utils import (
    build_text_query,
    contains_expected,
    extract_field,
    normalize_text,
    parse_json_object,
    read_jsonl,
    safe_div,
    score_answer,
    strip_model_artifacts,
    summarize_counter,
)
from src.image_index import open_local_image
from src.image_retriever import assess_image_match_confidence, search_image
from src.llm import call_llm, call_multimodal_llm
from src.prompt import build_multimodal_direct_prompt, build_multimodal_grounded_prompt
from src.progress import Progress
from src.retriever import retrieve


DATASET_PATH = Path("data/multimodal_eval/test_images.jsonl")
OUT_DIR = Path("outputs")
OUT_CSV = OUT_DIR / "eval_scheme_b_results.csv"
OUT_SUMMARY = OUT_DIR / "eval_scheme_b_summary.txt"
OUT_BREAKDOWN = OUT_DIR / "eval_scheme_b_breakdown.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Scheme B multimodal QA on the shared multimodal dataset."
    )
    parser.add_argument(
        "--dataset",
        default=str(DATASET_PATH),
        help="Image-level multimodal dataset. Default: data/multimodal_eval/test_images.jsonl",
    )
    parser.add_argument(
        "--mode",
        choices=["direct", "grounded", "both"],
        default="both",
        help="Scheme B mode to evaluate. Default: both.",
    )
    parser.add_argument(
        "--limit-images",
        type=int,
        default=0,
        help="Only evaluate the first N image samples. 0 means all.",
    )
    parser.add_argument(
        "--limit-questions",
        type=int,
        default=0,
        help="Only evaluate the first N QA pairs for each image. 0 means all.",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=0,
        help="Maximum multimodal LLM calls. 0 means no cap. Useful for slow local models.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Image retrieval candidates used by grounded mode.",
    )
    parser.add_argument(
        "--judge-llm",
        action="store_true",
        help="Use the configured text LLM as an optional semantic judge.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and run grounded-mode retrieval without calling the multimodal LLM.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when a model call fails.",
    )
    return parser.parse_args()


def _selected_modes(mode: str) -> list[str]:
    return ["direct", "grounded"] if mode == "both" else [mode]


def _retrieve_contexts(
    image,
    question: str,
    top_k: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "image_found": 0,
        "image_confident": 0,
        "confidence_reason": "",
        "recognized_name": "",
        "recognized_era": "",
        "recognized_museum": "",
        "recognized_score": "",
        "contexts": [],
        "retrieved_names": "",
        "retrieved_scores": "",
        "context_found": 0,
    }

    matches = search_image(image, top_k=top_k)
    payload["image_found"] = int(bool(matches))
    confident, reason = assess_image_match_confidence(matches)
    payload["image_confident"] = int(confident)
    payload["confidence_reason"] = reason

    if not matches:
        payload["confidence_reason"] = reason or "no_image_match"
        return payload

    best = matches[0]
    payload["recognized_name"] = str(best.get("name", "")).strip()
    payload["recognized_era"] = str(best.get("era", "")).strip()
    payload["recognized_museum"] = str(best.get("museum", "")).strip()
    payload["recognized_score"] = f"{float(best.get('score', 0.0)):.4f}"

    if not confident:
        return payload

    query = build_text_query(
        payload["recognized_name"],
        question,
        payload["recognized_era"],
        payload["recognized_museum"],
    )
    contexts = retrieve(query)
    payload["contexts"] = contexts
    payload["context_found"] = int(bool(contexts))
    payload["retrieved_names"] = "|".join(extract_field(doc, "展品名称") for doc, _score in contexts)
    payload["retrieved_scores"] = "|".join(f"{score:.4f}" for _doc, score in contexts)
    return payload


def _build_judge_prompt(question: str, gold_answer: str, model_answer: str) -> str:
    return f"""
你是一名严格但公平的博物馆问答评测员。请判断模型回答是否正确覆盖标准答案。

要求：
1. 只评估事实是否对齐，不因为表达方式不同而扣分。
2. 如果模型回答编造了与标准答案冲突的事实，应判为不通过。
3. 请只输出 JSON，不要输出其他文字。

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
    judge_raw = call_llm(_build_judge_prompt(question, gold_answer, model_answer))
    obj = parse_json_object(judge_raw)
    if not obj:
        return "", "", judge_raw[:300]

    judge_pass = str(obj.get("pass", "")).lower()
    judge_score = str(obj.get("score", ""))
    judge_reason = str(obj.get("reason", "")).strip()
    return judge_pass, judge_score, judge_reason


def _empty_grounding() -> dict[str, Any]:
    return {
        "image_found": "",
        "image_confident": "",
        "confidence_reason": "",
        "recognized_name": "",
        "recognized_era": "",
        "recognized_museum": "",
        "recognized_score": "",
        "contexts": [],
        "retrieved_names": "",
        "retrieved_scores": "",
        "context_found": "",
    }


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = read_jsonl(args.dataset, args.limit_images)
    if not samples:
        raise RuntimeError(f"没有读取到多模态评测数据，请检查：{args.dataset}")

    selected_modes = _selected_modes(args.mode)
    progress_total = 0
    for sample in samples:
        qa_pairs = list(sample.get("qa_pairs", []))
        if args.limit_questions > 0:
            qa_pairs = qa_pairs[: args.limit_questions]
        progress_total += len(qa_pairs) * len(selected_modes)
    progress = Progress(progress_total, "Scheme B")

    fieldnames = [
        "mode",
        "image_id",
        "artifact_id",
        "target_name",
        "target_era",
        "target_museum",
        "question_idx",
        "question",
        "answer_field",
        "gold_answer",
        "image_path",
        "image_found",
        "image_confident",
        "confidence_reason",
        "recognized_name",
        "recognized_era",
        "recognized_museum",
        "recognized_score",
        "recognized_name_correct",
        "context_found",
        "retrieved_names",
        "retrieved_scores",
        "llm_answer",
        "auto_correct",
        "auto_score",
        "target_name_mentioned",
        "gold_answer_mentioned",
        "judge_pass",
        "judge_score",
        "judge_reason",
        "latency_seconds",
        "error",
    ]

    mode_totals: Counter[str] = Counter()
    mode_correct: Counter[str] = Counter()
    mode_name_mentions: Counter[str] = Counter()
    mode_gold_mentions: Counter[str] = Counter()
    mode_context_found: Counter[str] = Counter()
    mode_image_confident: Counter[str] = Counter()
    mode_errors: Counter[str] = Counter()
    mode_score_sum: defaultdict[str, float] = defaultdict(float)
    rejection_reasons: Counter[str] = Counter()
    llm_calls = 0
    judge_calls = 0
    stop_eval = False

    with OUT_CSV.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for sample in samples:
            if stop_eval:
                break

            target_name = str(sample.get("artifact_name", "")).strip()
            target_era = str(sample.get("era", "")).strip()
            target_museum = str(sample.get("museum", "")).strip()
            image_path = str(sample.get("image_path", "")).strip()
            qa_pairs = list(sample.get("qa_pairs", []))
            if args.limit_questions > 0:
                qa_pairs = qa_pairs[: args.limit_questions]

            if not image_path or not Path(image_path).exists():
                rejection_reasons["image_missing"] += len(qa_pairs)
                progress.advance(len(qa_pairs) * len(selected_modes))
                continue

            image = open_local_image(image_path)
            grounding_cache: dict[str, Any] | None = None

            for question_idx, qa in enumerate(qa_pairs, start=1):
                if stop_eval:
                    break

                question = str(qa.get("question", "")).strip()
                answer_field = str(qa.get("answer_field", "")).strip()
                gold_answer = str(qa.get("answer", "")).strip()

                for mode in selected_modes:
                    if args.max_calls > 0 and llm_calls >= args.max_calls:
                        stop_eval = True
                        break

                    grounding = _empty_grounding()
                    prompt = ""
                    skip_reason = ""

                    if mode == "direct":
                        prompt = build_multimodal_direct_prompt(question)
                    else:
                        if grounding_cache is None:
                            grounding_cache = _retrieve_contexts(image, question, args.top_k)
                        grounding = grounding_cache
                        if not grounding["image_found"]:
                            skip_reason = "no_image_match"
                        elif not grounding["image_confident"]:
                            skip_reason = str(grounding["confidence_reason"] or "image_not_confident")
                        elif not grounding["context_found"]:
                            skip_reason = "context_not_found"
                        else:
                            prompt = build_multimodal_grounded_prompt(question, grounding["contexts"])

                    llm_answer = ""
                    error = skip_reason
                    latency = 0.0
                    auto_correct = 0
                    auto_score = 0.0
                    judge_pass = ""
                    judge_score = ""
                    judge_reason = ""

                    if prompt and not skip_reason and args.dry_run:
                        error = "dry_run"
                    elif prompt and not skip_reason:
                        started = time.perf_counter()
                        try:
                            llm_answer = strip_model_artifacts(call_multimodal_llm(prompt, image))
                            llm_calls += 1
                            error = ""
                        except Exception as exc:
                            error = str(exc)
                            mode_errors[mode] += 1
                            if args.stop_on_error:
                                raise
                        finally:
                            latency = time.perf_counter() - started

                    if llm_answer:
                        correct_bool, auto_score = score_answer(llm_answer, gold_answer)
                        auto_correct = int(correct_bool)

                        if args.judge_llm:
                            try:
                                judge_pass, judge_score, judge_reason = _judge_answer(
                                    question,
                                    gold_answer,
                                    llm_answer,
                                )
                                judge_calls += 1
                            except Exception as exc:
                                judge_reason = f"judge_error: {exc}"
                                if args.stop_on_error:
                                    raise

                    if error:
                        rejection_reasons[error] += 1

                    recognized_name = str(grounding.get("recognized_name", "")).strip()
                    recognized_name_correct = (
                        int(normalize_text(recognized_name) == normalize_text(target_name))
                        if mode == "grounded" and recognized_name
                        else ""
                    )
                    target_name_mentioned = contains_expected(llm_answer, target_name)
                    gold_answer_mentioned = contains_expected(llm_answer, gold_answer)

                    mode_totals[mode] += 1
                    mode_correct[mode] += auto_correct
                    mode_score_sum[mode] += auto_score
                    mode_name_mentions[mode] += target_name_mentioned
                    mode_gold_mentions[mode] += gold_answer_mentioned
                    if mode == "grounded":
                        mode_context_found[mode] += int(bool(grounding.get("context_found")))
                        mode_image_confident[mode] += int(bool(grounding.get("image_confident")))

                    writer.writerow(
                        {
                            "mode": mode,
                            "image_id": sample.get("image_id", ""),
                            "artifact_id": sample.get("artifact_id", ""),
                            "target_name": target_name,
                            "target_era": target_era,
                            "target_museum": target_museum,
                            "question_idx": question_idx,
                            "question": question,
                            "answer_field": answer_field,
                            "gold_answer": gold_answer,
                            "image_path": image_path,
                            "image_found": grounding.get("image_found", ""),
                            "image_confident": grounding.get("image_confident", ""),
                            "confidence_reason": grounding.get("confidence_reason", ""),
                            "recognized_name": recognized_name,
                            "recognized_era": grounding.get("recognized_era", ""),
                            "recognized_museum": grounding.get("recognized_museum", ""),
                            "recognized_score": grounding.get("recognized_score", ""),
                            "recognized_name_correct": recognized_name_correct,
                            "context_found": grounding.get("context_found", ""),
                            "retrieved_names": grounding.get("retrieved_names", ""),
                            "retrieved_scores": grounding.get("retrieved_scores", ""),
                            "llm_answer": llm_answer,
                            "auto_correct": auto_correct,
                            "auto_score": f"{auto_score:.3f}",
                            "target_name_mentioned": target_name_mentioned,
                            "gold_answer_mentioned": gold_answer_mentioned,
                            "judge_pass": judge_pass,
                            "judge_score": judge_score,
                            "judge_reason": judge_reason,
                            "latency_seconds": f"{latency:.2f}",
                            "error": error,
                        }
                    )
                    progress.advance()

    progress.close()

    summary_lines = [
        f"Dataset: {args.dataset}",
        f"Mode: {args.mode}",
        f"Dry run: {args.dry_run}",
        f"Rows evaluated: {sum(mode_totals.values())}",
        f"Multimodal LLM calls: {llm_calls}",
        f"Judge LLM calls: {judge_calls}",
    ]

    for mode in selected_modes:
        total = mode_totals[mode]
        summary_lines.extend(
            [
                f"[{mode}] rows: {total}",
                f"[{mode}] auto accuracy: {safe_div(mode_correct[mode], total):.3f}",
                f"[{mode}] avg auto score: {safe_div(mode_score_sum[mode], total):.3f}",
                f"[{mode}] target name mention rate: {safe_div(mode_name_mentions[mode], total):.3f}",
                f"[{mode}] gold answer mention rate: {safe_div(mode_gold_mentions[mode], total):.3f}",
                f"[{mode}] errors: {mode_errors[mode]}",
            ]
        )
        if mode == "grounded":
            summary_lines.extend(
                [
                    f"[{mode}] image confident rate: {safe_div(mode_image_confident[mode], total):.3f}",
                    f"[{mode}] context found rate: {safe_div(mode_context_found[mode], total):.3f}",
                ]
            )

    summary_lines.append(f"Top skip/error reasons: {summarize_counter(rejection_reasons)}")
    OUT_SUMMARY.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    breakdown = {
        "dataset": args.dataset,
        "mode": args.mode,
        "dry_run": args.dry_run,
        "rows_evaluated": sum(mode_totals.values()),
        "llm_calls": llm_calls,
        "judge_calls": judge_calls,
        "modes": {
            mode: {
                "rows": mode_totals[mode],
                "auto_accuracy": round(safe_div(mode_correct[mode], mode_totals[mode]), 4),
                "avg_auto_score": round(safe_div(mode_score_sum[mode], mode_totals[mode]), 4),
                "target_name_mention_rate": round(
                    safe_div(mode_name_mentions[mode], mode_totals[mode]),
                    4,
                ),
                "gold_answer_mention_rate": round(
                    safe_div(mode_gold_mentions[mode], mode_totals[mode]),
                    4,
                ),
                "errors": mode_errors[mode],
                "image_confident_rate": (
                    round(safe_div(mode_image_confident[mode], mode_totals[mode]), 4)
                    if mode == "grounded"
                    else None
                ),
                "context_found_rate": (
                    round(safe_div(mode_context_found[mode], mode_totals[mode]), 4)
                    if mode == "grounded"
                    else None
                ),
            }
            for mode in selected_modes
        },
        "top_skip_or_error_reasons": rejection_reasons.most_common(20),
    }
    OUT_BREAKDOWN.write_text(json.dumps(breakdown, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"Detailed results written to: {OUT_CSV}")
    print(f"Summary written to: {OUT_SUMMARY}")
    print(f"Breakdown written to: {OUT_BREAKDOWN}")


if __name__ == "__main__":
    main()
