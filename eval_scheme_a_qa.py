from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

from src.image_index import open_local_image
from src.image_retriever import assess_image_match_confidence, search_image
from src.kb import load_docs
from src.llm import call_llm
from src.prompt import build_prompt
from src.progress import Progress


DATASET_PATH = Path("data/multimodal_eval/test_images.jsonl")
OUT_DIR = Path("outputs/raw")
OUT_CSV = OUT_DIR / "eval_scheme_a_qa_results.csv"
OUT_SUMMARY = OUT_DIR / "eval_scheme_a_qa_summary.txt"
OUT_BREAKDOWN = OUT_DIR / "eval_scheme_a_qa_breakdown.json"
_TEXT_DOCS: list[str] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Scheme A multimodal QA on the generated multimodal dataset."
    )
    parser.add_argument(
        "--dataset",
        default=str(DATASET_PATH),
        help="Path to the image-level multimodal QA dataset. Default: data/multimodal_eval/test_images.jsonl",
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
        "--top-k",
        type=int,
        default=5,
        help="Number of image retrieval candidates to inspect.",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Also call the LLM and score the generated answer.",
    )
    return parser.parse_args()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip()).lower()


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[，。；：、,.!?！？()（）【】\[\]“”\"'·\-]", " ", text or "")
    return [token for token in cleaned.split() if token]


def _extract_field(doc: str, label: str) -> str:
    pattern = rf"【{re.escape(label)}】\s*(.*?)(?=【|$)"
    match = re.search(pattern, doc, flags=re.S)
    return match.group(1).strip() if match else ""


def _get_text_docs() -> list[str]:
    global _TEXT_DOCS
    if _TEXT_DOCS is None:
        _TEXT_DOCS = load_docs()
    return _TEXT_DOCS


def _load_dataset(path: str, limit_images: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if limit_images > 0:
        rows = rows[:limit_images]
    return rows


def _retrieve_exact_docs_by_name(name: str) -> list[tuple[str, float]]:
    normalized_name = _normalize(name)
    if not normalized_name:
        return []

    matches: list[tuple[str, float]] = []
    for doc in _get_text_docs():
        doc_name = _extract_field(doc, "展品名称")
        if _normalize(doc_name) == normalized_name:
            matches.append((doc, 1.0))
    return matches


def _build_query(recognized_name: str, question: str, era: str = "", museum: str = "") -> str:
    prefix_parts = [part.strip() for part in [recognized_name, era, museum] if part and part.strip() and part != "-"]
    prefix = " ".join(prefix_parts)
    if prefix:
        return f"{prefix}。问题：{question.strip()}"
    return question.strip()


def _is_answer_correct(predicted: str, expected: str) -> tuple[bool, float]:
    pred = (predicted or "").strip()
    gold = (expected or "").strip()
    if not pred or not gold:
        return False, 0.0

    pred_norm = _normalize(pred)
    gold_norm = _normalize(gold)
    if pred_norm == gold_norm:
        return True, 1.0

    if len(gold_norm) <= 16 and (gold_norm in pred_norm or pred_norm in gold_norm):
        return True, 0.9

    gold_tokens = _tokenize(gold)
    pred_tokens = _tokenize(pred)
    if gold_tokens and pred_tokens:
        overlap = sum(1 for token in gold_tokens if token in pred_tokens)
        coverage = overlap / max(len(gold_tokens), 1)
        if coverage >= 0.6:
            return True, coverage
        return False, coverage

    return False, 0.0


def _summarize_reasons(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    parts = [f"{reason}: {count}" for reason, count in counter.most_common(10)]
    return "; ".join(parts)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = _load_dataset(args.dataset, args.limit_images)
    if not samples:
        raise RuntimeError(f"没有读取到 QA 数据样本，请检查数据集文件：{args.dataset}")

    progress_total = 0
    for sample in samples:
        qa_pairs = list(sample.get("qa_pairs", []))
        if args.limit_questions > 0:
            qa_pairs = qa_pairs[: args.limit_questions]
        progress_total += len(qa_pairs)
    progress = Progress(progress_total, "Scheme A QA")

    fieldnames = [
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
        "retrieved_doc_name",
        "retrieved_doc_era",
        "fact_answer",
        "fact_correct",
        "fact_score",
        "llm_answer",
        "llm_correct",
        "llm_score",
    ]

    image_total = 0
    question_total = 0
    image_found_sum = 0
    image_confident_sum = 0
    name_correct_sum = 0
    context_found_sum = 0
    fact_correct_sum = 0
    llm_correct_sum = 0
    llm_calls = 0
    rejection_reasons: Counter[str] = Counter()
    field_total_counter: Counter[str] = Counter()
    field_fact_correct_counter: Counter[str] = Counter()
    field_llm_correct_counter: Counter[str] = Counter()
    confusion_counter: Counter[str] = Counter()

    with OUT_CSV.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for sample in samples:
            image_total += 1
            target_name = str(sample.get("artifact_name", "")).strip()
            target_era = str(sample.get("era", "")).strip()
            target_museum = str(sample.get("museum", "")).strip()
            image_path = str(sample.get("image_path", "")).strip()
            qa_pairs = list(sample.get("qa_pairs", []))
            if args.limit_questions > 0:
                qa_pairs = qa_pairs[: args.limit_questions]

            image_found = 0
            image_confident = 0
            confidence_reason = ""
            recognized_name = ""
            recognized_era = ""
            recognized_museum = ""
            recognized_score = ""
            recognized_name_correct = 0
            text_pairs: list[tuple[str, float]] = []

            if image_path and Path(image_path).exists():
                image = open_local_image(image_path)
                matches = search_image(image, top_k=args.top_k)
                image_found = int(bool(matches))
                image_found_sum += image_found

                confident, confidence_reason = assess_image_match_confidence(matches)
                image_confident = int(confident)
                image_confident_sum += image_confident

                if matches:
                    best = matches[0]
                    recognized_name = str(best.get("name", "")).strip()
                    recognized_era = str(best.get("era", "")).strip()
                    recognized_museum = str(best.get("museum", "")).strip()
                    recognized_score = f"{float(best.get('score', 0.0)):.4f}"
                    recognized_name_correct = int(_normalize(recognized_name) == _normalize(target_name))
                    name_correct_sum += recognized_name_correct
                    if not recognized_name_correct:
                        confusion_counter[f"{target_name} -> {recognized_name}"] += 1

                    if confident:
                        text_pairs = _retrieve_exact_docs_by_name(recognized_name)
                else:
                    rejection_reasons["no_image_match"] += 1
            else:
                confidence_reason = "image_missing"
                rejection_reasons["image_missing"] += 1

            if image_found and not image_confident:
                rejection_reasons[confidence_reason or "not_confident"] += 1

            context_found = int(bool(text_pairs))
            context_found_sum += context_found
            retrieved_doc_name = _extract_field(text_pairs[0][0], "展品名称") if text_pairs else ""
            retrieved_doc_era = _extract_field(text_pairs[0][0], "所属时代") if text_pairs else ""

            for question_idx, qa in enumerate(qa_pairs, start=1):
                question_total += 1
                question = str(qa.get("question", "")).strip()
                answer_field = str(qa.get("answer_field", "")).strip()
                gold_answer = str(qa.get("answer", "")).strip()
                field_total_counter[answer_field] += 1

                fact_answer = ""
                fact_correct = 0
                fact_score = 0.0
                llm_answer = ""
                llm_correct = 0
                llm_score = 0.0

                if text_pairs:
                    fact_answer = _extract_field(text_pairs[0][0], answer_field)
                    fact_correct, fact_score = _is_answer_correct(fact_answer, gold_answer)
                    fact_correct = int(fact_correct)
                    fact_correct_sum += fact_correct
                    field_fact_correct_counter[answer_field] += fact_correct

                    if args.with_llm:
                        llm_calls += 1
                        query = _build_query(recognized_name, question, recognized_era, recognized_museum)
                        prompt = build_prompt(query, text_pairs)
                        llm_answer = call_llm(prompt)
                        llm_correct_bool, llm_score = _is_answer_correct(llm_answer, gold_answer)
                        llm_correct = int(llm_correct_bool)
                        llm_correct_sum += llm_correct
                        field_llm_correct_counter[answer_field] += llm_correct

                writer.writerow(
                    {
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
                        "image_found": image_found,
                        "image_confident": image_confident,
                        "confidence_reason": confidence_reason,
                        "recognized_name": recognized_name,
                        "recognized_era": recognized_era,
                        "recognized_museum": recognized_museum,
                        "recognized_score": recognized_score,
                        "recognized_name_correct": recognized_name_correct,
                        "context_found": context_found,
                        "retrieved_doc_name": retrieved_doc_name,
                        "retrieved_doc_era": retrieved_doc_era,
                        "fact_answer": fact_answer,
                        "fact_correct": fact_correct,
                        "fact_score": f"{fact_score:.3f}",
                        "llm_answer": llm_answer,
                        "llm_correct": llm_correct,
                        "llm_score": f"{llm_score:.3f}",
                    }
                )
                progress.advance()

    progress.close()

    summary_lines = [
        f"Dataset: {args.dataset}",
        f"Image samples: {image_total}",
        f"QA samples: {question_total}",
        f"Image found rate: {image_found_sum / image_total:.3f}" if image_total else "Image found rate: 0.000",
        f"Image confident rate: {image_confident_sum / image_total:.3f}" if image_total else "Image confident rate: 0.000",
        f"Recognized name accuracy: {name_correct_sum / image_total:.3f}" if image_total else "Recognized name accuracy: 0.000",
        f"Context found rate: {context_found_sum / image_total:.3f}" if image_total else "Context found rate: 0.000",
        f"Fact answer accuracy: {fact_correct_sum / question_total:.3f}" if question_total else "Fact answer accuracy: 0.000",
        (
            f"LLM answer accuracy: {llm_correct_sum / question_total:.3f}"
            if args.with_llm and question_total
            else "LLM answer accuracy: skipped"
        ),
        f"LLM calls: {llm_calls}",
        f"Top rejection reasons: {_summarize_reasons(rejection_reasons)}",
    ]
    OUT_SUMMARY.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    field_breakdown: dict[str, dict[str, float | int | str]] = {}
    for field_name, total in sorted(field_total_counter.items()):
        field_breakdown[field_name] = {
            "total": total,
            "fact_correct": field_fact_correct_counter[field_name],
            "fact_accuracy": round(field_fact_correct_counter[field_name] / total, 4) if total else 0.0,
            "llm_correct": field_llm_correct_counter[field_name],
            "llm_accuracy": round(field_llm_correct_counter[field_name] / total, 4) if (args.with_llm and total) else 0.0,
        }

    breakdown = {
        "dataset": args.dataset,
        "image_samples": image_total,
        "qa_samples": question_total,
        "image_found_rate": round(image_found_sum / image_total, 4) if image_total else 0.0,
        "image_confident_rate": round(image_confident_sum / image_total, 4) if image_total else 0.0,
        "recognized_name_accuracy": round(name_correct_sum / image_total, 4) if image_total else 0.0,
        "context_found_rate": round(context_found_sum / image_total, 4) if image_total else 0.0,
        "fact_answer_accuracy": round(fact_correct_sum / question_total, 4) if question_total else 0.0,
        "llm_answer_accuracy": round(llm_correct_sum / question_total, 4) if (args.with_llm and question_total) else None,
        "field_breakdown": field_breakdown,
        "top_rejection_reasons": rejection_reasons.most_common(20),
        "top_name_confusions": confusion_counter.most_common(20),
    }
    OUT_BREAKDOWN.write_text(json.dumps(breakdown, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"Detailed results written to: {OUT_CSV}")
    print(f"Summary written to: {OUT_SUMMARY}")
    print(f"Breakdown written to: {OUT_BREAKDOWN}")


if __name__ == "__main__":
    main()
