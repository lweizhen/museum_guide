from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from src.image_embedder import encode_image, encode_text
from src.image_index import open_local_image
from src.image_retriever import assess_image_match_confidence, search_image
from src.kb import load_docs
from src.llm import call_llm
from src.prompt import build_prompt
from src.progress import iter_progress


DATASET_PATH = Path("data/multimodal_eval/test_images.jsonl")
OUT_DIR = Path("outputs/raw")
OUT_CSV = OUT_DIR / "eval_scheme_a_caption_results.csv"
OUT_SUMMARY = OUT_DIR / "eval_scheme_a_caption_summary.txt"
OUT_BREAKDOWN = OUT_DIR / "eval_scheme_a_caption_breakdown.json"
_TEXT_DOCS: list[str] | None = None


DESCRIPTION_FIELDS = [
    "功能用途",
    "历史意义",
    "文化价值",
    "纹饰与造型",
    "历史背景",
    "补充信息",
    "故事传说",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Scheme A caption generation on multimodal image samples."
    )
    parser.add_argument(
        "--dataset",
        default=str(DATASET_PATH),
        help="Path to the image-level multimodal dataset. Default: data/multimodal_eval/test_images.jsonl",
    )
    parser.add_argument(
        "--limit-images",
        type=int,
        default=0,
        help="Only evaluate the first N image samples. 0 means all.",
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
        help="Use the LLM to generate the final caption instead of the extractive baseline.",
    )
    return parser.parse_args()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip()).lower()


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


def _build_extractive_caption(doc: str) -> str:
    name = _extract_field(doc, "展品名称")
    era = _extract_field(doc, "所属时代")
    category = _extract_field(doc, "类别")
    museum = _extract_field(doc, "馆藏单位")

    lead = "，".join([part for part in [name, era, category, museum] if part])
    details = [_extract_field(doc, field) for field in DESCRIPTION_FIELDS]
    details = [value for value in details if value]

    if lead and details:
        return f"{lead}。{' '.join(details)}"
    if lead:
        return lead
    return " ".join(details)


def _build_caption_query(name: str, era: str, museum: str) -> str:
    parts = [part.strip() for part in [name, era, museum] if part and part.strip() and part != "-"]
    prefix = " ".join(parts)
    if prefix:
        return f"请根据资料为这件文物写一段简洁、准确的介绍：{prefix}"
    return "请根据资料为这件文物写一段简洁、准确的介绍。"


def _clipscore(image_path: str, text: str) -> float:
    if not text.strip():
        return 0.0
    image = open_local_image(image_path)
    image_emb = encode_image(image)[0]
    text_emb = encode_text(text)[0]
    return float(np.dot(image_emb, text_emb))


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def _rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = list(prediction.strip())
    ref_tokens = list(reference.strip())
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_len(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _coverage_score(prediction: str, reference: str) -> float:
    ref_tokens = [token for token in re.split(r"[，。；：、\s]+", reference) if token]
    pred_norm = _normalize(prediction)
    if not ref_tokens or not pred_norm:
        return 0.0
    hit = sum(1 for token in ref_tokens if _normalize(token) in pred_norm)
    return hit / len(ref_tokens)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = _load_dataset(args.dataset, args.limit_images)
    if not samples:
        raise RuntimeError(f"没有读取到描述评测样本，请检查数据集文件：{args.dataset}")

    fieldnames = [
        "image_id",
        "artifact_id",
        "target_name",
        "target_era",
        "target_museum",
        "image_path",
        "image_found",
        "image_confident",
        "confidence_reason",
        "recognized_name",
        "recognized_era",
        "recognized_museum",
        "recognized_score",
        "context_found",
        "reference_description",
        "generated_description",
        "rouge_l_f1",
        "coverage_score",
        "clipscore",
    ]

    image_total = 0
    image_found_sum = 0
    image_confident_sum = 0
    context_found_sum = 0
    rouge_sum = 0.0
    coverage_sum = 0.0
    clipscore_sum = 0.0
    rejection_reasons: Counter[str] = Counter()
    artifact_scores: list[tuple[str, float, float, float]] = []

    with OUT_CSV.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for sample in iter_progress(samples, label="Scheme A Caption"):
            image_total += 1
            target_name = str(sample.get("artifact_name", "")).strip()
            target_era = str(sample.get("era", "")).strip()
            target_museum = str(sample.get("museum", "")).strip()
            image_path = str(sample.get("image_path", "")).strip()
            reference_description = str(sample.get("reference_description", "")).strip()

            image_found = 0
            image_confident = 0
            confidence_reason = ""
            recognized_name = ""
            recognized_era = ""
            recognized_museum = ""
            recognized_score = ""
            context_found = 0
            generated_description = ""
            rouge_l_f1 = 0.0
            coverage_score = 0.0
            clipscore = 0.0

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

                    if confident:
                        text_pairs = _retrieve_exact_docs_by_name(recognized_name)
                        context_found = int(bool(text_pairs))
                        context_found_sum += context_found

                        if text_pairs:
                            if args.with_llm:
                                query = _build_caption_query(recognized_name, recognized_era, recognized_museum)
                                prompt = build_prompt(query, text_pairs)
                                generated_description = call_llm(prompt)
                            else:
                                generated_description = _build_extractive_caption(text_pairs[0][0])

                            rouge_l_f1 = _rouge_l_f1(generated_description, reference_description)
                            coverage_score = _coverage_score(generated_description, reference_description)
                            clipscore = _clipscore(image_path, generated_description)

                            rouge_sum += rouge_l_f1
                            coverage_sum += coverage_score
                            clipscore_sum += clipscore
                            artifact_scores.append(
                                (target_name, rouge_l_f1, coverage_score, clipscore)
                            )
                    else:
                        rejection_reasons[confidence_reason or "not_confident"] += 1
                else:
                    rejection_reasons["no_image_match"] += 1
            else:
                confidence_reason = "image_missing"
                rejection_reasons["image_missing"] += 1

            writer.writerow(
                {
                    "image_id": sample.get("image_id", ""),
                    "artifact_id": sample.get("artifact_id", ""),
                    "target_name": target_name,
                    "target_era": target_era,
                    "target_museum": target_museum,
                    "image_path": image_path,
                    "image_found": image_found,
                    "image_confident": image_confident,
                    "confidence_reason": confidence_reason,
                    "recognized_name": recognized_name,
                    "recognized_era": recognized_era,
                    "recognized_museum": recognized_museum,
                    "recognized_score": recognized_score,
                    "context_found": context_found,
                    "reference_description": reference_description,
                    "generated_description": generated_description,
                    "rouge_l_f1": f"{rouge_l_f1:.4f}",
                    "coverage_score": f"{coverage_score:.4f}",
                    "clipscore": f"{clipscore:.4f}",
                }
            )

    valid_caption_count = len(artifact_scores)
    avg_rouge = rouge_sum / valid_caption_count if valid_caption_count else 0.0
    avg_coverage = coverage_sum / valid_caption_count if valid_caption_count else 0.0
    avg_clipscore = clipscore_sum / valid_caption_count if valid_caption_count else 0.0
    lowest_cases = sorted(artifact_scores, key=lambda item: item[1])[:20]

    summary_lines = [
        f"Dataset: {args.dataset}",
        f"Image samples: {image_total}",
        f"Image found rate: {image_found_sum / image_total:.3f}" if image_total else "Image found rate: 0.000",
        f"Image confident rate: {image_confident_sum / image_total:.3f}" if image_total else "Image confident rate: 0.000",
        f"Context found rate: {context_found_sum / image_total:.3f}" if image_total else "Context found rate: 0.000",
        f"Captioned samples: {valid_caption_count}",
        f"Average ROUGE-L F1: {avg_rouge:.4f}",
        f"Average Coverage: {avg_coverage:.4f}",
        f"Average CLIPScore: {avg_clipscore:.4f}",
        f"Top rejection reasons: {json.dumps(rejection_reasons.most_common(10), ensure_ascii=False)}",
    ]
    OUT_SUMMARY.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    breakdown = {
        "dataset": args.dataset,
        "image_samples": image_total,
        "captioned_samples": valid_caption_count,
        "image_found_rate": round(image_found_sum / image_total, 4) if image_total else 0.0,
        "image_confident_rate": round(image_confident_sum / image_total, 4) if image_total else 0.0,
        "context_found_rate": round(context_found_sum / image_total, 4) if image_total else 0.0,
        "average_rouge_l_f1": round(avg_rouge, 4),
        "average_coverage": round(avg_coverage, 4),
        "average_clipscore": round(avg_clipscore, 4),
        "with_llm": args.with_llm,
        "top_rejection_reasons": rejection_reasons.most_common(20),
        "lowest_rouge_cases": [
            {
                "artifact_name": name,
                "rouge_l_f1": round(rouge, 4),
                "coverage_score": round(coverage, 4),
                "clipscore": round(clip, 4),
            }
            for name, rouge, coverage, clip in lowest_cases
        ],
    }
    OUT_BREAKDOWN.write_text(json.dumps(breakdown, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"Detailed results written to: {OUT_CSV}")
    print(f"Summary written to: {OUT_SUMMARY}")
    print(f"Breakdown written to: {OUT_BREAKDOWN}")


if __name__ == "__main__":
    main()
