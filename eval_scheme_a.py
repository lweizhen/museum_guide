from __future__ import annotations

import argparse
import csv
import os
import re
import time
from pathlib import Path

from src.image_index import open_local_image
from src.image_retriever import assess_image_match_confidence, get_image_meta, search_image
from src.kb import load_docs
from src.llm import call_llm
from src.prompt import build_prompt
from src.progress import iter_progress
from src.retriever import retrieve


OUT_DIR = os.path.join("outputs", "raw")
OUT_CSV = os.path.join(OUT_DIR, "eval_scheme_a_results.csv")
OUT_SUMMARY = os.path.join(OUT_DIR, "eval_scheme_a_summary.txt")
_TEXT_DOCS: list[str] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Scheme A: image retrieval -> linked text retrieval -> optional LLM answer.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only evaluate the first N image samples from image metadata. 0 means all.",
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
        help="Also call the LLM and record whether the answer mentions the target artifact name.",
    )
    return parser.parse_args()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip())


def _extract_field(doc: str, label: str) -> str:
    pattern = rf"【{re.escape(label)}】(.*?)(?=【|$)"
    match = re.search(pattern, doc, flags=re.S)
    return match.group(1).strip() if match else ""


def _build_image_query(match: dict[str, str | float], user_question: str = "") -> str:
    name = str(match.get("name", "")).strip()
    era = str(match.get("era", "")).strip()
    museum = str(match.get("museum", "")).strip()

    prefix_parts = [part for part in [name, era, museum] if part and part != "-"]
    prefix = " ".join(prefix_parts)

    if user_question.strip():
        return f"请结合这件文物的信息回答：{prefix}。{user_question.strip()}" if prefix else user_question.strip()

    if prefix:
        return f"请介绍这件文物：{prefix}"
    return "请介绍这件文物。"


def _safe_div(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def _load_samples(limit: int) -> list[dict[str, str]]:
    samples = list(get_image_meta())
    if limit > 0:
        samples = samples[:limit]
    return samples


def _get_text_docs() -> list[str]:
    global _TEXT_DOCS
    if _TEXT_DOCS is None:
        _TEXT_DOCS = load_docs()
    return _TEXT_DOCS


def _retrieve_scheme_a_text_contexts(query: str, recognized_name: str) -> list[tuple[str, float]]:
    normalized_name = _normalize(recognized_name)
    if normalized_name:
        exact_matches: list[tuple[str, float]] = []
        for doc in _get_text_docs():
            doc_name = _extract_field(doc, "展品名称")
            if _normalize(doc_name) == normalized_name:
                exact_matches.append((doc, 1.0))
        if exact_matches:
            return exact_matches
        return []

    return retrieve(query)


def main() -> None:
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    samples = _load_samples(args.limit)
    if not samples:
        raise RuntimeError("没有可评测的图片样本，请先运行 `python build_image_index.py`。")

    fieldnames = [
        "sample_id",
        "target_name",
        "target_era",
        "target_museum",
        "target_detail_url",
        "image_path",
        "image_found",
        "image_confident",
        "confidence_reason",
        "image_top1_name",
        "image_top1_era",
        "image_top1_museum",
        "image_top1_score",
        "image_hit_top1",
        "image_hit_top3",
        "linked_query",
        "text_retrieved_cnt",
        "text_top1_name",
        "text_top1_era",
        "text_hit_top1_name",
        "text_hit_topk_name",
        "text_hit_top1_name_era",
        "answer_mentions_target",
        "answer",
    ]

    total = 0
    image_found_sum = 0
    image_confident_sum = 0
    image_hit1_sum = 0
    image_hit3_sum = 0
    text_hit1_name_sum = 0
    text_hitk_name_sum = 0
    text_hit1_name_era_sum = 0
    answer_mentions_sum = 0
    llm_calls = 0

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for idx, sample in enumerate(iter_progress(samples, label="Scheme A"), start=1):
            total += 1
            target_name = (sample.get("name") or "").strip()
            target_era = (sample.get("era") or "").strip()
            target_museum = (sample.get("museum") or "").strip()
            target_detail_url = (sample.get("detail_url") or "").strip()
            image_path = (sample.get("local_path") or "").strip()

            image_found = 0
            image_confident = 0
            confidence_reason = ""
            image_top1_name = ""
            image_top1_era = ""
            image_top1_museum = ""
            image_top1_score = ""
            image_hit_top1 = 0
            image_hit_top3 = 0
            linked_query = ""
            text_retrieved_cnt = 0
            text_top1_name = ""
            text_top1_era = ""
            text_hit_top1_name = 0
            text_hit_topk_name = 0
            text_hit_top1_name_era = 0
            answer_mentions_target = 0
            answer = ""

            if not image_path or not Path(image_path).exists():
                confidence_reason = "图片文件不存在"
            else:
                image = open_local_image(image_path)
                matches = search_image(image, top_k=args.top_k)
                image_found = int(bool(matches))
                image_found_sum += image_found

                confident, confidence_reason = assess_image_match_confidence(matches)
                image_confident = int(confident)
                image_confident_sum += image_confident

                if matches:
                    best_match = matches[0]
                    image_top1_name = str(best_match.get("name", "")).strip()
                    image_top1_era = str(best_match.get("era", "")).strip()
                    image_top1_museum = str(best_match.get("museum", "")).strip()
                    image_top1_score = f"{float(best_match.get('score', 0.0)):.4f}"

                    image_hit_top1 = int(
                        target_detail_url
                        and str(best_match.get("detail_url", "")).strip() == target_detail_url
                    )
                    image_hit_top3 = int(
                        any(
                            target_detail_url
                            and str(item.get("detail_url", "")).strip() == target_detail_url
                            for item in matches[:3]
                        )
                    )
                    image_hit1_sum += image_hit_top1
                    image_hit3_sum += image_hit_top3

                    linked_query = _build_image_query(best_match)
                    text_pairs = _retrieve_scheme_a_text_contexts(linked_query, image_top1_name)
                    text_retrieved_cnt = len(text_pairs)
                    retrieved_docs = [doc for doc, _ in text_pairs]

                    if retrieved_docs:
                        text_top1_name = _extract_field(retrieved_docs[0], "展品名称")
                        text_top1_era = _extract_field(retrieved_docs[0], "所属时代")
                        retrieved_names = [_extract_field(doc, "展品名称") for doc in retrieved_docs]
                        retrieved_pairs = [
                            (_extract_field(doc, "展品名称"), _extract_field(doc, "所属时代"))
                            for doc in retrieved_docs
                        ]

                        text_hit_top1_name = int(_normalize(text_top1_name) == _normalize(target_name))
                        text_hit_topk_name = int(
                            any(_normalize(name) == _normalize(target_name) for name in retrieved_names)
                        )
                        text_hit_top1_name_era = int(
                            _normalize(text_top1_name) == _normalize(target_name)
                            and _normalize(text_top1_era) == _normalize(target_era)
                        )

                        text_hit1_name_sum += text_hit_top1_name
                        text_hitk_name_sum += text_hit_topk_name
                        text_hit1_name_era_sum += text_hit_top1_name_era

                        if args.with_llm:
                            llm_calls += 1
                            prompt = build_prompt(linked_query, text_pairs)
                            answer = call_llm(prompt)
                            answer_mentions_target = int(_normalize(target_name) in _normalize(answer))
                            answer_mentions_sum += answer_mentions_target

            writer.writerow(
                {
                    "sample_id": idx,
                    "target_name": target_name,
                    "target_era": target_era,
                    "target_museum": target_museum,
                    "target_detail_url": target_detail_url,
                    "image_path": image_path,
                    "image_found": image_found,
                    "image_confident": image_confident,
                    "confidence_reason": confidence_reason,
                    "image_top1_name": image_top1_name,
                    "image_top1_era": image_top1_era,
                    "image_top1_museum": image_top1_museum,
                    "image_top1_score": image_top1_score,
                    "image_hit_top1": image_hit_top1,
                    "image_hit_top3": image_hit_top3,
                    "linked_query": linked_query,
                    "text_retrieved_cnt": text_retrieved_cnt,
                    "text_top1_name": text_top1_name,
                    "text_top1_era": text_top1_era,
                    "text_hit_top1_name": text_hit_top1_name,
                    "text_hit_topk_name": text_hit_topk_name,
                    "text_hit_top1_name_era": text_hit_top1_name_era,
                    "answer_mentions_target": answer_mentions_target,
                    "answer": (answer or "").replace("\n", " ").strip(),
                }
            )

    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    summary_lines = [
        f"Time: {now}",
        f"Total samples: {total}",
        f"Image found rate: {_safe_div(image_found_sum, total):.3f}",
        f"Image confident rate: {_safe_div(image_confident_sum, total):.3f}",
        f"Image Top1 Acc: {_safe_div(image_hit1_sum, total):.3f}",
        f"Image Top3 Acc: {_safe_div(image_hit3_sum, total):.3f}",
        f"Text Retrieval Top1 Name Acc: {_safe_div(text_hit1_name_sum, total):.3f}",
        f"Text Retrieval TopK Name Acc: {_safe_div(text_hitk_name_sum, total):.3f}",
        f"Text Retrieval Top1 Name+Era Acc: {_safe_div(text_hit1_name_era_sum, total):.3f}",
    ]

    if args.with_llm:
        summary_lines.append(
            f"LLM Mention Rate: {_safe_div(answer_mentions_sum, llm_calls):.3f} "
            f"(calls={llm_calls})"
        )
    else:
        summary_lines.append("LLM Mention Rate: skipped")

    summary = "\n".join(summary_lines) + "\n"
    Path(OUT_SUMMARY).write_text(summary, encoding="utf-8")

    print(summary)
    print(f"[OK] Results written to {OUT_CSV}")
    print(f"[OK] Summary written to {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
