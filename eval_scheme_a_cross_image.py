from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import faiss

from src.config import IMAGE_MIN_GAP, IMAGE_MIN_SCORE, IMAGE_TOP_K
from src.eval_utils import read_jsonl, safe_div, summarize_counter
from src.image_embedder import encode_image, encode_images
from src.image_index import open_local_image
from src.progress import iter_progress


DATASET_PATH = Path("data/multimodal_eval/test_images.jsonl")
OUT_DIR = Path("outputs")
OUT_CSV = OUT_DIR / "eval_scheme_a_cross_image_results.csv"
OUT_SUMMARY = OUT_DIR / "eval_scheme_a_cross_image_summary.txt"
OUT_BREAKDOWN = OUT_DIR / "eval_scheme_a_cross_image_breakdown.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Scheme A image retrieval with different images of the same artifact. "
            "One image per artifact is used as a temporary gallery, and remaining images "
            "are used as queries."
        )
    )
    parser.add_argument(
        "--dataset",
        default=str(DATASET_PATH),
        help="Image-level multimodal dataset. Default: data/multimodal_eval/test_images.jsonl",
    )
    parser.add_argument(
        "--limit-artifacts",
        type=int,
        default=0,
        help="Only use the first N artifacts that have at least two valid images. 0 means all.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=IMAGE_TOP_K,
        help=f"Temporary gallery retrieval top-k. Default: {IMAGE_TOP_K}",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=IMAGE_MIN_SCORE,
        help=f"Confidence minimum top score. Default: {IMAGE_MIN_SCORE}",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=IMAGE_MIN_GAP,
        help=f"Confidence minimum gap between top-1 and top-2. Default: {IMAGE_MIN_GAP}",
    )
    return parser.parse_args()


def _valid_image_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for row in rows:
        image_path = str(row.get("image_path", "")).strip()
        artifact_id = str(row.get("artifact_id", "")).strip()
        if not artifact_id or not image_path:
            continue
        if not Path(image_path).exists():
            continue
        valid.append(row)
    return valid


def _group_by_artifact(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["artifact_id"])].append(row)
    return grouped


def _is_confident(scores: list[float], min_score: float, min_gap: float) -> tuple[int, str]:
    if not scores:
        return 0, "no_candidate"
    if scores[0] < min_score:
        return 0, "low_score"
    if len(scores) >= 2 and scores[0] - scores[1] < min_gap:
        return 0, "small_gap"
    return 1, ""


def _format_candidates(candidates: list[dict[str, Any]], field: str) -> str:
    values: list[str] = []
    for item in candidates:
        value = item.get(field, "")
        if field == "score":
            values.append(f"{float(value):.4f}")
        else:
            values.append(str(value))
    return "|".join(values)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = _valid_image_rows(read_jsonl(args.dataset))
    grouped = _group_by_artifact(rows)
    multi_image_groups = [
        group_rows
        for group_rows in grouped.values()
        if len(group_rows) >= 2
    ]
    multi_image_groups.sort(key=lambda group: str(group[0].get("artifact_id", "")))

    if args.limit_artifacts > 0:
        multi_image_groups = multi_image_groups[: args.limit_artifacts]

    if not multi_image_groups:
        raise RuntimeError(
            "No artifacts with at least two valid images were found. "
            "Run prepare_multimodal_eval_dataset.py and build/download image data first."
        )

    gallery_rows = [group_rows[0] for group_rows in multi_image_groups]
    query_rows = [
        row
        for group_rows in multi_image_groups
        for row in group_rows[1:]
    ]

    gallery_images = [
        open_local_image(str(row["image_path"]))
        for row in iter_progress(gallery_rows, label="Cross Image Gallery Load")
    ]
    gallery_embeddings = encode_images(gallery_images)
    index = faiss.IndexFlatIP(gallery_embeddings.shape[1])
    index.add(gallery_embeddings)

    fieldnames = [
        "query_image_id",
        "query_artifact_id",
        "query_artifact_name",
        "query_image_path",
        "target_gallery_image_id",
        "target_gallery_image_path",
        "retrieved_top1_artifact_id",
        "retrieved_top1_name",
        "retrieved_top1_score",
        "retrieved_artifact_ids",
        "retrieved_names",
        "retrieved_scores",
        "hit_top1",
        "hit_topk",
        "confident",
        "confidence_reason",
    ]

    total = 0
    hit_top1 = 0
    hit_topk = 0
    confident_total = 0
    confident_hit_top1 = 0
    confident_hit_topk = 0
    reason_counter: Counter[str] = Counter()

    with OUT_CSV.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for query_row in iter_progress(query_rows, label="Cross Image Queries"):
            total += 1
            query_image = open_local_image(str(query_row["image_path"]))
            query_embedding = encode_image(query_image)
            search_top_k = min(max(args.top_k, 1), len(gallery_rows))
            scores, indices = index.search(query_embedding, search_top_k)

            candidates: list[dict[str, Any]] = []
            for idx, score in zip(indices[0], scores[0]):
                if idx < 0:
                    continue
                gallery_row = gallery_rows[int(idx)]
                candidates.append(
                    {
                        "artifact_id": gallery_row.get("artifact_id", ""),
                        "artifact_name": gallery_row.get("artifact_name", ""),
                        "image_id": gallery_row.get("image_id", ""),
                        "image_path": gallery_row.get("image_path", ""),
                        "score": float(score),
                    }
                )

            query_artifact_id = str(query_row.get("artifact_id", ""))
            query_artifact_name = str(query_row.get("artifact_name", ""))
            top1 = candidates[0] if candidates else {}
            scores_list = [float(item["score"]) for item in candidates]
            confident, reason = _is_confident(scores_list, args.min_score, args.min_gap)

            row_hit_top1 = int(str(top1.get("artifact_id", "")) == query_artifact_id)
            row_hit_topk = int(
                any(str(item.get("artifact_id", "")) == query_artifact_id for item in candidates)
            )

            hit_top1 += row_hit_top1
            hit_topk += row_hit_topk
            confident_total += confident
            if confident:
                confident_hit_top1 += row_hit_top1
                confident_hit_topk += row_hit_topk
            else:
                reason_counter[reason] += 1

            target_gallery_row = next(
                row
                for row in gallery_rows
                if str(row.get("artifact_id", "")) == query_artifact_id
            )

            writer.writerow(
                {
                    "query_image_id": query_row.get("image_id", ""),
                    "query_artifact_id": query_artifact_id,
                    "query_artifact_name": query_artifact_name,
                    "query_image_path": query_row.get("image_path", ""),
                    "target_gallery_image_id": target_gallery_row.get("image_id", ""),
                    "target_gallery_image_path": target_gallery_row.get("image_path", ""),
                    "retrieved_top1_artifact_id": top1.get("artifact_id", ""),
                    "retrieved_top1_name": top1.get("artifact_name", ""),
                    "retrieved_top1_score": (
                        f"{float(top1.get('score', 0.0)):.4f}" if top1 else ""
                    ),
                    "retrieved_artifact_ids": _format_candidates(candidates, "artifact_id"),
                    "retrieved_names": _format_candidates(candidates, "artifact_name"),
                    "retrieved_scores": _format_candidates(candidates, "score"),
                    "hit_top1": row_hit_top1,
                    "hit_topk": row_hit_topk,
                    "confident": confident,
                    "confidence_reason": reason,
                }
            )

    summary_lines = [
        f"Dataset: {args.dataset}",
        f"Artifacts with >=2 images: {len(multi_image_groups)}",
        f"Gallery images: {len(gallery_rows)}",
        f"Query images: {len(query_rows)}",
        f"Top-k: {args.top_k}",
        f"Min score: {args.min_score}",
        f"Min gap: {args.min_gap}",
        f"Top-1 accuracy: {safe_div(hit_top1, total):.3f}",
        f"Top-k hit rate: {safe_div(hit_topk, total):.3f}",
        f"Confident rate: {safe_div(confident_total, total):.3f}",
        f"Confident Top-1 accuracy: {safe_div(confident_hit_top1, confident_total):.3f}",
        f"Confident Top-k hit rate: {safe_div(confident_hit_topk, confident_total):.3f}",
        f"Top rejection reasons: {summarize_counter(reason_counter)}",
    ]
    OUT_SUMMARY.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    breakdown = {
        "dataset": args.dataset,
        "artifacts_with_multiple_images": len(multi_image_groups),
        "gallery_images": len(gallery_rows),
        "query_images": len(query_rows),
        "top_k": args.top_k,
        "min_score": args.min_score,
        "min_gap": args.min_gap,
        "top1_accuracy": round(safe_div(hit_top1, total), 4),
        "topk_hit_rate": round(safe_div(hit_topk, total), 4),
        "confident_rate": round(safe_div(confident_total, total), 4),
        "confident_top1_accuracy": round(safe_div(confident_hit_top1, confident_total), 4),
        "confident_topk_hit_rate": round(safe_div(confident_hit_topk, confident_total), 4),
        "top_rejection_reasons": reason_counter.most_common(20),
    }
    OUT_BREAKDOWN.write_text(json.dumps(breakdown, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"Detailed results written to: {OUT_CSV}")
    print(f"Summary written to: {OUT_SUMMARY}")
    print(f"Breakdown written to: {OUT_BREAKDOWN}")


if __name__ == "__main__":
    main()
