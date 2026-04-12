from __future__ import annotations

import json
from pathlib import Path

import faiss
from PIL import Image

from .config import IMAGE_INDEX_PATH, IMAGE_META_PATH, IMAGE_MIN_GAP, IMAGE_MIN_SCORE, IMAGE_TOP_K
from .image_embedder import encode_image

_image_index = None
_image_meta: list[dict[str, str]] | None = None


def get_image_index():
    global _image_index
    if _image_index is None:
        _image_index = faiss.read_index(IMAGE_INDEX_PATH)
    return _image_index


def get_image_meta() -> list[dict[str, str]]:
    global _image_meta
    if _image_meta is None:
        _image_meta = json.loads(Path(IMAGE_META_PATH).read_text(encoding="utf-8"))
    return _image_meta


def search_image(image: Image.Image, top_k: int = IMAGE_TOP_K) -> list[dict[str, str | float]]:
    query = encode_image(image)
    index = get_image_index()
    meta = get_image_meta()
    scores, indices = index.search(query, top_k)

    grouped: dict[str, dict[str, str | float]] = {}
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0:
            continue
        item = meta[int(idx)]
        key = item.get("detail_url") or item.get("image_url") or str(idx)
        current = grouped.get(key)
        payload: dict[str, str | float] = {
            "name": item.get("name", ""),
            "era": item.get("era", ""),
            "museum": item.get("museum", ""),
            "category": item.get("category", ""),
            "detail_url": item.get("detail_url", ""),
            "image_url": item.get("image_url", ""),
            "local_path": item.get("local_path", ""),
            "score": float(score),
        }
        if current is None or float(current["score"]) < float(score):
            grouped[key] = payload

    return sorted(grouped.values(), key=lambda x: float(x["score"]), reverse=True)


def assess_image_match_confidence(
    matches: list[dict[str, str | float]],
    min_score: float = IMAGE_MIN_SCORE,
    min_gap: float = IMAGE_MIN_GAP,
) -> tuple[bool, str]:
    if not matches:
        return False, "没有找到相似文物。"

    best_score = float(matches[0]["score"])
    if best_score < min_score:
        return (
            False,
            f"最高相似度仅为 {best_score:.4f}，低于识别阈值 {min_score:.2f}，暂时无法可靠识别这件文物。",
        )

    if len(matches) >= 2:
        second_score = float(matches[1]["score"])
        if best_score - second_score < min_gap:
            return (
                False,
                "前两名候选的相似度过于接近，系统暂时无法稳定区分具体文物。",
            )

    return True, ""
