from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from .config import IMAGE_MODEL_NAME

_image_model: SentenceTransformer | None = None


def _resolve_local_model_path(model_name: str) -> str:
    candidate = Path(model_name)
    if candidate.exists():
        return str(candidate)

    cache_roots = []
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        cache_roots.append(Path(hf_home) / "hub")
    cache_roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    repo_ids = [model_name]
    if "/" not in model_name:
        repo_ids.append(f"sentence-transformers/{model_name}")

    for cache_root in cache_roots:
        for repo_id in repo_ids:
            repo_dir = cache_root / f"models--{repo_id.replace('/', '--')}" / "snapshots"
            if not repo_dir.exists():
                continue

            snapshots = [path for path in repo_dir.iterdir() if path.is_dir()]
            if snapshots:
                newest = max(snapshots, key=lambda path: path.stat().st_mtime)
                return str(newest)

    return model_name


def get_image_model() -> SentenceTransformer:
    global _image_model
    if _image_model is None:
        model_ref = _resolve_local_model_path(IMAGE_MODEL_NAME)
        _image_model = SentenceTransformer(model_ref, local_files_only=True)
    return _image_model


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    emb = np.asarray(embeddings, dtype="float32")
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    faiss.normalize_L2(emb)
    return emb


def encode_images(images: Iterable[Image.Image]) -> np.ndarray:
    image_list = list(images)
    if not image_list:
        raise RuntimeError("encode_images 需要至少一张图片。")
    model = get_image_model()
    embeddings = model.encode(image_list)
    return _normalize_embeddings(embeddings)


def encode_image(image: Image.Image) -> np.ndarray:
    return encode_images([image])


def encode_texts(texts: Iterable[str]) -> np.ndarray:
    text_list = list(texts)
    if not text_list:
        raise RuntimeError("encode_texts 输入不能为空。")
    model = get_image_model()
    embeddings = model.encode(text_list)
    return _normalize_embeddings(embeddings)


def encode_text(text: str) -> np.ndarray:
    return encode_texts([text])
