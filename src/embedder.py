from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBED_MODEL_NAME

_embed_model: SentenceTransformer | None = None


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


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        model_ref = _resolve_local_model_path(EMBED_MODEL_NAME)
        _embed_model = SentenceTransformer(model_ref, local_files_only=True)
    return _embed_model


def encode_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        raise RuntimeError("encode_texts 收到空 texts，请检查知识库读取是否正确。")

    model = get_embed_model()
    emb = model.encode(texts)

    emb = np.asarray(emb, dtype="float32")

    # 保证是二维：(n, dim)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)

    faiss.normalize_L2(emb)
    return emb
