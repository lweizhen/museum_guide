"""文本向量编码模块。

文本检索链路会把用户问题和知识库文物块编码成向量，再用 FAISS 做语义检索。
这里对 Hugging Face 缓存做了离线优先处理，适合远程 GPU 或无公网环境复现实验。
"""

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
    """把模型名解析为本地可用路径。

    如果传入的是实际目录，直接返回；否则在 Hugging Face 缓存中查找最新快照。
    找不到时返回原模型名，由 SentenceTransformer 自行处理。
    """
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
    """懒加载文本向量模型。

    第一次调用时加载模型，后续复用全局对象，避免评测时重复加载模型。
    """
    global _embed_model
    if _embed_model is None:
        model_ref = _resolve_local_model_path(EMBED_MODEL_NAME)
        _embed_model = SentenceTransformer(model_ref, local_files_only=True)
    return _embed_model


def encode_texts(texts: list[str]) -> np.ndarray:
    """把文本列表编码成 L2 归一化后的 float32 向量矩阵。

    返回形状为 `(文本数量, 向量维度)`，可直接用于 FAISS 内积相似度检索。
    """
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
