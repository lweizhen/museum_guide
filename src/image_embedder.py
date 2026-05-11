"""图像与图文检索向量编码模块。

图片检索链路使用同一个 SentenceTransformer/CLIP 风格模型编码图片和文本。
编码结果会做 L2 归一化，便于后续使用 FAISS 计算相似度。
"""

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
    """把图像编码模型名解析为本地路径。

    远程机器经常处于离线或弱联网环境，因此这里优先查找本地模型目录和
    Hugging Face 缓存，减少评测时的网络依赖。
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


def get_image_model() -> SentenceTransformer:
    """懒加载图像编码模型。

    图片索引构建、图片检索、文本到图像空间检索都会复用这个模型对象。
    """
    global _image_model
    if _image_model is None:
        model_ref = _resolve_local_model_path(IMAGE_MODEL_NAME)
        _image_model = SentenceTransformer(model_ref, local_files_only=True)
    return _image_model


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """把模型输出转换为二维 float32 矩阵，并做 L2 归一化。"""
    emb = np.asarray(embeddings, dtype="float32")
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    faiss.normalize_L2(emb)
    return emb


def encode_images(images: Iterable[Image.Image]) -> np.ndarray:
    """编码一批 PIL 图片，返回可用于 FAISS 检索的向量矩阵。"""
    image_list = list(images)
    if not image_list:
        raise RuntimeError("encode_images 需要至少一张图片。")
    model = get_image_model()
    embeddings = model.encode(image_list)
    return _normalize_embeddings(embeddings)


def encode_image(image: Image.Image) -> np.ndarray:
    """编码单张图片，返回形状为 `(1, dim)` 的向量矩阵。"""
    return encode_images([image])


def encode_texts(texts: Iterable[str]) -> np.ndarray:
    """把文本编码到图像检索模型的共享语义空间中。"""
    text_list = list(texts)
    if not text_list:
        raise RuntimeError("encode_texts 输入不能为空。")
    model = get_image_model()
    embeddings = model.encode(text_list)
    return _normalize_embeddings(embeddings)


def encode_text(text: str) -> np.ndarray:
    """编码单条文本，返回形状为 `(1, dim)` 的向量矩阵。"""
    return encode_texts([text])
