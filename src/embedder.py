import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .config import EMBED_MODEL_NAME

_embed_model = None

def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
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
