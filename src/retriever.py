import faiss
import numpy as np
from .config import INDEX_PATH, TOP_K, THRESHOLD, MARGIN
from .embedder import get_embed_model, encode_texts
from .kb import load_docs

_index = None
_docs = None


def get_index():
    global _index
    if _index is None:
        _index = faiss.read_index(INDEX_PATH)
    return _index


def get_docs():
    global _docs
    if _docs is None:
        _docs = load_docs()
    return _docs


def _extract_name(doc: str) -> str:
    """
    从文档里提取展品名，兼容两种格式：
    1) 【展品名称】 青铜鼎
    2) 【展品名称】青铜鼎
    """
    key = "【展品名称】"
    if key not in doc:
        return ""
    tail = doc.split(key, 1)[1].strip()
    return tail.splitlines()[0].strip()


def retrieve(query: str, top_k: int = TOP_K, threshold: float = THRESHOLD, margin: float = MARGIN):
    index = get_index()
    docs = get_docs()

    q = (query or "").strip()
    if not q:
        return []

    # 0) 规则优先：query 里包含展品名 -> 直接命中，避免被阈值误杀
    for doc in docs:
        name = _extract_name(doc)
        if name and name in q:
            return [(doc, 1.0)]  # 强制高分，表示强匹配

    # 1) 向量检索
    q_emb = encode_texts([q])

    scores, indices = index.search(q_emb, top_k)

    pairs = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0:
            continue
        pairs.append((docs[int(idx)], float(score)))

    # 2) 动态阈值：短 query 放宽一些（例如“兵马俑有什么用”）
    dyn_threshold = threshold
    if len(q) <= 8:
        dyn_threshold = min(threshold, 0.55)

    # 3) 绝对阈值过滤
    pairs = [(t, s) for (t, s) in pairs if s >= dyn_threshold]
    if not pairs:
        return []

    # 4) 相对阈值（动态Top-k）：只保留接近最优结果的一组
    max_score = max(s for _, s in pairs)
    pairs = [(t, s) for (t, s) in pairs if s >= max_score - margin]

    return pairs