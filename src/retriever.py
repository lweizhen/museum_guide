import faiss
import numpy as np
import re
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
    m = re.search(r"【展品名称】\s*([^\s【】]+)", doc)
    return m.group(1).strip() if m else ""


def _extract_aliases(doc: str) -> list[str]:
    """
    从文档里提取别名，支持：
    【别名】司母戊鼎
    【别名】司母戊鼎、司母戊
    """
    m = re.search(r"【别名】\s*([^\n【]+)", doc)
    if not m:
        return []

    raw = m.group(1).strip()
    if not raw:
        return []

    parts = re.split(r"[、，,;/；\s]+", raw)
    return [p.strip() for p in parts if p.strip()]


def _expand_query_with_doc_aliases(query: str, docs: list[str]) -> str:
    """将 query 中命中的知识库别名扩展为标准展品名，提高召回率。"""
    q = (query or "").strip()
    if not q:
        return ""

    matched_names: list[str] = []
    for doc in docs:
        name = _extract_name(doc)
        if not name:
            continue

        for alias in _extract_aliases(doc):
            if alias and alias in q and name not in q:
                matched_names.append(name)
                break

    if not matched_names:
        return q

    unique_names = list(dict.fromkeys(matched_names))
    return q + " " + " ".join(unique_names)


def retrieve(query: str, top_k: int = TOP_K, threshold: float = THRESHOLD, margin: float = MARGIN):
    index = get_index()
    docs = get_docs()

    q = (query or "").strip()
    if not q:
        return []

    q_expanded = _expand_query_with_doc_aliases(q, docs)

    for doc in docs:
        name = _extract_name(doc)
        aliases = _extract_aliases(doc)
        if name and (name in q or name in q_expanded):
            return [(doc, 1.0)]
        if any(alias in q for alias in aliases):
            return [(doc, 1.0)]

    q_emb = encode_texts([q_expanded])

    scores, indices = index.search(q_emb, top_k)

    pairs = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0:
            continue
        pairs.append((docs[int(idx)], float(score)))

    dyn_threshold = threshold
    if len(q) <= 8:
        dyn_threshold = max(threshold, 0.52)

    pairs = [(t, s) for (t, s) in pairs if s >= dyn_threshold]
    if not pairs:
        return []

    max_score = max(s for _, s in pairs)
    pairs = [(t, s) for (t, s) in pairs if s >= max_score - margin]

    return pairs
