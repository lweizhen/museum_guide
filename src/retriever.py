"""文本知识库检索模块。

本模块是文本 RAG 的核心：加载 FAISS 文本索引和知识库文档，
把用户问题编码为向量，并返回相似的文物知识块。
"""

import faiss
import numpy as np
import re
from .config import INDEX_PATH, TOP_K, THRESHOLD, MARGIN
from .embedder import get_embed_model, encode_texts
from .kb import load_docs

_index = None
_docs = None


def get_index():
    """懒加载文本 FAISS 索引。"""
    global _index
    if _index is None:
        _index = faiss.read_index(INDEX_PATH)
    return _index


def get_docs():
    """懒加载统一知识库文档块。"""
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
    """根据用户问题检索相关文物知识块。

    检索策略包含三层：
    1. 空问题直接返回空列表。
    2. 如果问题中直接出现文物名称或别名，优先返回对应知识块。
    3. 否则使用向量检索，并根据阈值和分数差距保留可信候选。
    """
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
