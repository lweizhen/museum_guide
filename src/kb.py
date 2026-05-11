"""知识库读取模块。

系统的文本 RAG 和多模态 RAG 都依赖统一知识库文件。
本模块负责把 `data/exhibits_combined.txt` 按文物块读取成字符串列表。
"""

from .config import DATA_PATH


def load_docs() -> list[str]:
    """读取统一文物知识库，并按空行切分为多个文物知识块。

    每个知识块通常对应一件文物，包含名称、时代、馆藏单位、用途、
    历史背景、文化价值等字段。检索模块会把这些知识块作为召回单位。
    """
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    docs = [blk.strip().replace("\n", " ") for blk in raw.split("\n\n") if blk.strip()]

    if not docs:
        raise RuntimeError(
            "知识库为空：请检查 data/exhibits_combined.txt 是否有内容，"
            "并确认不同文物之间使用空行分隔。"
        )

    return docs
