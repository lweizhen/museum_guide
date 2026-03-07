from .config import DATA_PATH

def load_docs() -> list[str]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    docs = [blk.strip().replace("\n", " ") for blk in raw.split("\n\n") if blk.strip()]

    if not docs:
        raise RuntimeError(
            "知识库为空：请检查 data/exhibits.txt 是否有内容，且展品之间要用空行分隔（\\n\\n）。"
        )

    return docs
