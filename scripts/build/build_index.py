import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import faiss
from src.kb import load_docs
from src.embedder import encode_texts
from src.config import INDEX_PATH

def main():
    docs = load_docs()
    emb = encode_texts(docs)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    print(f"OK，已构建向量库：{len(docs)} 条 -> {INDEX_PATH}")

if __name__ == "__main__":
    main()
