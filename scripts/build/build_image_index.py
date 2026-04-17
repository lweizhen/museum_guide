from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import os

import faiss

from src.config import IMAGE_INDEX_PATH, IMAGE_META_PATH
from src.image_embedder import encode_images
from src.image_index import (
    build_image_records,
    download_image,
    dump_image_meta,
    is_valid_local_image,
    load_image_rows,
    open_local_image,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="构建文物图片索引库")
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 张图片，0 表示全部")
    parser.add_argument("--skip-download", action="store_true", help="跳过图片下载，只使用本地缓存")
    args = parser.parse_args()

    rows = load_image_rows()
    records = build_image_records(rows)
    if args.limit > 0:
        records = records[: args.limit]

    valid_meta: list[dict[str, str]] = []
    valid_images = []

    for record in records:
        if not args.skip_download:
            try:
                download_image(record)
            except Exception:
                continue

        if not is_valid_local_image(record.local_path):
            continue

        image = open_local_image(record.local_path)
        valid_images.append(image)
        valid_meta.append(
            {
                "name": record.name,
                "era": record.era,
                "museum": record.museum,
                "category": record.category,
                "detail_url": record.detail_url,
                "source": record.source,
                "image_url": record.image_url,
                "local_path": record.local_path,
            }
        )

    if not valid_images:
        raise RuntimeError("没有可用于建索引的有效图片。")

    embeddings = encode_images(valid_images)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(os.path.dirname(IMAGE_INDEX_PATH), exist_ok=True)
    faiss.write_index(index, IMAGE_INDEX_PATH)
    dump_image_meta(valid_meta, IMAGE_META_PATH)

    print(f"OK，已构建图片索引：{len(valid_meta)} 张 -> {IMAGE_INDEX_PATH}")
    print(f"图片元数据已写入：{IMAGE_META_PATH}")


if __name__ == "__main__":
    main()
