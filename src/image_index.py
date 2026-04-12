from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image, UnidentifiedImageError

from .config import IMAGE_CACHE_DIR, IMAGE_CSV_PATH, IMAGE_MAX_IMAGES_PER_ITEM


@dataclass
class ImageRecord:
    name: str
    era: str
    museum: str
    category: str
    detail_url: str
    source: str
    image_url: str
    local_path: str


def load_image_rows(csv_path: str = IMAGE_CSV_PATH) -> list[dict[str, str]]:
    path = Path(csv_path)
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def parse_image_urls(raw: str, max_images_per_item: int = IMAGE_MAX_IMAGES_PER_ITEM) -> list[str]:
    urls = [item.strip() for item in (raw or "").split("|") if item.strip()]
    return urls[:max_images_per_item]


def build_image_records(
    rows: list[dict[str, str]],
    cache_dir: str = IMAGE_CACHE_DIR,
    max_images_per_item: int = IMAGE_MAX_IMAGES_PER_ITEM,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    for row in rows:
        urls = parse_image_urls(row.get("image_urls", ""), max_images_per_item=max_images_per_item)
        for image_url in urls:
            suffix = Path(image_url.split("?")[0]).suffix or ".jpg"
            digest = hashlib.sha1(image_url.encode("utf-8")).hexdigest()
            local_path = cache_root / f"{digest}{suffix}"
            records.append(
                ImageRecord(
                    name=(row.get("name") or "").strip(),
                    era=(row.get("era") or "").strip(),
                    museum=(row.get("museum") or "").strip(),
                    category=(row.get("category") or "").strip(),
                    detail_url=(row.get("detail_url") or "").strip(),
                    source=(row.get("source") or "").strip(),
                    image_url=image_url,
                    local_path=str(local_path),
                )
            )
    return records


def download_image(record: ImageRecord, timeout: int = 30) -> bool:
    local_path = Path(record.local_path)
    if local_path.exists():
        return True

    response = requests.get(record.image_url, timeout=timeout)
    response.raise_for_status()
    local_path.write_bytes(response.content)
    return True


def open_local_image(path: str) -> Image.Image:
    image = Image.open(path)
    return image.convert("RGB")


def is_valid_local_image(path: str) -> bool:
    try:
        with open_local_image(path):
            return True
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return False


def dump_image_meta(records: list[dict[str, str]], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
