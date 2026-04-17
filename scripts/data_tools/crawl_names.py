from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.museumschina.cn"
START_URL = "https://www.museumschina.cn/collection"
OUTPUT_CSV = "bowuguozhongguo_names_filtered.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

SLEEP_SECONDS = 1.0
MAX_PAGES = 50
DETAIL_SLEEP_SECONDS = 0.5
OUTPUT_FIELDS = [
    "name",
    "category",
    "era",
    "museum",
    "level",
    "accession_year",
    "material",
    "image_urls",
    "detail_url",
    "source",
]
INVALID_NAMES = {"VR", "藏品", "文物"}

DYNASTY_KEYWORDS = [
    "旧石器",
    "新石器",
    "夏",
    "商",
    "西周",
    "东周",
    "春秋",
    "战国",
    "秦",
    "汉",
    "三国",
    "西晋",
    "东晋",
    "南北朝",
    "隋",
    "唐",
    "五代",
    "北宋",
    "南宋",
    "宋",
    "辽",
    "金",
    "元",
    "明",
    "清",
]

RELIC_KEYWORDS = [
    "铜",
    "玉",
    "瓷",
    "陶",
    "鼎",
    "壶",
    "尊",
    "罐",
    "盘",
    "镜",
    "印",
    "玺",
    "佛",
    "碑",
    "俑",
    "钱",
    "像",
    "轴",
    "扇面",
    "簋",
    "钺",
    "盉",
    "匜",
    "觚",
    "洗",
    "炉",
    "瓶",
    "盒",
    "砚",
    "拓片",
]

MODERN_EXCLUDE_KEYWORDS = [
    "20世纪",
    "近现代",
    "现代",
    "当代",
    "红军",
    "毛泽东",
    "导弹",
    "地雷",
    "火箭",
    "手榴弹",
    "粮票",
    "像章",
    "著作选读",
    "出版",
    "乙炔灯",
    "宣传画",
    "公债",
    "公民证",
    "委任令",
    "契纸",
    "地图",
    "工资标准",
    "放行证明书",
    "存款折",
    "会旗照片",
    "唱片",
    "推荐信",
    "皮包",
    "浴血齐鲁",
]

YEAR_PATTERN = re.compile(r"(18|19|20)\d{2}年")


def fetch(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    return response.text


def dedupe_items(items: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for item in items:
        name = (item.get("name") or "").strip()
        detail_url = (item.get("detail_url") or "").strip()
        key = (name, detail_url)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def strip_era_prefix(name: str, era: str) -> str:
    normalized_name = (name or "").strip()
    normalized_era = (era or "").strip()
    if not normalized_name or not normalized_era:
        return normalized_name

    candidates = [normalized_era]
    for suffix in ("时代", "时期"):
        if normalized_era.endswith(suffix):
            candidates.append(normalized_era[: -len(suffix)])
    if normalized_era.endswith("文化"):
        candidates.append(normalized_era[: -len("文化")])

    for candidate in sorted(set(filter(None, candidates)), key=len, reverse=True):
        if normalized_name.startswith(candidate):
            stripped = normalized_name[len(candidate) :].lstrip()
            if stripped:
                return stripped
    return normalized_name


def extract_list_block_fields(text_lines: list[str]) -> tuple[str, str]:
    category = ""
    era = ""
    for line in text_lines:
        if "类别" in line:
            category = line.replace("类别", "").replace("：", "").replace(":", "").strip()
        elif "年代" in line:
            era = line.replace("年代", "").replace("：", "").replace(":", "").strip()
    return category, era


def parse_collection_page(html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    items: list[dict[str, str]] = []

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        name = anchor.get_text(strip=True)
        if not name or not href.startswith("/collection/"):
            continue

        parent = anchor.parent
        block_text = parent.get_text("\n", strip=True) if parent else ""
        lines = [line.strip() for line in block_text.split("\n") if line.strip()]
        category, era = extract_list_block_fields(lines)

        museum = ""
        search_root = parent.parent if parent and parent.parent else parent
        if search_root:
            for museum_anchor in search_root.find_all("a", href=True):
                if "/museums/details" in museum_anchor["href"].lower():
                    museum = museum_anchor.get_text(strip=True)
                    break

        items.append(
            {
                "name": name,
                "category": category,
                "era": era,
                "museum": museum,
                "level": "",
                "accession_year": "",
                "material": "",
                "image_urls": "",
                "detail_url": urljoin(BASE_URL, href),
                "source": "博物中国",
            }
        )

    return dedupe_items(items)


def find_next_page(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    for anchor in soup.find_all("a", href=True):
        text = anchor.get_text(strip=True)
        if text in {"下一页", ">", "后页", "末页"}:
            return urljoin(BASE_URL, anchor["href"])
    return None


def crawl_collection(max_pages: int = MAX_PAGES, sleep_seconds: float = SLEEP_SECONDS) -> list[dict[str, str]]:
    url = START_URL
    all_items: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for page_num in range(1, max_pages + 1):
        if not url or url in seen_urls:
            break

        print(f"[{page_num}] 正在抓取列表页: {url}")
        seen_urls.add(url)

        try:
            html = fetch(url)
        except Exception as exc:
            print(f"列表页抓取失败: {exc}")
            break

        page_items = parse_collection_page(html)
        print(f"  本页提取到 {len(page_items)} 条候选")
        all_items.extend(page_items)

        url = find_next_page(html)
        time.sleep(sleep_seconds)

    return dedupe_items(all_items)


def parse_detail_page(html: str, seed_item: dict[str, str]) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    info_block = soup.select_one("div.details_info")
    if info_block is None:
        return seed_item

    row = {field: (seed_item.get(field) or "").strip() for field in OUTPUT_FIELDS}
    title = info_block.select_one(".info_tit")
    if title and title.get_text(strip=True):
        row["name"] = title.get_text(strip=True)

    for paragraph in info_block.select("p"):
        label = paragraph.select_one("span")
        label_text = label.get_text(strip=True) if label else ""
        value_text = paragraph.get_text(" ", strip=True)
        if label_text:
            value_text = value_text.replace(label_text, "", 1).strip()

        if label_text == "收藏单位" and value_text:
            row["museum"] = value_text
        elif label_text == "类别" and value_text:
            row["category"] = value_text
        elif label_text == "年代" and value_text:
            row["era"] = value_text
        elif label_text == "级别" and value_text:
            row["level"] = value_text
        elif label_text == "入藏年度" and value_text:
            row["accession_year"] = value_text
        elif label_text == "质地" and value_text:
            row["material"] = value_text

    image_urls: list[str] = []
    for image in soup.select("#pic img"):
        src = (image.get("src") or "").strip()
        if not src:
            continue
        full_url = urljoin(BASE_URL, src)
        if full_url not in image_urls:
            image_urls.append(full_url)
    row["image_urls"] = "|".join(image_urls)
    return row


def enrich_items_from_details(
    items: Iterable[dict[str, str]],
    sleep_seconds: float = DETAIL_SLEEP_SECONDS,
) -> list[dict[str, str]]:
    enriched: list[dict[str, str]] = []
    for idx, item in enumerate(items, start=1):
        detail_url = (item.get("detail_url") or "").strip()
        if not detail_url:
            continue
        print(f"[detail {idx}] 正在抓取详情页: {detail_url}")
        try:
            html = fetch(detail_url)
            enriched.append(parse_detail_page(html, item))
        except Exception as exc:
            print(f"  详情页抓取失败，保留种子字段: {exc}")
            enriched.append(item)
        time.sleep(sleep_seconds)
    return dedupe_items(enriched)


def normalize_item(item: dict[str, str]) -> dict[str, str] | None:
    normalized = {field: (item.get(field) or "").strip() for field in OUTPUT_FIELDS}
    if not normalized["name"] or not normalized["detail_url"]:
        return None
    if normalized["name"] in INVALID_NAMES:
        return None
    normalized["name"] = strip_era_prefix(normalized["name"], normalized["era"])
    return normalized


def has_ancient_signal(item: dict[str, str]) -> bool:
    combined = " ".join(
        [
            item["name"],
            item["category"],
            item["era"],
            item["museum"],
            item["material"],
        ]
    )
    return any(keyword in combined for keyword in DYNASTY_KEYWORDS) or any(
        keyword in item["name"] for keyword in RELIC_KEYWORDS
    )


def has_modern_signal(item: dict[str, str]) -> bool:
    combined = " ".join(
        [
            item["name"],
            item["category"],
            item["era"],
            item["museum"],
            item["material"],
        ]
    )
    return any(keyword in combined for keyword in MODERN_EXCLUDE_KEYWORDS) or bool(
        YEAR_PATTERN.search(combined)
    )


def filter_for_project(items: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    for item in items:
        normalized = normalize_item(item)
        if normalized is None:
            continue
        if has_modern_signal(normalized):
            continue
        if not has_ancient_signal(normalized):
            continue
        filtered.append(normalized)
    return dedupe_items(filtered)


def write_csv(path: str | Path, rows: Iterable[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_seed_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    seeds: list[dict[str, str]] = []
    for row in rows:
        seed = {field: (row.get(field) or "").strip() for field in OUTPUT_FIELDS}
        if seed["detail_url"]:
            seeds.append(seed)
    return dedupe_items(seeds)


def run_pipeline(
    input_csv: str | None,
    max_pages: int = MAX_PAGES,
    list_sleep_seconds: float = SLEEP_SECONDS,
    detail_sleep_seconds: float = DETAIL_SLEEP_SECONDS,
) -> None:
    if input_csv:
        seed_items = read_seed_csv(input_csv)
        print(f"从现有 CSV 载入 {len(seed_items)} 条种子记录")
    else:
        seed_items = crawl_collection(max_pages=max_pages, sleep_seconds=list_sleep_seconds)
        print(f"列表页阶段获得 {len(seed_items)} 条候选记录")

    enriched_items = enrich_items_from_details(seed_items, sleep_seconds=detail_sleep_seconds)
    filtered_items = filter_for_project(enriched_items)
    write_csv(OUTPUT_CSV, filtered_items)
    print(f"详情补全后共有 {len(enriched_items)} 条记录")
    print(f"最终保留 {len(filtered_items)} 条 -> {OUTPUT_CSV}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="抓取博物中国藏品并生成适合导览项目使用的文物候选集。")
    parser.add_argument("--input-csv", help="从现有 CSV 读取 detail_url 作为种子，不重新抓列表页")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES, help="最多抓取列表页页数")
    parser.add_argument("--sleep", type=float, default=SLEEP_SECONDS, help="列表页抓取间隔秒数")
    parser.add_argument(
        "--detail-sleep",
        type=float,
        default=DETAIL_SLEEP_SECONDS,
        help="详情页抓取间隔秒数",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(
        input_csv=args.input_csv,
        max_pages=args.max_pages,
        list_sleep_seconds=args.sleep,
        detail_sleep_seconds=args.detail_sleep,
    )


if __name__ == "__main__":
    main()
