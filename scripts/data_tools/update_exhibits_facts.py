#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path


def _replace_once(line: str, old: str, new: str) -> tuple[str, bool]:
    if old not in line:
        return line, False
    return line.replace(old, new, 1), True


def apply_updates(lines: list[str]) -> tuple[list[str], list[str], list[str]]:
    changed: list[str] = []
    missing: list[str] = []

    new_lines = list(lines)

    def replace_line_startswith(tag: str, prefix: str, new_line: str) -> None:
        for i, ln in enumerate(new_lines):
            if ln.startswith(prefix):
                new_lines[i] = new_line
                changed.append(tag)
                return
        missing.append(tag)

    def replace_in_line(tag: str, marker: str, old: str, new: str) -> None:
        for i, ln in enumerate(new_lines):
            if marker in ln:
                updated, ok = _replace_once(ln, old, new)
                if ok:
                    new_lines[i] = updated
                    changed.append(tag)
                else:
                    missing.append(tag)
                return
        missing.append(tag)

    replace_line_startswith(
        tag="四羊方尊-发现叙述修正",
        prefix="【故事传说】四羊方尊的发现堪称机缘巧合。1938年，湖南宁乡农民黄材",
        new_line=(
            "【故事传说】四羊方尊于1938年出土于湖南宁乡黄材一带。"
            "这件国宝在战乱年代辗转保存并曾受损，后由文物工作者修复。"
            "器物四角圆雕卷角羊首，肩颈部纹饰繁复，体现了商代青铜铸造与分铸工艺的高水平。"
            "它既是礼器艺术的代表，也见证了中国文物保护工作的艰难历程。"
        ),
    )

    replace_in_line(
        tag="后母戊鼎-耳部表述修正",
        marker="后母戊鼎的发现充满了传奇色彩",
        old="抗战胜利后，大鼎被重新挖出，但已失去了原有的耳部。后来考古学家根据残痕成功修复。",
        new="抗战胜利后，大鼎被重新挖出，出土时有鼎耳受损，后经文物工作者修复。",
    )

    replace_in_line(
        tag="勾践剑-防腐机理修正",
        marker="越王勾践剑承载着\"卧薪尝胆\"的传奇故事",
        old="专家研究发现，剑身经过硫化处理，这是它千年不锈的秘密。",
        new="专家研究认为，剑身保存与合金成分、表面状态及埋藏环境等因素有关，具体机理仍有讨论。",
    )

    replace_in_line(
        tag="曾侯乙编钟-第八奇迹表述修正",
        marker="湖北随州擂鼓墩发现了一座战国早期的大型墓葬——曾侯乙墓",
        old="曾侯乙编钟的出土被誉为\"世界第八大奇迹\"，它不仅是中国古代音乐文化的瑰宝，更是中华文明辉煌成就的见证。",
        new="曾侯乙编钟的出土被视为先秦礼乐文明研究的里程碑发现，它不仅是中国古代音乐文化的瑰宝，更是中华文明辉煌成就的见证。",
    )

    replace_in_line(
        tag="长信宫灯-清水吸收表述修正",
        marker="长信宫灯的设计体现了汉代工匠的非凡智慧",
        old="最巧妙的是，灯燃烧产生的烟尘会通过宫女的袖管进入体内，被体内清水吸收，避免污染室内空气，堪称古代的\"环保灯具\"。",
        new="最巧妙的是，灯燃烧产生的烟尘会通过宫女的袖管导入灯体内部，从而减少烟尘外逸，堪称古代设计智慧的代表。",
    )

    replace_in_line(
        tag="千里江山图-早逝定论降级",
        marker="千里江山图的作者王希孟是中国绘画史上最传奇的天才少年",
        old="画作完成后不久，王希孟便英年早逝，年仅20余岁，这幅画成为他留下的唯一传世之作。",
        new="关于王希孟后续生平，后世常见英年早逝的说法，但史料并不充分；这幅画通常被视为其最重要的传世作品。",
    )

    replace_in_line(
        tag="贾湖骨笛-现代乐曲措辞降级",
        marker="8000年前的贾湖先民创造了令现代人惊叹的音乐奇迹",
        old="考古学家发现，贾湖骨笛的音准度极高，甚至可以演奏现代乐曲。",
        new="考古研究表明，部分贾湖骨笛具备较高音准，可用于复原演示早期音阶。",
    )

    return new_lines, changed, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="批量修正文物知识库中的高风险史实表述")
    parser.add_argument(
        "--file",
        default="data/exhibits.txt",
        help="目标知识库文件路径（默认：data/exhibits.txt）",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="实际写回文件（默认仅预览）",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")

    original = path.read_text(encoding="utf-8")
    lines = original.splitlines()
    new_lines, changed, missing = apply_updates(lines)

    print("=== 文物知识库修正脚本 ===")
    print(f"目标文件: {path}")
    print(f"命中并更新: {len(changed)} 条")
    if changed:
        for item in changed:
            print(f"  - {item}")

    if missing:
        print(f"未命中规则: {len(missing)} 条")
        for item in missing:
            print(f"  - {item}")

    if not args.apply:
        print("\n当前为预览模式，未写回文件。")
        print("如需应用修改，请执行: python update_exhibits_facts.py --apply")
        return

    updated = "\n".join(new_lines) + ("\n" if original.endswith("\n") else "")
    if updated == original:
        print("\n没有实际变更，文件保持不变。")
        return

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(updated, encoding="utf-8")
    tmp_path.replace(path)
    print("\n修改已写回文件。")


if __name__ == "__main__":
    main()
