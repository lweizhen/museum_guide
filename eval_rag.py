# eval_rag.py
# -*- coding: utf-8 -*-

import os
import re
import json
import csv
import time
from typing import Any, Dict, List, Tuple, Optional

from src.retriever import retrieve
from src.llm import call_llm
from src.progress import iter_progress


# ------------------------------
# 配置区（按你当前目录结构写死）
# ------------------------------
EVAL_JSONL = os.path.join("data", "test_questions.jsonl")
OUT_DIR = os.path.join("outputs", "raw")
OUT_CSV = os.path.join(OUT_DIR, "eval_results.csv")
OUT_SUMMARY = os.path.join(OUT_DIR, "eval_summary.txt")

# 评测时每条问题最多拿多少条检索结果来做prompt（避免上下文太长）
MAX_CTX = 3


# ------------------------------
# 工具函数：读取 / 解析
# ------------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSONL 解析失败：{path} 第{ln}行。内容：{line[:120]}") from e
    return items


def extract_name_from_doc(doc: str) -> str:
    """
    从知识库文本块中抽取展品名称，兼容：
    【展品名称】青铜鼎
    【展品名称】 青铜鼎
    """
    if not doc:
        return ""
    m = re.search(r"【展品名称】\s*([^\n【]+)", doc)
    return m.group(1).strip() if m else ""


def join_names(names: List[str]) -> str:
    return "|".join([n for n in names if n])


def join_scores(scores: List[float]) -> str:
    return "|".join([f"{s:.4f}" for s in scores])


def is_refusal(ans: str) -> bool:
    ans = (ans or "").strip()
    if not ans:
        return True
    # 你系统设定的拒答语
    return "根据现有资料无法确定" in ans or "未在知识库中检索到足够相关的内容" in ans


# ------------------------------
# Prompt（评测版，尽量稳定）
# ------------------------------
def build_prompt(query: str, ctx_docs: List[str]) -> str:
    """
    用更“可评测”的模板，避免模型自由发挥太多
    """
    prompt = (
        "你是一名专业的博物馆讲解员，请严格依据【展品资料】回答【观众问题】。\n"
        "要求：\n"
        "1) 只依据给定资料回答，不要编造。\n"
        "2) 如果资料不足，请直接回答：根据现有资料无法确定。\n"
        "3) 用中文回答，150字以内。\n\n"
        "【展品资料】\n"
    )
    for d in ctx_docs:
        prompt += d.strip() + "\n"
    prompt += "\n【观众问题】\n" + query.strip() + "\n\n请开始讲解："
    return prompt


# ------------------------------
# 评测指标
# ------------------------------
def calc_hits(target: str, retrieved_docs: List[str]) -> Tuple[int, int, List[str]]:
    """
    target: 纯展品名，如 “贾湖骨笛”
    retrieved_docs: 检索返回的doc文本块列表
    return: (hit_top1, hit_topk, retrieved_names)
    """
    names = [extract_name_from_doc(d) for d in retrieved_docs]
    names = [n for n in names if n]

    if not target:
        return (0, 0, names)

    hit_top1 = int(len(names) > 0 and names[0] == target)
    hit_topk = int(target in names)
    return (hit_top1, hit_topk, names)


def mention_ok(target: str, answer: str) -> int:
    if not target:
        return 0
    ans = (answer or "")
    return int(target in ans)


def grounded_weak(target: str, answer: str, retrieved_docs: List[str]) -> int:
    """
    弱grounded：避免误杀讲解口吻扩写
    - 若拒答：算 grounded=1
    - 正例：回答里提到了 target 或者回答里出现了资料字段（所属时代/出土地/历史意义/作者）且上下文也有
    - 或者回答包含ctx里“历史意义”后的一段关键短语
    """
    ans = (answer or "").strip()
    if not ans:
        return 0
    if is_refusal(ans):
        return 1

    ctx = "\n".join(retrieved_docs)

    if target and target in ans:
        return 1

    # 字段锚点（宽松）
    for k in ["所属时代", "出土地", "历史意义", "作者"]:
        if (k in ans) and (k in ctx):
            return 1

    # 历史意义关键短语锚点（更稳一点）
    m = re.search(r"【历史意义】\s*([^\n]{6,30})", ctx)
    if m:
        key_phrase = m.group(1).strip()
        if key_phrase and key_phrase in ans:
            return 1

    return 0


# ------------------------------
# 主流程
# ------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(EVAL_JSONL):
        raise FileNotFoundError(f"找不到评测文件：{EVAL_JSONL}")

    items = read_jsonl(EVAL_JSONL)

    # 统计
    total = 0
    pos_cnt = 0
    neg_cnt = 0
    hit1_sum = 0
    hitk_sum = 0
    refuse_ok_sum = 0
    mention_sum = 0
    grounded_sum = 0

    # 写CSV
    fieldnames = [
        "id", "query", "target",
        "retrieved_top1", "retrieved_names", "retrieved_scores", "retrieved_cnt",
        "answer",
        "hit_top1", "hit_topk",
        "refuse_ok", "mention_ok", "grounded_ok",
    ]

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        for item in iter_progress(items, label="eval_rag"):
            total += 1
            qid = (item.get("id") or "").strip()
            query = (item.get("query") or "").strip()
            target = (item.get("target") or "").strip()  # 允许空：负例

            if target:
                pos_cnt += 1
            else:
                neg_cnt += 1

            # 1) 检索
            pairs = retrieve(query)  # List[(doc, score)]
            # 控制最大上下文条数
            pairs = pairs[:MAX_CTX]

            retrieved_docs = [d for d, s in pairs]
            retrieved_scores = [float(s) for d, s in pairs]

            # 2) 命中判定（关键：按“展品名称”对齐）
            hit_top1, hit_topk, retrieved_names = calc_hits(target, retrieved_docs)

            # 3) 生成
            if len(retrieved_docs) == 0:
                # 没检索到就直接拒答（这对负例评测很友好）
                answer = "未在知识库中检索到足够相关的内容，根据现有资料无法确定。"
            else:
                prompt = build_prompt(query, retrieved_docs)
                answer = call_llm(prompt)

            # 4) 指标判定
            if target:
                # 正例：评 mention/grounded（refuse_ok 不计入）
                m_ok = mention_ok(target, answer)
                g_ok = grounded_weak(target, answer, retrieved_docs)
                refuse_ok = 0
            else:
                # 负例：必须拒答才算对
                refuse_ok = int(is_refusal(answer))
                m_ok = 0
                g_ok = 0

            # 5) 累计统计
            if target:
                hit1_sum += hit_top1
                hitk_sum += hit_topk
                mention_sum += m_ok
                grounded_sum += g_ok
            else:
                refuse_ok_sum += refuse_ok

            # 6) 写行
            writer.writerow({
                "id": qid,
                "query": query,
                "target": target if target else "",
                "retrieved_top1": retrieved_docs[0] if retrieved_docs else "",
                "retrieved_names": join_names(retrieved_names),
                "retrieved_scores": join_scores(retrieved_scores),
                "retrieved_cnt": len(retrieved_docs),
                "answer": (answer or "").replace("\n", " ").strip(),
                "hit_top1": hit_top1,
                "hit_topk": hit_topk,
                "refuse_ok": refuse_ok,
                "mention_ok": m_ok,
                "grounded_ok": g_ok,
            })

    # 写 summary
    def safe_div(a: int, b: int) -> float:
        return float(a) / float(b) if b else 0.0

    top1_acc = safe_div(hit1_sum, pos_cnt)
    topk_acc = safe_div(hitk_sum, pos_cnt)
    neg_refuse_acc = safe_div(refuse_ok_sum, neg_cnt)
    mention_rate = safe_div(mention_sum, pos_cnt)
    grounded_rate = safe_div(grounded_sum, pos_cnt)

    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    summary = (
        f"Time: {now}\n"
        f"Total: {total}\n"
        f"Positive (target!=null): {pos_cnt}\n"
        f"Negative (target==null): {neg_cnt}\n\n"
        f"Retrieval Top1 Acc: {top1_acc:.3f}\n"
        f"Retrieval TopK  Acc: {topk_acc:.3f}\n"
        f"Negative Refuse Acc: {neg_refuse_acc:.3f}\n\n"
        f"Generation Mention Rate (pos): {mention_rate:.3f}\n"
        f"Generation Grounded Rate (weak): {grounded_rate:.3f}\n"
    )

    with open(OUT_SUMMARY, "w", encoding="utf-8") as fsum:
        fsum.write(summary)

    print(summary)
    print(f"[OK] 写入：{OUT_CSV}")
    print(f"[OK] 写入：{OUT_SUMMARY}")


if __name__ == "__main__":
    main()
