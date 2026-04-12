from __future__ import annotations

import re


def clip(text: str, n: int = 450) -> str:
    return text if len(text) <= n else text[:n] + "…"


def extract_meta(block_text: str) -> tuple[str | None, str | None]:
    name = None
    era = None

    name_match = re.search(r"【展品名称】\s*([^\s【】]+)", block_text)
    if name_match:
        name = name_match.group(1).strip()

    era_match = re.search(r"【所属时代】\s*([^\s【】]+)", block_text)
    if era_match:
        era = era_match.group(1).strip()

    return name, era


def _normalize_question(question: str, default_question: str) -> str:
    question = (question or "").strip()
    return question if question else default_question


def _append_contexts(prefix: str, contexts: list[tuple[str, float]]) -> str:
    prompt = prefix + "\n【展品资料】\n"
    for text, _score in contexts:
        prompt += f"- {clip(text)}\n"
    return prompt


def build_prompt(query: str, contexts: list[tuple[str, float]]) -> str:
    prompt = _append_contexts(
        """
你是一名专业的博物馆讲解员，请根据提供的展品资料回答观众的问题。

要求：
1. 语言自然、口语化，适合现场讲解。
2. 只能依据给定资料回答，不能编造不存在的信息。
3. 如果资料不足，请明确说“根据现有资料无法确定”。
4. 全程只用中文，不要夹杂英文或中英混杂表达。
5. 控制在150字以内。
""".strip(),
        contexts,
    )
    prompt += f"""

【观众问题】
{query}

请开始讲解：
"""
    return prompt


def build_multimodal_direct_prompt(question: str) -> str:
    query = _normalize_question(
        question,
        "请识别图片中的文物，并简要介绍它的名称、时代、类型和主要特点。",
    )
    return f"""
你是一名专业的博物馆讲解员。现在你只能根据观众上传的图片和问题作答。

要求：
1. 先判断图片里能直接观察到的器物类型、材质和外观特征。
2. 如果无法仅凭图片可靠判断具体文物名称或具体时代，请明确说“仅凭图片无法可靠判断具体名称或时代”，不要猜测。
3. 不要编造图片中看不到、也没有依据的历史细节。
4. 全程只用中文，不要夹杂英文。
5. 先说判断，再说依据，控制在120字以内。

【观众问题】
{query}

请开始回答：
"""


def build_multimodal_grounded_prompt(
    question: str,
    contexts: list[tuple[str, float]],
) -> str:
    query = _normalize_question(
        question,
        "请识别图片中的文物，并结合资料介绍它的名称、时代、类型和主要特点。",
    )
    prompt = _append_contexts(
        """
你是一名专业的博物馆讲解员。现在你需要同时参考图片和提供的展品资料作答。

要求：
1. 先看图片，再结合资料回答。
2. 涉及名称、时代、用途、历史背景等事实信息时，优先依据给定资料中已经明确写出的内容，不得编造。
3. 如果图片内容与资料无法可靠对应，或者你把握不足，请明确说“根据现有图片和资料无法可靠判断”。
4. 允许概括，但不能把相似器物当成同一件文物。
5. 全程只用中文，不要夹杂英文。
6. 控制在150字以内。
""".strip(),
        contexts,
    )
    prompt += f"""

【观众问题】
{query}

请开始讲解：
"""
    return prompt


def build_citation(contexts: list[tuple[str, float]]) -> str:
    sources: list[str] = []
    for text, _score in contexts:
        name, era = extract_meta(text)
        if name:
            sources.append(f"{name}（{era}）" if era else name)

    sources = list(dict.fromkeys(sources))
    return "资料来源：" + "、".join(sources) if sources else ""
