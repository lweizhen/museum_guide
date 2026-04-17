from __future__ import annotations

import re

LB = "\u3010"
RB = "\u3011"
NAME_LABEL = "\u5c55\u54c1\u540d\u79f0"
ERA_LABEL = "\u6240\u5c5e\u65f6\u4ee3"
DATA_LABEL = "\u5c55\u54c1\u8d44\u6599"
QUESTION_LABEL = "\u89c2\u4f17\u95ee\u9898"
SOURCE_PREFIX = "\u8d44\u6599\u6765\u6e90\uff1a"
CN_COMMA = "\u3001"
CN_LP = "\uff08"
CN_RP = "\uff09"


def clip(text: str, n: int = 450) -> str:
    return text if len(text) <= n else text[:n] + "..."


def extract_meta(block_text: str) -> tuple[str | None, str | None]:
    name = None
    era = None

    name_match = re.search(rf"{LB}{NAME_LABEL}{RB}\s*([^\s{LB}{RB}]+)", block_text)
    if name_match:
        name = name_match.group(1).strip()

    era_match = re.search(rf"{LB}{ERA_LABEL}{RB}\s*([^\s{LB}{RB}]+)", block_text)
    if era_match:
        era = era_match.group(1).strip()

    return name, era


def _normalize_question(question: str, default_question: str) -> str:
    question = (question or "").strip()
    return question if question else default_question


def _append_contexts(prefix: str, contexts: list[tuple[str, float]]) -> str:
    prompt = prefix + f"\n{LB}{DATA_LABEL}{RB}\n"
    for text, _score in contexts:
        prompt += f"- {clip(text)}\n"
    return prompt


def build_prompt(query: str, contexts: list[tuple[str, float]]) -> str:
    prompt = _append_contexts(
        """
You are a professional museum guide. Answer the visitor's question using only the provided exhibit records.

Requirements:
1. Answer naturally in Chinese, suitable for an on-site audio guide.
2. Do not invent facts that are not supported by the records.
3. If the records are insufficient, say in Chinese that the current records are insufficient to determine the answer.
4. Use Chinese only. Do not mix English into the final answer.
5. Keep the answer within 150 Chinese characters when possible.
""".strip(),
        contexts,
    )
    prompt += f"""

{LB}{QUESTION_LABEL}{RB}
{query}

Please answer in Chinese:
"""
    return prompt


def build_multimodal_direct_prompt(question: str) -> str:
    query = _normalize_question(
        question,
        "Please identify the artifact in the image and briefly introduce its name, period, type, and key features.",
    )
    return f"""
You are a professional museum guide. You may only use the uploaded image and the visitor's question.

Requirements:
1. First judge the visible object type, material, and visual features.
2. If the exact artifact name or period cannot be reliably identified from the image alone, explicitly say so in Chinese and do not guess.
3. Do not invent historical details that are not visible or supported.
4. Use Chinese only.
5. Give the judgement first, then the basis. Keep the answer within 120 Chinese characters when possible.

{LB}{QUESTION_LABEL}{RB}
{query}

Please answer in Chinese:
"""


def build_multimodal_grounded_prompt(
    question: str,
    contexts: list[tuple[str, float]],
) -> str:
    query = _normalize_question(
        question,
        "Please identify the artifact in the image and introduce it using the provided records.",
    )
    prompt = _append_contexts(
        """
You are a professional museum guide. You must answer by jointly considering the image and the provided exhibit records.

Requirements:
1. Inspect the image, then use the records as the primary source for factual details.
2. For names, periods, uses, and historical background, rely on explicit information in the records.
3. If the image and records cannot be reliably matched, say in Chinese that the current image and records are insufficient for a reliable judgement.
4. You may summarize, but do not treat similar artifacts as the same artifact.
5. Use Chinese only.
6. Keep the answer within 150 Chinese characters when possible.
""".strip(),
        contexts,
    )
    prompt += f"""

{LB}{QUESTION_LABEL}{RB}
{query}

Please answer in Chinese:
"""
    return prompt


def build_citation(contexts: list[tuple[str, float]]) -> str:
    sources: list[str] = []
    for text, _score in contexts:
        name, era = extract_meta(text)
        if name:
            sources.append(f"{name}{CN_LP}{era}{CN_RP}" if era else name)

    sources = list(dict.fromkeys(sources))
    return SOURCE_PREFIX + CN_COMMA.join(sources) if sources else ""
