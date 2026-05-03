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
你是一名专业的中文博物馆导览员。请只依据下方提供的展品资料回答观众问题。

要求：
1. 回答应自然、清楚，适合现场语音导览播报。
2. 不要编造资料中没有的信息。
3. 如果资料不足以判断答案，请明确说明“根据现有资料暂时无法确定”。
4. 最终回答只使用中文，不要夹杂英文。
5. 在保证信息完整的前提下，尽量控制在 150 个汉字以内。
""".strip(),
        contexts,
    )
    prompt += f"""

{LB}{QUESTION_LABEL}{RB}
{query}

请用中文回答：
"""
    return prompt


def build_multimodal_direct_prompt(question: str) -> str:
    query = _normalize_question(
        question,
        "请识别图片中的文物，并简要介绍它的名称、年代、类型和主要特征。",
    )
    return f"""
你是一名专业的中文博物馆导览员。你只能根据上传图片和观众问题作答。

要求：
1. 先判断图片中可见器物的类型、材质和外观特征。
2. 如果无法仅凭图片可靠判断具体文物名称或年代，请明确说明，不能猜测。
3. 不要编造图片中看不出、也没有依据的历史信息。
4. 最终回答只使用中文。
5. 先给出判断，再说明依据；在保证清楚的前提下，尽量控制在 120 个汉字以内。

{LB}{QUESTION_LABEL}{RB}
{query}

请用中文回答：
"""


_GUIDE_REQUEST_KEYWORDS = (
    "介绍",
    "讲解",
    "导览",
    "讲一讲",
    "说说",
    "讲述",
    "解读",
    "说明",
    "概括",
    "一段话",
    "一段介绍",
    "一段讲解",
)

_FACT_QUESTION_KEYWORDS = (
    "叫什么",
    "名称是什么",
    "属于什么",
    "是什么时代",
    "哪个朝代",
    "什么年代",
    "收藏于哪里",
    "馆藏单位",
    "博物馆",
    "什么材质",
    "主要材质",
    "功能用途",
    "有什么用途",
    "文化价值",
    "历史背景",
    "反映了什么",
    "体现了什么",
    "类别是什么",
    "类型是什么",
)


def _looks_like_guide_request(question: str) -> bool:
    query = (question or "").strip()
    return any(keyword in query for keyword in _GUIDE_REQUEST_KEYWORDS)


def _looks_like_fact_question(question: str) -> bool:
    query = (question or "").strip()
    if not query:
        return False
    if _looks_like_guide_request(query):
        return False
    return any(keyword in query for keyword in _FACT_QUESTION_KEYWORDS)


def _build_multimodal_fact_prompt(
    question: str,
    contexts: list[tuple[str, float]],
) -> str:
    query = _normalize_question(
        question,
        "请识别这件文物，并回答观众提出的具体问题。",
    )
    prompt = _append_contexts(
        """
你是一名专业的中文博物馆导览员。请综合上传图片和下方展品资料回答观众问题，其中事实信息应优先依据展品资料。

要求：
1. 先直接回答问题，再用一两句简洁说明补充依据。
2. 涉及名称、时代、用途、馆藏单位、历史背景、文化价值等事实信息时，只能使用资料中明确提供的信息。
3. 如果图片与资料不足以支持可靠判断，请明确说明“当前图片与资料不足以作出可靠判断”。
4. 不要把字段型问题讲成完整导览稿，不要偏离观众问题本身。
5. 最终回答只使用中文。
6. 在保证准确的前提下，尽量控制在 80 到 140 个汉字以内。
""".strip(),
        contexts,
    )
    prompt += f"""

{LB}{QUESTION_LABEL}{RB}
{query}

请用中文直接回答：
"""
    return prompt


def _build_multimodal_guide_prompt(
    question: str,
    contexts: list[tuple[str, float]],
) -> str:
    query = _normalize_question(
        question,
        "请为这件文物生成一段自然的中文导览讲解。",
    )
    prompt = _append_contexts(
        """
你是一名专业的中文博物馆导览员。请综合上传图片和下方展品资料，生成一段适合观众收听的导览讲解。

要求：
1. 回答要像真实导览员面对观众讲话，自然连贯，不要只罗列字段。
2. 先介绍这件文物是什么，再讲它的特点、用途、背景或文化价值。
3. 资料里没有明确写出的事实不要编造，也不要把相似文物混为一谈。
4. 可以适度组织语言，但事实信息必须以资料为依据。
5. 最终回答只使用中文，适合直接用于语音播报。
6. 回答应尽量完整，但避免空泛抒情。
7. 尽量控制在 220 到 320 个汉字之间。
""".strip(),
        contexts,
    )
    prompt += f"""

{LB}{QUESTION_LABEL}{RB}
{query}

请生成中文导览讲解：
"""
    return prompt


def build_multimodal_grounded_prompt(
    question: str,
    contexts: list[tuple[str, float]],
) -> str:
    if _looks_like_fact_question(question):
        return _build_multimodal_fact_prompt(question, contexts)
    return _build_multimodal_guide_prompt(question, contexts)


def build_multimodal_guide_prompt(
    question: str,
    contexts: list[tuple[str, float]],
) -> str:
    return _build_multimodal_guide_prompt(question, contexts)


def build_citation(contexts: list[tuple[str, float]]) -> str:
    sources: list[str] = []
    for text, _score in contexts:
        name, era = extract_meta(text)
        if name:
            sources.append(f"{name}{CN_LP}{era}{CN_RP}" if era else name)

    sources = list(dict.fromkeys(sources))
    return SOURCE_PREFIX + CN_COMMA.join(sources) if sources else ""
