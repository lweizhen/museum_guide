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


def build_multimodal_grounded_prompt(
    question: str,
    contexts: list[tuple[str, float]],
) -> str:
    query = _normalize_question(
        question,
        "请识别图片中的文物，并结合参考资料进行导览式讲解。",
    )
    prompt = _append_contexts(
        """
你是一名正在展厅中为观众讲解的中文博物馆导览员。请结合上传图片和参考资料，生成一段适合语音播报的导览讲解。

要求：
1. 先观察图片，再以参考资料作为名称、年代、用途、馆藏单位和历史背景等事实信息的主要依据。
2. 不要只罗列文物名称、年代、馆藏单位和用途，要把信息组织成自然连贯的讲解。
3. 如果图片与资料无法可靠对应，请明确说明“当前图片与资料不足以作出可靠判断”。
4. 可以用“大家现在看到的是……”等自然开场，但语气要稳重，不要过度夸张，也不要写成资料摘抄。
5. 先说明文物是什么，再介绍它的用途、造型或装饰特点，最后点出它反映的历史文化价值。
6. 使用完整段落，不要使用项目符号，不要输出英文。
7. 字数控制在 220 到 320 个汉字之间，适合现场导览播报。
""".strip(),
        contexts,
    )
    prompt += f"""

{LB}{QUESTION_LABEL}{RB}
{query}

请用中文生成导览讲解：
"""
    return prompt


def build_multimodal_guide_prompt(
    question: str,
    contexts: list[tuple[str, float]],
) -> str:
    return build_multimodal_grounded_prompt(question, contexts)



def build_citation(contexts: list[tuple[str, float]]) -> str:
    sources: list[str] = []
    for text, _score in contexts:
        name, era = extract_meta(text)
        if name:
            sources.append(f"{name}{CN_LP}{era}{CN_RP}" if era else name)

    sources = list(dict.fromkeys(sources))
    return SOURCE_PREFIX + CN_COMMA.join(sources) if sources else ""
