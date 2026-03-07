import re

def clip(s: str, n: int = 450) -> str:
    return s if len(s) <= n else s[:n] + "…"

def extract_meta(block_text: str):
    name = None
    era = None
    m1 = re.search(r"【展品名称】\s*([^\s【】]+)", block_text)
    if m1:
        name = m1.group(1).strip()
    m2 = re.search(r"【所属时代】\s*([^\s【】]+)", block_text)
    if m2:
        era = m2.group(1).strip()
    return name, era

def build_prompt(query: str, contexts: list[tuple[str, float]]) -> str:
    prompt = f"""
你是一名专业的博物馆讲解员，请根据提供的展品资料回答观众的问题。

要求：
1. 语言自然、口语化，适合现场讲解。
2. 只能依据给定资料回答，不能编造不存在的信息。
3. 如果资料不足，请说明“根据现有资料无法确定”。
4. 控制在150字以内。
5. 用中文回答。

【展品资料】
"""
    for text, score in contexts:
        prompt += f"- {clip(text)}\n"

    prompt += f"""
【观众问题】
{query}

请开始讲解：
"""
    return prompt

def build_citation(contexts: list[tuple[str, float]]) -> str:
    sources = []
    for text, score in contexts:
        name, era = extract_meta(text)
        if name:
            sources.append(f"{name}（{era}）" if era else name)
    # 去重保序
    sources = list(dict.fromkeys(sources))
    return "资料来源：" + "、".join(sources) if sources else ""
