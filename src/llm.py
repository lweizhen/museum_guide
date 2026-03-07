# src/llm.py
from __future__ import annotations

import json
import urllib.request
import urllib.error

import dashscope
from dashscope import Generation, MultiModalConversation

from .config import (
    # provider
    LLM_PROVIDER,
    # dashscope
    QWEN_MODEL,
    TEMPERATURE,
    get_api_key,
    # ollama
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
)


# ==================== DashScope ====================

def _is_multimodal_model(model_name: str) -> bool:
    """
    DashScope 部分模型（如 qwen3.5-plus）走 multimodal-generation 端点。
    这里做一个简单判断，避免出现 url error.
    """
    name = (model_name or "").lower()
    return ("qwen3.5" in name) or ("qwen-vl" in name) or ("vl" in name)


def _call_dashscope(prompt: str) -> str:
    dashscope.api_key = get_api_key()

    # 1) qwen3.5-plus 等：走 MultiModalConversation
    if _is_multimodal_model(QWEN_MODEL):
        messages = [
            {"role": "system", "content": [{"text": "你是一名专业的博物馆讲解员。"}]},
            {"role": "user", "content": [{"text": prompt}]},
        ]

        resp = MultiModalConversation.call(
            model=QWEN_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            result_format="message",
        )

        if resp.status_code != 200:
            raise RuntimeError(f"LLM调用失败：{resp.message}")

        # multimodal 返回格式：choices[0].message.content 是一个数组，取 text
        content = resp.output.choices[0].message.content
        if isinstance(content, list) and content:
            # 找到第一个包含 text 的块
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    return (block["text"] or "").strip()
        # 兜底
        return str(content).strip()

    # 2) 其他文本模型：走 Generation
    resp = Generation.call(
        model=QWEN_MODEL,
        prompt=prompt,
        temperature=TEMPERATURE,
    )

    if resp.status_code == 200:
        return (resp.output.text or "").strip()
    raise RuntimeError(f"LLM调用失败：{resp.message}")


# ==================== Ollama ====================

def _call_ollama(prompt: str) -> str:
    """
    调用本地 Ollama:
      POST http://localhost:11434/api/generate
    使用 stream=False 一次性拿完整文本。
    """
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
        },
    }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
            text = (obj.get("response") or "").strip()
            if not text:
                raise RuntimeError(f"Ollama 返回空响应：{raw[:200]}")
            return text
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Ollama 调用失败（{url}），请确认：\n"
            f"1) Ollama 已启动\n"
            f"2) 已 pull 模型：ollama pull {OLLAMA_MODEL}\n"
            f"3) 端口/地址正确\n"
            f"原始错误：{e}"
        ) from e


# ==================== 统一入口 ====================

def call_llm(prompt: str) -> str:
    """
    统一入口：根据 LLM_PROVIDER 选择 dashscope / ollama
    """
    provider = (LLM_PROVIDER or "").strip().lower()

    if provider == "dashscope":
        return _call_dashscope(prompt)

    if provider == "ollama":
        return _call_ollama(prompt)

    raise RuntimeError(
        f"未知 LLM_PROVIDER={LLM_PROVIDER}，可选：dashscope / ollama"
    )
