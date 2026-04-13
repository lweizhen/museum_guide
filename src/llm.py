# src/llm.py
from __future__ import annotations

import base64
import json
import socket
import tempfile
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

import dashscope
from dashscope import Generation, MultiModalConversation
from PIL import Image

from .config import (
    JUDGE_API_KEY,
    JUDGE_BASE_URL,
    JUDGE_MODEL,
    JUDGE_PROVIDER,
    JUDGE_TEMPERATURE,
    JUDGE_TIMEOUT_SECONDS,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT_SECONDS,
    OUTPUT_DIR,
    QWEN_MODEL,
    TEMPERATURE,
    get_api_key,
)


def _is_multimodal_model(model_name: str) -> bool:
    name = (model_name or "").lower()
    return ("qwen3.5" in name) or ("qwen-vl" in name) or ("vl" in name)


def _is_ollama_multimodal_model(model_name: str) -> bool:
    name = (model_name or "").lower()
    keywords = [
        "llava",
        "vl",
        "vision",
        "minicpm-v",
        "qwen2.5vl",
        "qwen2-vl",
        "moondream",
        "bakllava",
    ]
    return any(keyword in name for keyword in keywords)


def _extract_multimodal_text(content: object) -> str:
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and "text" in block:
                return str(block.get("text", "")).strip()
    return str(content).strip()


def _call_dashscope_messages(messages: list[dict]) -> str:
    dashscope.api_key = get_api_key()
    resp = MultiModalConversation.call(
        model=QWEN_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        result_format="message",
    )

    if resp.status_code != 200:
        raise RuntimeError(f"LLM调用失败：{resp.message}")

    content = resp.output.choices[0].message.content
    return _extract_multimodal_text(content)


def _call_dashscope(prompt: str) -> str:
    if _is_multimodal_model(QWEN_MODEL):
        return _call_dashscope_messages(
            [
                {"role": "system", "content": [{"text": "你是一名专业的博物馆讲解员。"}]},
                {"role": "user", "content": [{"text": prompt}]},
            ]
        )

    dashscope.api_key = get_api_key()
    resp = Generation.call(
        model=QWEN_MODEL,
        prompt=prompt,
        temperature=TEMPERATURE,
    )

    if resp.status_code == 200:
        return (resp.output.text or "").strip()
    raise RuntimeError(f"LLM调用失败：{resp.message}")


def _post_ollama(payload: dict, timeout_seconds: int = OLLAMA_TIMEOUT_SECONDS) -> str:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
            text = (obj.get("response") or "").strip()
            if not text:
                raise RuntimeError(f"Ollama 返回空响应：{raw[:200]}")
            return text
    except (TimeoutError, socket.timeout) as e:
        raise RuntimeError(
            f"Ollama 请求超时（超过 {OLLAMA_TIMEOUT_SECONDS} 秒）。\n"
            "当前本地多模态模型可能正在 CPU 上运行，图片问答会明显变慢。\n"
            "建议检查：\n"
            "1) 运行 `ollama ps` 查看 PROCESSOR 是否为 GPU\n"
            "2) 如继续使用当前模型，可调大 `ollama.timeout_seconds`\n"
            "3) 如长期只能走 CPU，建议改用更小的多模态模型，或租用 GPU 做方案B实验"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Ollama 调用失败（{url}），请确认：\n"
            f"1) Ollama 已启动\n"
            f"2) 已 pull 模型：ollama pull {OLLAMA_MODEL}\n"
            f"3) 端口和地址配置正确\n"
            f"原始错误：{e}"
        ) from e


def _call_ollama_model(
    prompt: str,
    model: str,
    temperature: float,
    timeout_seconds: int = OLLAMA_TIMEOUT_SECONDS,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }
    return _post_ollama(payload, timeout_seconds=timeout_seconds)


def _call_ollama(prompt: str) -> str:
    return _call_ollama_model(prompt, OLLAMA_MODEL, OLLAMA_TEMPERATURE)


def _call_dashscope_text_model(prompt: str, model: str, temperature: float) -> str:
    dashscope.api_key = get_api_key()
    resp = Generation.call(
        model=model,
        prompt=prompt,
        temperature=temperature,
    )

    if resp.status_code == 200:
        return (resp.output.text or "").strip()
    raise RuntimeError(f"DashScope judge call failed: {resp.message}")


def _call_openai_compatible(prompt: str, model: str) -> str:
    if not JUDGE_API_KEY:
        raise RuntimeError(
            "Judge provider is openai, but no judge.api_key or JUDGE_API_KEY is configured."
        )
    if not model:
        raise RuntimeError(
            "Judge provider is openai, but no judge.model or JUDGE_MODEL is configured."
        )

    url = f"{JUDGE_BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict museum QA evaluator. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": JUDGE_TEMPERATURE,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JUDGE_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=JUDGE_TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
    except (TimeoutError, socket.timeout) as e:
        raise RuntimeError(
            f"Judge OpenAI-compatible request timed out after {JUDGE_TIMEOUT_SECONDS}s."
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Judge OpenAI-compatible request failed: {e}") from e

    try:
        return obj["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected judge response: {json.dumps(obj)[:300]}") from e


def _encode_image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _save_temp_image(image: Image.Image) -> Path:
    temp_dir = Path(OUTPUT_DIR) / "multimodal_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".png",
        dir=temp_dir,
        delete=False,
    ) as file:
        image.save(file, format="PNG")
        return Path(file.name)


def _call_dashscope_multimodal(prompt: str, image: Image.Image) -> str:
    image_path = _save_temp_image(image)
    try:
        return _call_dashscope_messages(
            [
                {"role": "system", "content": [{"text": "你是一名专业的博物馆讲解员。"}]},
                {
                    "role": "user",
                    "content": [
                        {"image": str(image_path)},
                        {"text": prompt},
                    ],
                },
            ]
        )
    finally:
        image_path.unlink(missing_ok=True)


def _call_ollama_multimodal(prompt: str, image: Image.Image) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "images": [_encode_image_to_base64(image)],
        "stream": False,
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
        },
    }
    return _post_ollama(payload)


def call_llm(prompt: str) -> str:
    provider = (LLM_PROVIDER or "").strip().lower()

    if provider == "dashscope":
        return _call_dashscope(prompt)

    if provider == "ollama":
        return _call_ollama(prompt)

    raise RuntimeError(
        f"未知 LLM_PROVIDER={LLM_PROVIDER}，可选：dashscope / ollama"
    )


def call_judge_llm(prompt: str) -> str:
    """Call the configured semantic judge model.

    The judge path is intentionally separate from the generation path so
    evaluations can use a stronger or different model than the model under test.
    """
    provider = (JUDGE_PROVIDER or "same").strip().lower()

    if provider in {"", "same"}:
        return call_llm(prompt)

    if provider == "dashscope":
        return _call_dashscope_text_model(
            prompt,
            JUDGE_MODEL or QWEN_MODEL,
            JUDGE_TEMPERATURE,
        )

    if provider == "ollama":
        return _call_ollama_model(
            prompt,
            JUDGE_MODEL or OLLAMA_MODEL,
            JUDGE_TEMPERATURE,
            JUDGE_TIMEOUT_SECONDS,
        )

    if provider == "openai":
        return _call_openai_compatible(prompt, JUDGE_MODEL)

    raise RuntimeError(
        f"Unknown JUDGE_PROVIDER={JUDGE_PROVIDER}; choose same / dashscope / ollama / openai."
    )


def call_multimodal_llm(prompt: str, image: Image.Image) -> str:
    provider = (LLM_PROVIDER or "").strip().lower()

    if provider == "dashscope":
        if not _is_multimodal_model(QWEN_MODEL):
            raise RuntimeError(
                "当前 DashScope 模型不是多模态模型。请把 config.yaml 中 "
                "dashscope.model 切换为支持图片输入的模型后再试。"
            )
        return _call_dashscope_multimodal(prompt, image)

    if provider == "ollama":
        if not _is_ollama_multimodal_model(OLLAMA_MODEL):
            raise RuntimeError(
                "当前 Ollama 模型看起来不是多模态模型。请切换为 llava、"
                "qwen2.5vl、minicpm-v 等支持图片输入的模型后再试。"
            )
        return _call_ollama_multimodal(prompt, image)

    raise RuntimeError(
        f"未知 LLM_PROVIDER={LLM_PROVIDER}，可选：dashscope / ollama"
    )
