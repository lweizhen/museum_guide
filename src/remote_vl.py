from __future__ import annotations

import json
from io import BytesIO
from typing import Any

import requests
from PIL import Image

from .config import REMOTE_VL_BASE_URL, REMOTE_VL_TIMEOUT_SECONDS


def call_remote_vl_rag_lora(
    *,
    image: Image.Image,
    question: str,
    contexts: list[tuple[str, float]],
    base_url: str = REMOTE_VL_BASE_URL,
    timeout: int = REMOTE_VL_TIMEOUT_SECONDS,
) -> str:
    """Call the remote GPU Qwen2.5-VL+RAG+LoRA generation service."""

    base_url = base_url.rstrip("/")
    payload_contexts = [
        {"text": text, "score": float(score)}
        for text, score in contexts
    ]

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    buffer.seek(0)

    try:
        resp = requests.post(
            f"{base_url}/generate",
            data={
                "question": question,
                "contexts_json": json.dumps(payload_contexts, ensure_ascii=False),
            },
            files={"image": ("query.jpg", buffer, "image/jpeg")},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            "远程多模态服务连接失败。请确认 SSH 隧道已开启，"
            f"并且远程服务正在监听：{base_url}"
        ) from exc

    if resp.status_code >= 400:
        raise RuntimeError(f"远程多模态服务返回错误：{resp.status_code} {resp.text}")

    data: dict[str, Any] = resp.json()
    answer = str(data.get("answer", "")).strip()
    if not answer:
        raise RuntimeError(f"远程多模态服务未返回有效回答：{data}")
    return answer
