"""远程 GPU 导览服务客户端。

本地 Streamlit 页面不会直接加载 Qwen2.5-VL，而是通过 SSH 隧道把图片和问题
发送到远程 FastAPI 服务。本模块负责封装 HTTP 请求和错误提示。
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import requests
from PIL import Image

from .config import REMOTE_VL_BASE_URL, REMOTE_VL_TIMEOUT_SECONDS


GuideResult = dict[str, Any]


def _post_image_request(
    *,
    endpoint: str,
    image: Image.Image,
    question: str,
    base_url: str,
    timeout: int,
) -> GuideResult:
    """把图片和问题发送到远程指定接口，并返回 JSON 结果。"""
    base_url = base_url.rstrip("/")

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    buffer.seek(0)

    try:
        resp = requests.post(
            f"{base_url}{endpoint}",
            data={"question": question},
            files={"image": ("query.jpg", buffer, "image/jpeg")},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            "远程导览服务连接失败。请确认 SSH 隧道已开启，"
            f"并且远程服务正在监听：{base_url}"
        ) from exc

    if resp.status_code >= 400:
        raise RuntimeError(f"远程导览服务返回错误：{resp.status_code} {resp.text}")

    data: GuideResult = resp.json()
    answer = str(data.get("answer", "")).strip()
    if not answer:
        raise RuntimeError(f"远程导览服务未返回有效回答：{data}")
    return data


def call_remote_scheme_a(
    *,
    image: Image.Image,
    question: str,
    base_url: str = REMOTE_VL_BASE_URL,
    timeout: int = REMOTE_VL_TIMEOUT_SECONDS,
) -> GuideResult:
    """调用远程方案 A：图片检索 + 文本 RAG + 文本大模型。"""

    return _post_image_request(
        endpoint="/scheme-a/generate",
        image=image,
        question=question,
        base_url=base_url,
        timeout=timeout,
    )


def call_remote_vl_rag_lora(
    *,
    image: Image.Image,
    question: str,
    base_url: str = REMOTE_VL_BASE_URL,
    timeout: int = REMOTE_VL_TIMEOUT_SECONDS,
) -> GuideResult:
    """调用远程方案 B4：图片检索 + RAG + Qwen2.5-VL LoRA。"""

    return _post_image_request(
        endpoint="/vl-rag-lora/generate",
        image=image,
        question=question,
        base_url=base_url,
        timeout=timeout,
    )
