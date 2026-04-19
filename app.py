from __future__ import annotations

from io import BytesIO
from typing import Any

import streamlit as st
from PIL import Image

from src.config import REMOTE_VL_BASE_URL
from src.prompt import build_citation
from src.remote_vl import call_remote_scheme_a, call_remote_vl_rag_lora
from src.tts import text_to_mp3


DEFAULT_IMAGE_QUESTION = "请介绍这件文物的名称、年代、特点和文化价值。"


st.set_page_config(page_title="文物语音导览助手", layout="centered")


def load_uploaded_image(uploaded_file) -> Image.Image:
    data = uploaded_file.read()
    return Image.open(BytesIO(data)).convert("RGB")


def normalize_contexts(raw_contexts: list[dict[str, Any]] | None) -> list[tuple[str, float]]:
    contexts: list[tuple[str, float]] = []
    for item in raw_contexts or []:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        contexts.append((text, score))
    return contexts


def render_contexts(contexts: list[tuple[str, float]]) -> None:
    if not contexts:
        return
    with st.expander("检索到的知识依据", expanded=True):
        for idx, (text, score) in enumerate(contexts, start=1):
            st.markdown(f"**依据 {idx}**（score={score:.4f}）")
            st.write(text)


def render_image_matches(matches: list[dict[str, Any]] | None) -> None:
    if not matches:
        return
    with st.expander("图像检索候选文物", expanded=True):
        for idx, item in enumerate(matches, start=1):
            name = item.get("name", "-")
            era = item.get("era", "-")
            museum = item.get("museum", "-")
            try:
                score = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            st.write(f"{idx}. {name} | {era} | {museum} | score={score:.4f}")


def render_answer(
    answer: str,
    contexts: list[tuple[str, float]],
    audio_name: str,
) -> None:
    citation = build_citation(contexts)
    final_text = answer.strip() + (("\n\n" + citation) if citation else "")

    st.subheader("导览回答")
    st.write(final_text)

    try:
        mp3_path = text_to_mp3(final_text, filename=audio_name)
        with open(mp3_path, "rb") as file:
            st.audio(file.read(), format="audio/mp3")
    except Exception as exc:  # pragma: no cover - UI fallback
        st.warning("语音播报生成失败，但文本回答已正常生成。")
        st.caption(str(exc))


def render_remote_result(result: dict[str, Any], audio_name: str) -> None:
    query = str(result.get("query", "")).strip()
    if query:
        st.caption(f"远程实际查询：{query}")

    render_image_matches(result.get("matches"))
    contexts = normalize_contexts(result.get("contexts"))
    render_contexts(contexts)
    render_answer(str(result.get("answer", "")), contexts, audio_name)


def main() -> None:
    st.title("文物语音导览助手")
    st.caption(
        "本地页面只负责上传图片、输入问题和展示结果；方案 A 与方案 B4 "
        "均通过 SSH 隧道调用远程 GPU 服务完成检索和模型生成。"
    )

    st.sidebar.header("远程 GPU 设置")
    remote_base_url = st.sidebar.text_input(
        "远程导览服务地址",
        value=REMOTE_VL_BASE_URL,
        help="本地通过 SSH 隧道访问远程服务时通常保持默认值即可。",
    )
    st.sidebar.markdown("本地另开终端建立隧道：")
    st.sidebar.code(
        "ssh -p 21870 -L 8000:127.0.0.1:8000 "
        "root@connect.cqa1.seetacloud.com",
        language="bash",
    )

    scheme = st.radio(
        "选择问答方案",
        [
            "方案 A：图像检索增强文本问答（远程 GPU）",
            "方案 B4：Qwen2.5-VL + RAG + LoRA（远程 GPU）",
        ],
    )

    uploaded_file = st.file_uploader(
        "上传文物图片",
        type=["jpg", "jpeg", "png", "webp"],
    )
    user_question = st.text_input(
        "输入问题",
        value=DEFAULT_IMAGE_QUESTION,
        placeholder="例如：这件文物是什么年代的？有什么文化价值？",
    )

    if uploaded_file is not None:
        image = load_uploaded_image(uploaded_file)
        st.image(image, caption="已上传图片", use_container_width=True)
    else:
        image = None

    if st.button("生成导览回答", type="primary"):
        if image is None:
            st.error("请先上传一张文物图片。")
            return

        try:
            with st.spinner("正在调用远程 GPU 服务生成回答，请稍候..."):
                if scheme.startswith("方案 A"):
                    result = call_remote_scheme_a(
                        image=image,
                        question=user_question,
                        base_url=remote_base_url,
                    )
                    render_remote_result(result, "scheme_a_answer.mp3")
                else:
                    result = call_remote_vl_rag_lora(
                        image=image,
                        question=user_question,
                        base_url=remote_base_url,
                    )
                    render_remote_result(result, "vl_rag_lora_answer.mp3")
        except Exception as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
