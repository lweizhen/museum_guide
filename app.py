from __future__ import annotations

from io import BytesIO
from typing import Any

import streamlit as st
from PIL import Image

from src.config import REMOTE_VL_BASE_URL
from src.prompt import build_citation
from src.remote_vl import call_remote_scheme_a, call_remote_vl_rag_lora
from src.tts import text_to_mp3


DEFAULT_IMAGE_QUESTION = "请介绍这件文物，并说明它最值得关注的特点。"


st.set_page_config(page_title="文物语音导览助手", layout="centered")


SCHEME_A_LABEL = "方案 A：图像检索增强文本问答（远程 GPU）"
SCHEME_B4_LABEL = "方案 B4：Qwen2.5-VL + RAG + LoRA（自动切换提示词，远程 GPU）"


SCHEME_DESCRIPTIONS = {
    SCHEME_A_LABEL: (
        "先进行图片检索，再结合文本知识库完成回答生成。"
        "该方案链路清晰、可解释性强，是当前工程基线。"
    ),
    SCHEME_B4_LABEL: (
        "使用图片检索、文本检索、Qwen2.5-VL 与 LoRA 适配器联合推理。"
        "当前版本默认启用按问题类型自动切换提示词，字段型问题优先短答，"
        "导览型问题优先生成更自然的讲解。"
    ),
}


EXAMPLE_QUESTIONS = [
    "这件文物叫什么名字？",
    "这件文物属于什么时代？",
    "这件文物主要有什么用途？",
    "这件文物体现了怎样的文化价值？",
    "请为这件文物生成一段导览讲解。",
]


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
            st.markdown(f"**片段 {idx}**：score={score:.4f}")
            st.write(text)


def render_image_matches(matches: list[dict[str, Any]] | None) -> None:
    if not matches:
        return
    with st.expander("图片检索候选文物", expanded=True):
        for idx, item in enumerate(matches, start=1):
            name = item.get("name", "-")
            era = item.get("era", "-")
            museum = item.get("museum", "-")
            try:
                score = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            st.write(f"{idx}. {name} | {era} | {museum} | score={score:.4f}")


def render_answer(answer: str, contexts: list[tuple[str, float]], audio_name: str) -> None:
    citation = build_citation(contexts)
    final_text = answer.strip() + ((chr(10) + chr(10) + citation) if citation else "")

    st.subheader("系统回答")
    st.write(final_text)

    try:
        mp3_path = text_to_mp3(final_text, filename=audio_name)
        with open(mp3_path, "rb") as file:
            st.audio(file.read(), format="audio/mp3")
    except Exception as exc:  # pragma: no cover - UI fallback
        st.warning("语音播报生成失败，已保留文本结果。")
        st.caption(str(exc))


def render_remote_result(result: dict[str, Any], audio_name: str) -> None:
    query = str(result.get("query", "")).strip()
    if query:
        st.caption(f"检索查询：{query}")

    render_image_matches(result.get("matches"))
    contexts = normalize_contexts(result.get("contexts"))
    render_contexts(contexts)
    render_answer(str(result.get("answer", "")), contexts, audio_name)


def main() -> None:
    st.title("文物语音导览助手")
    st.caption(
        "当前页面支持工程基线方案 A 和多模态 LoRA 方案 B4。"
        "两条链路都通过 SSH 隧道调用远程 GPU 服务完成推理。"
    )

    st.sidebar.header("远程 GPU 设置")
    remote_base_url = st.sidebar.text_input(
        "远程服务地址",
        value=REMOTE_VL_BASE_URL,
        help="如果你已经建立 SSH 隧道，通常保持默认地址即可。",
    )
    st.sidebar.markdown("推荐的 SSH 隧道命令")
    st.sidebar.code(
        "ssh -p 21870 -L 8000:127.0.0.1:8000 root@connect.cqa1.seetacloud.com",
        language="bash",
    )
    st.sidebar.info(
        "如果你当前重点展示最新版系统，推荐优先使用方案 B4。"
        "该方案默认启用自动切换提示词，字段型问题与导览型问题会采用不同回答策略。"
    )

    scheme = st.radio("选择演示方案", [SCHEME_A_LABEL, SCHEME_B4_LABEL])
    st.markdown(f"**当前方案说明**：{SCHEME_DESCRIPTIONS[scheme]}")

    with st.expander("示例问题", expanded=False):
        for example in EXAMPLE_QUESTIONS:
            st.write(f"- {example}")

    uploaded_file = st.file_uploader("上传文物图片", type=["jpg", "jpeg", "png", "webp"])
    user_question = st.text_input(
        "输入问题",
        value=DEFAULT_IMAGE_QUESTION,
        placeholder="例如：这件文物叫什么？它反映了怎样的历史背景？",
    )

    if uploaded_file is not None:
        image = load_uploaded_image(uploaded_file)
        st.image(image, caption="上传的图片", use_container_width=True)
    else:
        image = None

    if st.button("开始生成回答", type="primary"):
        if image is None:
            st.error("请先上传图片。")
            return

        try:
            with st.spinner("正在调用远程 GPU 服务，请稍候..."):
                if scheme == SCHEME_A_LABEL:
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
