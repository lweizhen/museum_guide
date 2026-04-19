from __future__ import annotations

from io import BytesIO

import streamlit as st
from PIL import Image

from src.config import REMOTE_VL_BASE_URL
from src.image_retriever import assess_image_match_confidence, search_image
from src.llm import call_llm
from src.prompt import build_citation, build_prompt
from src.remote_vl import call_remote_vl_rag_lora
from src.retriever import retrieve
from src.tts import text_to_mp3


DEFAULT_IMAGE_QUESTION = "请介绍这件文物的名称、年代、特点和文化价值。"


st.set_page_config(page_title="文物语音导览助手", layout="centered")


def load_uploaded_image(uploaded_file) -> Image.Image:
    data = uploaded_file.read()
    return Image.open(BytesIO(data)).convert("RGB")


def render_contexts(contexts: list[tuple[str, float]]) -> None:
    with st.expander("检索到的知识依据", expanded=True):
        for idx, (text, score) in enumerate(contexts, start=1):
            st.markdown(f"**依据 {idx}**（score={score:.4f}）")
            st.write(text)


def render_image_matches(matches: list[dict[str, str | float]]) -> None:
    with st.expander("图像检索候选文物", expanded=True):
        for idx, item in enumerate(matches, start=1):
            name = item.get("name", "-")
            era = item.get("era", "-")
            museum = item.get("museum", "-")
            score = float(item.get("score", 0.0))
            st.write(f"{idx}. {name} | {era} | {museum} | score={score:.4f}")


def render_answer(
    answer: str,
    contexts: list[tuple[str, float]] | None,
    audio_name: str,
) -> None:
    contexts = contexts or []
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


def build_image_query(match: dict[str, str | float], user_question: str) -> str:
    name = str(match.get("name", "")).strip()
    era = str(match.get("era", "")).strip()
    museum = str(match.get("museum", "")).strip()

    prefix_parts = [part for part in [name, era, museum] if part and part != "-"]
    prefix = " ".join(prefix_parts)
    question = user_question.strip() or DEFAULT_IMAGE_QUESTION

    if prefix:
        return f"{prefix} {question}"
    return question


def get_image_matches(image: Image.Image) -> list[dict[str, str | float]]:
    matches = search_image(image)
    if not matches:
        raise RuntimeError("没有检索到候选文物，请确认图片索引已经构建。")
    return matches


def get_confident_match(
    matches: list[dict[str, str | float]],
) -> dict[str, str | float]:
    confident, reason = assess_image_match_confidence(matches)
    if not confident:
        raise RuntimeError(f"图像检索结果不够稳定：{reason}")
    return matches[0]


def run_scheme_a_image_qa(image: Image.Image, user_question: str) -> None:
    matches = get_image_matches(image)
    render_image_matches(matches)
    best_match = get_confident_match(matches)

    query = build_image_query(best_match, user_question)
    contexts = retrieve(query)
    if not contexts:
        raise RuntimeError("没有检索到可用于回答的文物知识依据。")

    render_contexts(contexts)
    prompt = build_prompt(query, contexts)
    answer = call_llm(prompt)
    render_answer(answer, contexts, "scheme_a_answer.mp3")


def run_vl_rag_lora_remote(
    image: Image.Image,
    user_question: str,
    remote_base_url: str,
) -> None:
    matches = get_image_matches(image)
    render_image_matches(matches)
    best_match = get_confident_match(matches)

    query = build_image_query(best_match, user_question)
    contexts = retrieve(query)
    if not contexts:
        raise RuntimeError("没有检索到可用于远程多模态问答的文物知识依据。")

    render_contexts(contexts)
    answer = call_remote_vl_rag_lora(
        image=image,
        question=query,
        contexts=contexts,
        base_url=remote_base_url,
    )
    render_answer(answer, contexts, "vl_rag_lora_answer.mp3")


def main() -> None:
    st.title("文物语音导览助手")
    st.caption(
        "页面提供两个展示入口：方案 A 使用本地图像检索增强文本 RAG；"
        "方案 B4 通过 SSH 隧道调用远程 GPU 上的 Qwen2.5-VL+RAG+LoRA。"
    )

    st.sidebar.header("远程 GPU 设置")
    remote_base_url = st.sidebar.text_input(
        "远程多模态服务地址",
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
            "方案 A：图像检索增强文本问答",
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
            with st.spinner("正在检索文物并生成回答，请稍候..."):
                if scheme.startswith("方案 A"):
                    run_scheme_a_image_qa(image, user_question)
                else:
                    run_vl_rag_lora_remote(
                        image,
                        user_question,
                        remote_base_url,
                    )
        except Exception as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
