from __future__ import annotations

from io import BytesIO

import streamlit as st
from PIL import Image

from src.image_retriever import assess_image_match_confidence, search_image
from src.llm import call_llm, call_multimodal_llm
from src.prompt import (
    build_citation,
    build_multimodal_direct_prompt,
    build_multimodal_grounded_prompt,
    build_prompt,
)
from src.retriever import retrieve
from src.tts import text_to_mp3


st.set_page_config(page_title="Museum Guide", layout="centered")


def render_contexts(contexts: list[tuple[str, float]]) -> None:
    with st.expander("检索到的展品资料", expanded=True):
        for text, score in contexts:
            st.write(f"- (score={score:.4f}) {text}")


def render_image_matches(matches: list[dict[str, str | float]]) -> None:
    with st.expander("相似文物候选", expanded=True):
        for item in matches:
            st.write(
                f"- {item['name']}（{item['era']}）"
                f" | {item['museum']}"
                f" | score={float(item['score']):.4f}"
            )


def render_answer(
    answer: str,
    contexts: list[tuple[str, float]] | None,
    audio_name: str,
) -> None:
    contexts = contexts or []
    citation = build_citation(contexts)
    final_text = answer.strip() + (("\n\n" + citation) if citation else "")

    st.subheader("回答")
    st.write(final_text)

    mp3_path = text_to_mp3(final_text, filename=audio_name)
    with open(mp3_path, "rb") as file:
        st.audio(file.read(), format="audio/mp3")


def run_text_qa(query: str, audio_name: str) -> None:
    contexts = retrieve(query)
    if not contexts:
        st.error("没有检索到相关文物信息，请换一种问法试试。")
        return

    render_contexts(contexts)
    prompt = build_prompt(query, contexts)
    answer = call_llm(prompt)
    render_answer(answer, contexts, audio_name)


def build_image_query(match: dict[str, str | float], user_question: str) -> str:
    name = str(match.get("name", "")).strip()
    era = str(match.get("era", "")).strip()
    museum = str(match.get("museum", "")).strip()

    prefix_parts = [part for part in [name, era, museum] if part and part != "-"]
    prefix = " ".join(prefix_parts)
    default_question = "请介绍这件文物的名称、时代、类型和主要特点。"

    if user_question.strip():
        if prefix:
            return f"{prefix} {user_question.strip()}"
        return user_question.strip()

    if prefix:
        return f"{prefix} {default_question}"
    return default_question


def load_uploaded_image(uploaded_file) -> Image.Image | None:
    if uploaded_file is None:
        st.warning("请先上传一张文物图片。")
        return None

    try:
        image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
        st.image(image, caption="上传的文物图片", use_container_width=True)
        return image
    except Exception as exc:
        st.error(f"图片读取失败：{exc}")
        return None


def get_image_matches(image: Image.Image) -> list[dict[str, str | float]] | None:
    try:
        matches = search_image(image, top_k=5)
    except Exception as exc:
        st.error(
            "图片索引检索失败。请先确认已经执行 `python build_image_index.py` 构建图片索引。"
        )
        st.caption(str(exc))
        return None

    if not matches:
        st.error("没有找到相似文物，请更换更清晰的图片再试。")
        return None

    return matches


def get_confident_image_match(
    image: Image.Image,
) -> tuple[dict[str, str | float], list[dict[str, str | float]]] | None:
    matches = get_image_matches(image)
    if not matches:
        return None

    confident, reason = assess_image_match_confidence(matches)
    if not confident:
        st.error(reason)
        st.info("当前结果不够可靠，系统已主动拒答。建议换一张更清晰、角度更正的图片再试。")
        render_image_matches(matches)
        return None

    best_match = matches[0]
    st.subheader("图片识别结果")
    st.write(
        f"最可能的文物：{best_match['name']}（{best_match['era']}）"
        f" | 馆藏单位：{best_match['museum']}"
        f" | 相似度：{float(best_match['score']):.4f}"
    )
    render_image_matches(matches)
    return best_match, matches


def run_scheme_a_image_qa(image: Image.Image, user_question: str) -> None:
    match_result = get_confident_image_match(image)
    if not match_result:
        return

    best_match, _matches = match_result
    query = build_image_query(best_match, user_question)
    st.caption(f"转换后的文本查询：{query}")
    run_text_qa(query, "output_streamlit_scheme_a.mp3")


def run_scheme_b_direct(image: Image.Image, user_question: str) -> None:
    st.info("当前为方案B实验分支：图片和问题将直接送入多模态模型，不先经过知识库校验。")
    prompt = build_multimodal_direct_prompt(user_question)
    answer = call_multimodal_llm(prompt, image)
    render_answer(answer, None, "output_streamlit_scheme_b_direct.mp3")


def run_scheme_b_grounded(image: Image.Image, user_question: str) -> None:
    match_result = get_confident_image_match(image)
    if not match_result:
        return

    best_match, _matches = match_result
    query = build_image_query(best_match, user_question)
    st.caption(f"用于知识库增强的查询：{query}")

    contexts = retrieve(query)
    if not contexts:
        st.error("图片候选已找到，但没有检索到足够可靠的文物资料。")
        return

    render_contexts(contexts)
    prompt = build_multimodal_grounded_prompt(user_question, contexts)
    answer = call_multimodal_llm(prompt, image)
    render_answer(answer, contexts, "output_streamlit_scheme_b_grounded.mp3")


st.title("博物馆导览助手")
st.caption("支持文字问答、图片识别增强问答，以及端到端多模态实验分支。")

text_tab, image_tab = st.tabs(["文字问答", "图片识别"])

with text_tab:
    text_query = st.text_input(
        "请输入你想了解的展品问题",
        placeholder="例如：请介绍一下后母戊鼎的发现过程",
    )
    if st.button("开始问答", type="primary"):
        if not text_query.strip():
            st.warning("请输入一个问题。")
        else:
            run_text_qa(text_query.strip(), "output_streamlit_text.mp3")

with image_tab:
    image_mode = st.radio(
        "请选择图片问答模式",
        options=[
            "方案A：图片检索 + 文本RAG",
            "方案B：端到端多模态直答",
            "方案B：多模态 + 知识库增强",
        ],
        index=0,
    )
    uploaded_file = st.file_uploader(
        "上传一张文物图片",
        type=["png", "jpg", "jpeg", "webp"],
    )
    image_question = st.text_input(
        "如果你有具体问题，也可以一起问",
        placeholder="例如：这件文物有什么特点？",
    )

    if st.button("开始图片问答"):
        image = load_uploaded_image(uploaded_file)
        if image is None:
            st.stop()

        try:
            if image_mode == "方案A：图片检索 + 文本RAG":
                run_scheme_a_image_qa(image, image_question)
            elif image_mode == "方案B：端到端多模态直答":
                run_scheme_b_direct(image, image_question)
            else:
                run_scheme_b_grounded(image, image_question)
        except Exception as exc:
            st.error("图片问答执行失败。")
            st.caption(str(exc))
