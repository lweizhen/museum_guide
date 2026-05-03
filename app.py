from __future__ import annotations

from io import BytesIO
from typing import Any

import streamlit as st
from PIL import Image

from src.config import REMOTE_VL_BASE_URL
from src.prompt import build_citation
from src.remote_vl import call_remote_scheme_a, call_remote_vl_rag_lora
from src.tts import text_to_mp3


DEFAULT_IMAGE_QUESTION = "??????????????????????"


st.set_page_config(page_title="????????", layout="centered")


SCHEME_A_LABEL = "?? A?????????????? GPU?"
SCHEME_B4_LABEL = "?? B4?Qwen2.5-VL + RAG + LoRA??????????? GPU?"


SCHEME_DESCRIPTIONS = {
    SCHEME_A_LABEL: (
        "???????????????????????????"
        "???????????????????"
    ),
    SCHEME_B4_LABEL: (
        "???????????????? Qwen2.5-VL + RAG + LoRA ?????"
        "??????????????????????????????????????????"
    ),
}


EXAMPLE_QUESTIONS = [
    "??????????",
    "???????????",
    "????????????????",
    "???????????????",
    "??????????????",
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
    with st.expander("????????", expanded=True):
        for idx, (text, score) in enumerate(contexts, start=1):
            st.markdown(f"**?? {idx}**?score={score:.4f}?")
            st.write(text)



def render_image_matches(matches: list[dict[str, Any]] | None) -> None:
    if not matches:
        return
    with st.expander("????????", expanded=True):
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

    st.subheader("????")
    st.write(final_text)

    try:
        mp3_path = text_to_mp3(final_text, filename=audio_name)
        with open(mp3_path, "rb") as file:
            st.audio(file.read(), format="audio/mp3")
    except Exception as exc:  # pragma: no cover - UI fallback
        st.warning("????????????????????")
        st.caption(str(exc))



def render_remote_result(result: dict[str, Any], audio_name: str) -> None:
    query = str(result.get("query", "")).strip()
    if query:
        st.caption(f"???????{query}")

    render_image_matches(result.get("matches"))
    contexts = normalize_contexts(result.get("contexts"))
    render_contexts(contexts)
    render_answer(str(result.get("answer", "")), contexts, audio_name)



def main() -> None:
    st.title("????????")
    st.caption(
        "??????????????????????"
        "?? A ??? B4 ??? SSH ?????? GPU ????????????"
    )

    st.sidebar.header("?? GPU ??")
    remote_base_url = st.sidebar.text_input(
        "????????",
        value=REMOTE_VL_BASE_URL,
        help="???? SSH ???????????????????",
    )
    st.sidebar.markdown("???????? SSH ???")
    st.sidebar.code(
        "ssh -p 21870 -L 8000:127.0.0.1:8000 root@connect.cqa1.seetacloud.com",
        language="bash",
    )
    st.sidebar.info(
        "????????????????????? B4?"
        "???? B4 ????????????????"
        "????????????????????"
    )

    scheme = st.radio("??????", [SCHEME_A_LABEL, SCHEME_B4_LABEL])
    st.markdown(f"**??????**?{SCHEME_DESCRIPTIONS[scheme]}")

    with st.expander("????", expanded=False):
        for example in EXAMPLE_QUESTIONS:
            st.write(f"- {example}")

    uploaded_file = st.file_uploader("??????", type=["jpg", "jpeg", "png", "webp"])
    user_question = st.text_input(
        "????",
        value=DEFAULT_IMAGE_QUESTION,
        placeholder="??????????????????????",
    )

    if uploaded_file is not None:
        image = load_uploaded_image(uploaded_file)
        st.image(image, caption="?????", use_container_width=True)
    else:
        image = None

    if st.button("??????", type="primary"):
        if image is None:
            st.error("???????????")
            return

        try:
            with st.spinner("?????? GPU ??????????..."):
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
