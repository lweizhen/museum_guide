import streamlit as st
import os
from src.retriever import retrieve
from src.prompt import build_prompt, build_citation
from src.llm import call_llm
from src.tts import text_to_mp3

st.set_page_config(page_title="博物馆智能语音导览", layout="centered")

st.title("🧭 博物馆智能语音导览助手")
st.caption("RAG（FAISS语义检索） + 通义千问/本地大模型生成 + TTS语音播报")

query = st.text_input("请输入你的问题：", placeholder="例如：介绍一下马踏飞燕？")

if st.button("生成讲解", type="primary"):
    if not query.strip():
        st.warning("请输入问题")
    else:
        contexts = retrieve(query)
        if not contexts:
            st.error("未在知识库中检索到足够相关的内容，请换个问法或补充展品名称。")
        else:
            with st.expander("检索到的知识（可解释性）", expanded=True):
                for text, score in contexts:
                    st.write(f"- (score={score:.4f}) {text}")

            prompt = build_prompt(query, contexts)
            answer = call_llm(prompt)

            citation = build_citation(contexts)
            final_text = answer.strip() + (("\n\n" + citation) if citation else "")

            st.subheader("生成的讲解")
            st.write(final_text)

            mp3_path = text_to_mp3(final_text, filename="output_streamlit.mp3")
            st.subheader("语音播放")
            with open(mp3_path, "rb") as f:
                st.audio(f.read(), format="audio/mp3")
            st.success("语音已生成并可播放")
