"""语音合成模块。

系统生成文本回答后，可以通过 Edge TTS 转换为 MP3，
供 Streamlit 页面直接播放，形成“文字回答 + 语音播报”的导览体验。
"""

import os
import asyncio
import edge_tts
from .config import VOICE, OUTPUT_DIR


async def _gen_mp3(text: str, out_path: str):
    """调用 Edge TTS 异步生成 MP3 文件。"""
    communicate = edge_tts.Communicate(text, voice=VOICE)
    await communicate.save(out_path)


def text_to_mp3(text: str, filename: str = "output.mp3") -> str:
    """把回答文本保存为 MP3，并返回音频文件路径。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, filename)
    asyncio.run(_gen_mp3(text, out_path))
    return out_path
