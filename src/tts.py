import os
import asyncio
import edge_tts
from .config import VOICE, OUTPUT_DIR

async def _gen_mp3(text: str, out_path: str):
    communicate = edge_tts.Communicate(text, voice=VOICE)
    await communicate.save(out_path)

def text_to_mp3(text: str, filename: str = "output.mp3") -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, filename)
    asyncio.run(_gen_mp3(text, out_path))
    return out_path
