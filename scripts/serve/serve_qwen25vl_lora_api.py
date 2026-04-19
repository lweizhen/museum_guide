from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from src.hf_qwen_vl import HfQwenVlGenerator  # noqa: E402
from src.prompt import build_multimodal_grounded_prompt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve Qwen2.5-VL + LoRA for local Streamlit demo via HTTP."
    )
    parser.add_argument("--model-path", required=True, help="HF Qwen2.5-VL model path.")
    parser.add_argument("--adapter-path", required=True, help="LoRA adapter path.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-pixels", type=int, default=512 * 512)
    parser.add_argument("--no-bf16", action="store_true")
    return parser.parse_args()


def parse_contexts(raw: str) -> list[tuple[str, float]]:
    if not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("contexts_json is not valid JSON") from exc

    if not isinstance(data, list):
        raise ValueError("contexts_json must be a list")

    contexts: list[tuple[str, float]] = []
    for item in data:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            score = float(item.get("score", 0.0))
        elif isinstance(item, (list, tuple)) and item:
            text = str(item[0]).strip()
            score = float(item[1]) if len(item) > 1 else 0.0
        else:
            continue
        if text:
            contexts.append((text, score))
    return contexts


ARGS = parse_args()
RUNNER = HfQwenVlGenerator(
    ARGS.model_path,
    ARGS.adapter_path,
    bf16=not ARGS.no_bf16,
    max_pixels=ARGS.max_pixels,
)

app = FastAPI(title="Museum Guide Qwen2.5-VL LoRA API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate")
async def generate(
    question: str = Form(""),
    contexts_json: str = Form("[]"),
    image: UploadFile = File(...),
) -> JSONResponse:
    try:
        contexts = parse_contexts(contexts_json)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not contexts:
        raise HTTPException(status_code=400, detail="contexts_json is empty")

    suffix = Path(image.filename or "query.jpg").suffix or ".jpg"
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await image.read())
            tmp_path = Path(tmp.name)

        prompt = build_multimodal_grounded_prompt(question, contexts)
        answer = RUNNER.generate(
            prompt,
            tmp_path,
            max_new_tokens=ARGS.max_new_tokens,
        )
        return JSONResponse({"answer": answer})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=ARGS.host, port=ARGS.port)
