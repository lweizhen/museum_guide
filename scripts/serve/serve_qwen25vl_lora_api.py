from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from src.hf_qwen_vl import HfQwenVlGenerator  # noqa: E402
from src.image_retriever import assess_image_match_confidence, search_image  # noqa: E402
from src.llm import call_llm  # noqa: E402
from src.prompt import build_multimodal_grounded_prompt, build_prompt  # noqa: E402
from src.retriever import retrieve  # noqa: E402


DEFAULT_IMAGE_QUESTION = "请介绍这件文物的名称、年代、特点和文化价值。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Serve remote Scheme A and Qwen2.5-VL+RAG+LoRA for local Streamlit demo."
        )
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


def serialize_contexts(contexts: list[tuple[str, float]]) -> list[dict[str, Any]]:
    return [{"text": text, "score": float(score)} for text, score in contexts]


def serialize_matches(matches: list[dict[str, str | float]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for item in matches:
        serialized.append(
            {
                "name": item.get("name", ""),
                "era": item.get("era", ""),
                "museum": item.get("museum", ""),
                "category": item.get("category", ""),
                "detail_url": item.get("detail_url", ""),
                "image_url": item.get("image_url", ""),
                "local_path": item.get("local_path", ""),
                "score": float(item.get("score", 0.0)),
            }
        )
    return serialized


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


async def save_upload_to_temp(image: UploadFile) -> Path:
    suffix = Path(image.filename or "query.jpg").suffix or ".jpg"
    content = await image.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        return Path(tmp.name)


def build_grounding(image_path: Path, question: str) -> dict[str, Any]:
    pil_image = Image.open(image_path).convert("RGB")
    matches = search_image(pil_image)
    if not matches:
        raise RuntimeError("没有检索到候选文物，请确认图片索引已经构建。")

    confident, reason = assess_image_match_confidence(matches)
    if not confident:
        raise RuntimeError(f"图像检索结果不够稳定：{reason}")

    query = build_image_query(matches[0], question)
    contexts = retrieve(query)
    if not contexts:
        raise RuntimeError("没有检索到可用于回答的文物知识依据。")

    return {
        "query": query,
        "matches": matches,
        "contexts": contexts,
    }


ARGS = parse_args()
RUNNER = HfQwenVlGenerator(
    ARGS.model_path,
    ARGS.adapter_path,
    bf16=not ARGS.no_bf16,
    max_pixels=ARGS.max_pixels,
)

app = FastAPI(title="Museum Guide Remote Demo API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/scheme-a/generate")
async def generate_scheme_a(
    question: str = Form(""),
    image: UploadFile = File(...),
) -> JSONResponse:
    tmp_path: Path | None = None
    try:
        tmp_path = await save_upload_to_temp(image)
        grounding = build_grounding(tmp_path, question)
        prompt = build_prompt(grounding["query"], grounding["contexts"])
        answer = call_llm(prompt)
        return JSONResponse(
            {
                "answer": answer,
                "query": grounding["query"],
                "matches": serialize_matches(grounding["matches"]),
                "contexts": serialize_contexts(grounding["contexts"]),
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.post("/vl-rag-lora/generate")
async def generate_vl_rag_lora(
    question: str = Form(""),
    image: UploadFile = File(...),
) -> JSONResponse:
    tmp_path: Path | None = None
    try:
        tmp_path = await save_upload_to_temp(image)
        grounding = build_grounding(tmp_path, question)
        prompt = build_multimodal_grounded_prompt(
            grounding["query"],
            grounding["contexts"],
        )
        answer = RUNNER.generate(
            prompt,
            tmp_path,
            max_new_tokens=ARGS.max_new_tokens,
        )
        return JSONResponse(
            {
                "answer": answer,
                "query": grounding["query"],
                "matches": serialize_matches(grounding["matches"]),
                "contexts": serialize_contexts(grounding["contexts"]),
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.post("/generate")
async def generate_legacy(
    question: str = Form(""),
    contexts_json: str = Form("[]"),
    image: UploadFile = File(...),
) -> JSONResponse:
    """Backward-compatible endpoint for answer-only VLM generation."""

    try:
        contexts = parse_contexts(contexts_json)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not contexts:
        raise HTTPException(status_code=400, detail="contexts_json is empty")

    tmp_path: Path | None = None
    try:
        tmp_path = await save_upload_to_temp(image)
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
