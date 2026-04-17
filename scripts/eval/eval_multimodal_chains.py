from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.eval_utils import (  # noqa: E402
    build_text_query,
    contains_expected,
    extract_field,
    normalize_text,
    read_jsonl,
    safe_div,
    score_answer,
    strip_model_artifacts,
    summarize_counter,
)
from src.hf_qwen_vl import HfQwenVlGenerator  # noqa: E402
from src.image_index import open_local_image  # noqa: E402
from src.image_retriever import assess_image_match_confidence, search_image  # noqa: E402
from src.progress import Progress  # noqa: E402
from src.prompt import build_multimodal_direct_prompt, build_multimodal_grounded_prompt, build_prompt  # noqa: E402
from src.retriever import get_docs, retrieve  # noqa: E402


DATASET_PATH = Path("data/multimodal_eval/test_images.jsonl")
OUT_DIR = Path("outputs/raw")
OUT_CSV = OUT_DIR / "eval_multimodal_chains_results.csv"
OUT_SUMMARY = OUT_DIR / "eval_multimodal_chains_summary.txt"

SUPPORTED_CHAINS = (
    "retrieval_rag_text",
    "vl_direct",
    "vl_rag",
    "vl_lora",
    "vl_rag_lora",
)
RAG_CHAINS = {"retrieval_rag_text", "vl_rag", "vl_rag_lora"}
LORA_CHAINS = {"vl_lora", "vl_rag_lora"}
NAME_LABEL = "\u5c55\u54c1\u540d\u79f0"

FIELDNAMES = [
    "chain",
    "image_id",
    "artifact_id",
    "target_name",
    "target_era",
    "target_museum",
    "question_idx",
    "question",
    "answer_field",
    "gold_answer",
    "image_path",
    "image_found",
    "image_confident",
    "confidence_reason",
    "recognized_name",
    "recognized_era",
    "recognized_museum",
    "recognized_score",
    "recognized_name_correct",
    "context_found",
    "retrieved_names",
    "retrieved_scores",
    "prediction",
    "auto_correct",
    "auto_score",
    "target_name_mentioned",
    "gold_answer_mentioned",
    "latency_seconds",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified five-chain multimodal QA evaluation on test_images.jsonl."
    )
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument(
        "--chains",
        required=True,
        help="Comma-separated chains. Choices: " + ",".join(SUPPORTED_CHAINS),
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="HF Qwen2.5-VL-3B-Instruct model path or id.",
    )
    parser.add_argument(
        "--adapter-path",
        default="",
        help="LoRA adapter path. Required for vl_lora and vl_rag_lora.",
    )
    parser.add_argument("--output", default=str(OUT_CSV))
    parser.add_argument("--summary", default=str(OUT_SUMMARY))
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--limit-questions", type=int, default=0)
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split dataset rows into N shards for multi-process or multi-GPU evaluation.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Shard index to run, 0-based. Used with --num-shards.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-pixels", type=int, default=512 * 512)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true", help="Skip existing chain/image/question rows in output CSV.")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def _parse_chains(raw: str) -> list[str]:
    chains = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [chain for chain in chains if chain not in SUPPORTED_CHAINS]
    if invalid:
        raise ValueError(f"Unsupported chains: {invalid}. Supported: {SUPPORTED_CHAINS}")
    return list(dict.fromkeys(chains))


def _load_existing_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str, str]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            keys.add((row.get("chain", ""), row.get("image_id", ""), row.get("question_idx", "")))
    return keys


def _load_all_output_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _exact_docs_by_name(name: str) -> list[tuple[str, float]]:
    name_norm = normalize_text(name)
    if not name_norm:
        return []
    docs = get_docs()
    exact = [
        (doc, 1.0)
        for doc in docs
        if normalize_text(extract_field(doc, NAME_LABEL)) == name_norm
    ]
    return exact


def _empty_grounding() -> dict[str, Any]:
    return {
        "image_found": "",
        "image_confident": "",
        "confidence_reason": "",
        "recognized_name": "",
        "recognized_era": "",
        "recognized_museum": "",
        "recognized_score": "",
        "recognized_name_correct": "",
        "contexts": [],
        "context_found": "",
        "retrieved_names": "",
        "retrieved_scores": "",
    }


def _retrieve_grounding(image: Any, question: str, top_k: int) -> dict[str, Any]:
    payload = _empty_grounding()
    payload["image_found"] = 0
    payload["image_confident"] = 0
    payload["context_found"] = 0

    matches = search_image(image, top_k=top_k)
    payload["image_found"] = int(bool(matches))
    confident, reason = assess_image_match_confidence(matches)
    payload["image_confident"] = int(confident)
    payload["confidence_reason"] = reason

    if not matches:
        payload["confidence_reason"] = reason or "no_image_match"
        return payload

    best = matches[0]
    recognized_name = str(best.get("name", "")).strip()
    recognized_era = str(best.get("era", "")).strip()
    recognized_museum = str(best.get("museum", "")).strip()
    payload["recognized_name"] = recognized_name
    payload["recognized_era"] = recognized_era
    payload["recognized_museum"] = recognized_museum
    payload["recognized_score"] = f"{float(best.get('score', 0.0)):.4f}"

    if not confident:
        return payload

    query = build_text_query(recognized_name, question, recognized_era, recognized_museum)
    contexts = _exact_docs_by_name(recognized_name) or retrieve(query)
    payload["contexts"] = contexts
    payload["context_found"] = int(bool(contexts))
    payload["retrieved_names"] = "|".join(extract_field(doc, NAME_LABEL) for doc, _score in contexts)
    payload["retrieved_scores"] = "|".join(f"{score:.4f}" for _doc, score in contexts)
    return payload


def _make_prompt(chain: str, question: str, grounding: dict[str, Any]) -> tuple[str, str | None]:
    if chain in {"vl_direct", "vl_lora"}:
        return build_multimodal_direct_prompt(question), None

    if not grounding.get("image_found"):
        return "", "no_image_match"
    if not grounding.get("image_confident"):
        return "", str(grounding.get("confidence_reason") or "image_not_confident")
    if not grounding.get("context_found"):
        return "", "context_not_found"

    contexts = grounding.get("contexts") or []
    if chain == "retrieval_rag_text":
        recognized_name = str(grounding.get("recognized_name", ""))
        recognized_era = str(grounding.get("recognized_era", ""))
        recognized_museum = str(grounding.get("recognized_museum", ""))
        query = build_text_query(recognized_name, question, recognized_era, recognized_museum)
        return build_prompt(query, contexts), None

    return build_multimodal_grounded_prompt(question, contexts), None


def _get_runner(
    chain: str,
    base_runner: HfQwenVlGenerator,
    lora_runner: HfQwenVlGenerator | None,
) -> HfQwenVlGenerator:
    if chain in LORA_CHAINS:
        if lora_runner is None:
            raise RuntimeError(f"{chain} requires --adapter-path")
        return lora_runner
    return base_runner


def _summarize(path: Path, selected_chains: list[str], summary_path: Path) -> None:
    rows = _load_all_output_rows(path)
    totals: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    name_mentions: Counter[str] = Counter()
    gold_mentions: Counter[str] = Counter()
    score_sums: defaultdict[str, float] = defaultdict(float)
    latency_sums: defaultdict[str, float] = defaultdict(float)
    errors: Counter[str] = Counter()
    error_reasons: Counter[str] = Counter()
    context_found: Counter[str] = Counter()
    image_confident: Counter[str] = Counter()

    for row in rows:
        chain = row.get("chain", "")
        if selected_chains and chain not in selected_chains:
            continue
        totals[chain] += 1
        correct[chain] += int(row.get("auto_correct") or 0)
        name_mentions[chain] += int(row.get("target_name_mentioned") or 0)
        gold_mentions[chain] += int(row.get("gold_answer_mentioned") or 0)
        try:
            score_sums[chain] += float(row.get("auto_score") or 0.0)
        except ValueError:
            pass
        try:
            latency_sums[chain] += float(row.get("latency_seconds") or 0.0)
        except ValueError:
            pass
        if row.get("error"):
            errors[chain] += 1
            error_reasons[f"{chain}:{row.get('error')}"] += 1
        if row.get("context_found") not in {"", None}:
            context_found[chain] += int(row.get("context_found") or 0)
        if row.get("image_confident") not in {"", None}:
            image_confident[chain] += int(row.get("image_confident") or 0)

    lines = [
        f"Output: {path}",
        f"Rows in selected chains: {sum(totals.values())}",
        "",
    ]
    for chain in selected_chains:
        total = totals[chain]
        lines.extend(
            [
                f"[{chain}] rows: {total}",
                f"[{chain}] auto accuracy: {safe_div(correct[chain], total):.4f}",
                f"[{chain}] avg auto score: {safe_div(score_sums[chain], total):.4f}",
                f"[{chain}] target name mention rate: {safe_div(name_mentions[chain], total):.4f}",
                f"[{chain}] gold answer mention rate: {safe_div(gold_mentions[chain], total):.4f}",
                f"[{chain}] avg latency seconds: {safe_div(latency_sums[chain], total):.4f}",
                f"[{chain}] errors: {errors[chain]}",
            ]
        )
        if chain in RAG_CHAINS:
            lines.extend(
                [
                    f"[{chain}] image confident rate: {safe_div(image_confident[chain], total):.4f}",
                    f"[{chain}] context found rate: {safe_div(context_found[chain], total):.4f}",
                ]
            )
        lines.append("")

    lines.append("Error reasons:")
    lines.append(summarize_counter(error_reasons))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    chains = _parse_chains(args.chains)
    if any(chain in LORA_CHAINS for chain in chains) and not args.adapter_path:
        raise ValueError("--adapter-path is required for vl_lora or vl_rag_lora")

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")

    samples = read_jsonl(args.dataset, args.limit_images)
    if args.num_shards > 1:
        samples = [
            sample
            for index, sample in enumerate(samples)
            if index % args.num_shards == args.shard_index
        ]
    if not samples:
        raise RuntimeError(f"No dataset rows loaded: {args.dataset}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary)

    existing_keys = _load_existing_keys(output_path) if args.resume else set()
    write_header = not output_path.exists() or not args.resume
    file_mode = "a" if args.resume and output_path.exists() else "w"

    total = 0
    for sample in samples:
        qa_pairs = list(sample.get("qa_pairs", []))
        if args.limit_questions > 0:
            qa_pairs = qa_pairs[: args.limit_questions]
        total += len(qa_pairs) * len(chains)
    progress = Progress(total, "Five-chain eval")

    base_runner = HfQwenVlGenerator(
        args.model_path,
        bf16=args.bf16,
        max_pixels=args.max_pixels,
    )
    lora_runner = None
    if any(chain in LORA_CHAINS for chain in chains):
        lora_runner = HfQwenVlGenerator(
            args.model_path,
            args.adapter_path,
            bf16=args.bf16,
            max_pixels=args.max_pixels,
        )

    with output_path.open(file_mode, encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for sample in samples:
            image_path = str(sample.get("image_path", "")).strip()
            qa_pairs = list(sample.get("qa_pairs", []))
            if args.limit_questions > 0:
                qa_pairs = qa_pairs[: args.limit_questions]

            image = None
            image_error = ""
            if not image_path or not Path(image_path).exists():
                image_error = "image_missing"
            else:
                try:
                    image = open_local_image(image_path)
                except Exception as exc:  # noqa: BLE001
                    image_error = f"image_open_error: {exc}"

            grounding_by_question: dict[int, dict[str, Any]] = {}

            for question_idx, qa in enumerate(qa_pairs, start=1):
                question = str(qa.get("question", "")).strip()
                gold_answer = str(qa.get("answer", "")).strip()
                answer_field = str(qa.get("answer_field", "")).strip()

                for chain in chains:
                    key = (chain, str(sample.get("image_id", "")), str(question_idx))
                    if key in existing_keys:
                        progress.advance()
                        continue

                    grounding = _empty_grounding()
                    prediction = ""
                    error = image_error
                    latency = 0.0
                    prompt = ""

                    try:
                        if not error and chain in RAG_CHAINS:
                            if question_idx not in grounding_by_question:
                                grounding_by_question[question_idx] = _retrieve_grounding(
                                    image,
                                    question,
                                    args.top_k,
                                )
                            grounding = grounding_by_question[question_idx]

                        if not error:
                            prompt, skip_reason = _make_prompt(chain, question, grounding)
                            error = skip_reason or ""

                        if not error:
                            runner = _get_runner(chain, base_runner, lora_runner)
                            started = time.perf_counter()
                            prediction = strip_model_artifacts(
                                runner.generate(
                                    prompt,
                                    None if chain == "retrieval_rag_text" else image_path,
                                    max_new_tokens=args.max_new_tokens,
                                )
                            )
                            latency = time.perf_counter() - started
                    except Exception as exc:  # noqa: BLE001
                        error = str(exc)
                        if args.stop_on_error:
                            raise

                    correct_bool, auto_score = score_answer(prediction, gold_answer) if prediction else (False, 0.0)
                    target_name = str(sample.get("artifact_name", "")).strip()
                    target_era = str(sample.get("era", "")).strip()
                    target_museum = str(sample.get("museum", "")).strip()
                    recognized_name = str(grounding.get("recognized_name", "")).strip()
                    recognized_name_correct = ""
                    if chain in RAG_CHAINS and recognized_name:
                        recognized_name_correct = int(normalize_text(recognized_name) == normalize_text(target_name))

                    writer.writerow(
                        {
                            "chain": chain,
                            "image_id": sample.get("image_id", ""),
                            "artifact_id": sample.get("artifact_id", ""),
                            "target_name": target_name,
                            "target_era": target_era,
                            "target_museum": target_museum,
                            "question_idx": question_idx,
                            "question": question,
                            "answer_field": answer_field,
                            "gold_answer": gold_answer,
                            "image_path": image_path,
                            "image_found": grounding.get("image_found", ""),
                            "image_confident": grounding.get("image_confident", ""),
                            "confidence_reason": grounding.get("confidence_reason", ""),
                            "recognized_name": recognized_name,
                            "recognized_era": grounding.get("recognized_era", ""),
                            "recognized_museum": grounding.get("recognized_museum", ""),
                            "recognized_score": grounding.get("recognized_score", ""),
                            "recognized_name_correct": recognized_name_correct,
                            "context_found": grounding.get("context_found", ""),
                            "retrieved_names": grounding.get("retrieved_names", ""),
                            "retrieved_scores": grounding.get("retrieved_scores", ""),
                            "prediction": prediction,
                            "auto_correct": int(correct_bool),
                            "auto_score": f"{auto_score:.4f}",
                            "target_name_mentioned": contains_expected(prediction, target_name),
                            "gold_answer_mentioned": contains_expected(prediction, gold_answer),
                            "latency_seconds": f"{latency:.4f}",
                            "error": error,
                        }
                    )
                    file.flush()
                    progress.advance()

    progress.close()
    _summarize(output_path, chains, summary_path)
    print(f"Results written to: {output_path}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
