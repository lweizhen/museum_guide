# AGENTS.md

Guidance for autonomous coding agents working in this repository.
This root file is the operational source of truth for build/test/style expectations.
Directory-level `AGENTS.md` files may add local details, but should not duplicate
or override this file unless they state a narrower rule explicitly.

## 1) Repository Snapshot

- Language: Python 3.x
- App types: CLI (`run_cli.py`), Streamlit UI (`app.py`), offline eval (`eval_rag.py`)
- Core architecture: text RAG with FAISS retrieval + LLM generation + TTS output
- Multimodal status:
  - Scheme A: image retrieval with CLIP embeddings + artifact candidate linking + text RAG QA
  - Scheme B: true image-bearing multimodal LLM QA branch for side-by-side comparison
  - Evaluation scripts now include terminal progress reporting for long full-run jobs
- Dependencies are managed via `requirements.txt` (no Poetry/Pipenv/Conda lock files here)
- Local rule hierarchy:
  - `AGENTS.md`: full-project rules and commands
  - `src/AGENTS.md`: focused notes for source-package modules only
- No CI config or task runner (no Makefile, no tox, no pytest config found)

## 2) Rule Files Discovery

Checked and **not present** in current repo:

- `.cursor/rules/`
- `.cursorrules`
- `.github/copilot-instructions.md`

If these files are added later, update this AGENTS.md and prioritize explicit repo rules.

## 3) Setup and Build Commands

Use these commands from repository root.

### Install

```bash
python -m pip install -r requirements.txt
```

### Build combined text knowledge base

```bash
python prepare_combined_kb.py
```

Expected artifact:

- `data/exhibits_combined.txt`

### Build vector index (required before retrieval flows)

```bash
python build_index.py
```

Expected artifact:

- `index/exhibits.index`

### Build image index (required before image retrieval flows)

```bash
python build_image_index.py
```

Expected artifacts:

- `index/exhibits_images.index`
- `index/exhibits_images_meta.json`

### Run CLI app

```bash
python run_cli.py
```

### Run Streamlit app

```bash
streamlit run app.py
```

### Run evaluation pipeline

```bash
python eval_rag.py
```

Expected artifacts:

- `outputs/raw/eval_results.csv`
- `outputs/raw/eval_summary.txt`

### Run Scheme A multimodal evaluation

```bash
python eval_scheme_a.py
```

Optional:

```bash
python eval_scheme_a.py --limit 20
python eval_scheme_a.py --with-llm
```

Expected artifacts:

- `outputs/raw/eval_scheme_a_results.csv`
- `outputs/raw/eval_scheme_a_summary.txt`

### Run Scheme A multimodal QA evaluation

```bash
python eval_scheme_a_qa.py
```

Optional:

```bash
python eval_scheme_a_qa.py --limit-images 20
python eval_scheme_a_qa.py --limit-images 20 --limit-questions 3
python eval_scheme_a_qa.py --with-llm
```

Expected artifacts:

- `outputs/raw/eval_scheme_a_qa_results.csv`
- `outputs/raw/eval_scheme_a_qa_summary.txt`
- `outputs/raw/eval_scheme_a_qa_breakdown.json`

### Run Scheme A caption-generation evaluation

```bash
python eval_scheme_a_caption.py
```

Optional:

```bash
python eval_scheme_a_caption.py --limit-images 20
python eval_scheme_a_caption.py --with-llm
```

Expected artifacts:

- `outputs/raw/eval_scheme_a_caption_results.csv`
- `outputs/raw/eval_scheme_a_caption_summary.txt`
- `outputs/raw/eval_scheme_a_caption_breakdown.json`

### Run Scheme A cross-image retrieval evaluation

```bash
python eval_scheme_a_cross_image.py
```

Optional:

```bash
python eval_scheme_a_cross_image.py --limit-artifacts 20
```

Expected artifacts:

- `outputs/raw/eval_scheme_a_cross_image_results.csv`
- `outputs/raw/eval_scheme_a_cross_image_summary.txt`
- `outputs/raw/eval_scheme_a_cross_image_breakdown.json`

### Run Scheme B multimodal QA evaluation

```bash
python eval_scheme_b.py
```

Optional:

```bash
python eval_scheme_b.py --mode direct --limit-images 5 --limit-questions 2
python eval_scheme_b.py --mode grounded --limit-images 5 --limit-questions 2
python eval_scheme_b.py --mode both --limit-images 5 --limit-questions 2 --max-calls 10
python eval_scheme_b.py --mode grounded --limit-images 2 --limit-questions 1 --dry-run
```

Expected artifacts:

- `outputs/raw/eval_scheme_b_results.csv`
- `outputs/raw/eval_scheme_b_summary.txt`
- `outputs/raw/eval_scheme_b_breakdown.json`

### Run Scheme B semantic judge as a separate post-processing step

Recommended for slow multimodal full runs:

1. First generate Scheme B answers without `--judge-llm`.
2. Then run the judge script on the saved CSV.

```bash
python eval_scheme_b.py --mode both --stop-on-error
python judge_scheme_b_results.py --input outputs/raw/eval_scheme_b_results.csv
```

Optional:

```bash
python judge_scheme_b_results.py --mode grounded --limit 200
python judge_scheme_b_results.py --input outputs/raw/eval_scheme_b_results.csv --output outputs/judged/eval_scheme_b_judged_results.csv --overwrite
```

Expected artifacts:

- `outputs/judged/eval_scheme_b_judged_results.csv`
- `outputs/judged/eval_scheme_b_judged_summary.txt`
- `outputs/judged/eval_scheme_b_judged_breakdown.json`

### Run guide-style quality judge as a separate post-processing step

Use this after Scheme A or Scheme B answer generation to evaluate whether the
generated text sounds like a qualified museum guide explanation. It scores
factuality, groundedness, guide style, clarity, completeness, fluency,
engagement, and overall quality.

```bash
python judge_guide_quality.py --input outputs/raw/eval_scheme_a_qa_results.csv
python judge_guide_quality.py --input outputs/judged/eval_scheme_b_judged_results.csv --group-cols mode
```

Optional:

```bash
python judge_guide_quality.py --input outputs/raw/eval_scheme_a_qa_results.csv --dry-run
python judge_guide_quality.py --input outputs/judged/eval_scheme_b_judged_results.csv --limit 200 --overwrite
```

Expected artifacts:

- `outputs/judged/<input_stem>_guide_quality.csv`
- `outputs/judged/<input_stem>_guide_quality_summary.txt`
- `outputs/judged/<input_stem>_guide_quality_breakdown.json`

### Run text-generation metrics for evaluated answers

Use this after Scheme A/Scheme B answer generation, especially after Scheme B
answers have been judged separately. It computes ROUGE-L, BLEU-1/2/4, and
semantic similarity against the reference answer/description.

```bash
python eval_metrics.py --input outputs/judged/eval_scheme_b_judged_results.csv --group-cols mode
```

Optional:

```bash
python eval_metrics.py --input outputs/raw/eval_scheme_a_qa_results.csv
python eval_metrics.py --input outputs/judged/eval_scheme_b_judged_results.csv --mode grounded
python eval_metrics.py --input outputs/judged/eval_scheme_b_judged_results.csv --skip-embedding
```

Expected artifacts:

- `outputs/metrics/eval_scheme_b_judged_results_metrics.csv`
- `outputs/metrics/eval_scheme_b_judged_results_metrics_summary.txt`
- `outputs/metrics/eval_scheme_b_judged_results_metrics_breakdown.json`

### Build unified multimodal evaluation dataset

```bash
python prepare_multimodal_eval_dataset.py
```

Optional:

```bash
python prepare_multimodal_eval_dataset.py --limit-artifacts 20
```

Expected artifacts:

- `data/multimodal_eval/artifacts.jsonl`
- `data/multimodal_eval/train.jsonl`
- `data/multimodal_eval/val.jsonl`
- `data/multimodal_eval/test.jsonl`
- `data/multimodal_eval/train_images.jsonl`
- `data/multimodal_eval/val_images.jsonl`
- `data/multimodal_eval/test_images.jsonl`
- `data/multimodal_eval/summary.json`

## 4) Lint / Format Commands

There is currently **no enforced linter/formatter config** in repo (no `pyproject.toml`, `setup.cfg`, `.flake8`, etc.).

When making changes, use these safe local checks if tools are available:

```bash
python -m compileall src app.py run_cli.py build_index.py build_image_index.py eval_rag.py eval_scheme_a.py eval_scheme_a_caption.py eval_scheme_a_cross_image.py eval_scheme_a_qa.py eval_scheme_b.py judge_scheme_b_results.py judge_guide_quality.py eval_metrics.py prepare_combined_kb.py prepare_multimodal_eval_dataset.py
```

Optional (only if installed in your environment):

```bash
python -m ruff check .
python -m black --check .
```

If you introduce a formatter/linter config in a PR, document exact commands here.

## 5) Test Commands (Including Single-Test Guidance)

Current state:

- No `tests/` directory and no unit test framework config committed yet.
- Primary validation is smoke execution of scripts.

### Existing smoke-test commands

```bash
python prepare_combined_kb.py
python build_index.py
python build_image_index.py --limit 10
python eval_rag.py
python eval_scheme_a.py --limit 10
python eval_scheme_b.py --mode grounded --limit-images 2 --limit-questions 1 --dry-run
python run_cli.py
```

### Single-case evaluation tip (without unit tests)

To quickly validate one scenario, temporarily reduce `data/test_questions.jsonl` to one case
or create a one-line temp file and point eval code to it before running `python eval_rag.py`.

### If pytest tests are added (recommended pattern)

Run all tests:

```bash
python -m pytest -q
```

Run a single file:

```bash
python -m pytest tests/test_retriever.py -q
```

Run a single test function:

```bash
python -m pytest tests/test_retriever.py::test_dynamic_threshold -q
```

Run tests matching keyword:

```bash
python -m pytest -k retriever -q
```

## 6) Environment and Runtime Conventions

### Configuration loading priority

`src/config.py` reads settings in this order (first non-empty wins):

1. **`config.yaml`** in project root (recommended for local development)
2. **Environment variables** (for CI / deployment overrides)
3. **Code defaults** in `config.py`

`config.yaml` requires `pyyaml`; if the package is missing or the file is absent, the system falls back silently to env vars + defaults.

### Key configuration items

- `LLM_PROVIDER` / `llm_provider`: `openai`, `dashscope`, or `ollama`
- OpenAI-compatible: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_TEMPERATURE` (yaml: `openai.*`)
- DashScope text generation: `DASHSCOPE_API_KEY`, `QWEN_MODEL`, `TEMPERATURE` (yaml: `dashscope.*`)
- DashScope multimodal generation: `QWEN_MULTIMODAL_MODEL` (yaml: `dashscope.multimodal_model`)
- Ollama text generation: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_TEMPERATURE` (yaml: `ollama.*`)
- Ollama multimodal generation: `OLLAMA_MULTIMODAL_MODEL` (yaml: `ollama.multimodal_model`)
- Ollama timeout: `OLLAMA_TIMEOUT_SECONDS` (yaml: `ollama.timeout_seconds`)
- Judge LLM: `JUDGE_PROVIDER`, `JUDGE_MODEL`, `JUDGE_BASE_URL`, `JUDGE_API_KEY`, `JUDGE_TEMPERATURE`, `JUDGE_TIMEOUT_SECONDS` (yaml: `judge.*`)
- Embedding / retrieval: `EMBED_MODEL_NAME`, `DATA_PATH`, `INDEX_PATH`, `TOP_K`, `THRESHOLD`, `MARGIN`
- Default text KB path should point to `data/exhibits_combined.txt`, generated by `prepare_combined_kb.py`.
- Image retrieval: `IMAGE_CSV_PATH`, `IMAGE_CACHE_DIR`, `IMAGE_INDEX_PATH`, `IMAGE_META_PATH`, `IMAGE_MODEL_NAME`, `IMAGE_TOP_K`, `IMAGE_MIN_SCORE`, `IMAGE_MIN_GAP`, `IMAGE_MAX_IMAGES_PER_ITEM`
- TTS: `VOICE`, `OUTPUT_DIR` (default: `outputs/media`)

Agent behavior requirements:

- Never hardcode API keys or secrets.
- Read config via `src/config.py` helpers (`_get`, `get_api_key`, `get_openai_api_key`).
- Preserve current default values unless task explicitly changes behavior.
- When adding new config items, update `config.yaml`, `src/config.py`, and this section.

## 7) Code Style Guidelines

Follow existing style in `src/` and top-level scripts.

### Imports

- Use absolute imports for external packages, relative imports within `src` (e.g., `from .config import ...`).
- Keep import groups ordered: stdlib, third-party, local.
- Remove unused imports; do not leave dead aliases.

### Formatting

- Follow PEP 8 and keep lines readable (target ~88-100 chars where practical).
- Use 4-space indentation, no tabs.
- Keep functions focused and short when possible.
- Prefer explicit intermediate variables over dense one-liners in retrieval/LLM logic.

### Types

- Add type hints for new/changed function signatures.
- Use built-in generics (`list[str]`, `tuple[str, float]`) consistently with existing code.
- Prefer precise return types over `Any` unless unavoidable.

### Naming

- Functions/variables: `snake_case`.
- Constants: `UPPER_SNAKE_CASE` in config-style modules.
- Internal helpers: prefix with `_` (e.g., `_call_ollama`, `_extract_name`).
- Use meaningful names tied to RAG domain (`contexts`, `retrieved_docs`, `top_k`).

### Error handling

- Raise `RuntimeError`/`FileNotFoundError` with actionable messages when inputs or env are invalid.
- Keep user-facing failures clear and recovery-oriented (what to check next).
- Do not swallow exceptions silently.
- Preserve existing Chinese-language user messages in end-user paths unless asked to refactor language.

### I/O and side effects

- Use UTF-8 for reading/writing text files.
- Create directories with `exist_ok=True` before file writes.
- Keep output artifact paths consistent (`outputs/raw`, `outputs/judged`, `outputs/metrics`, `outputs/media`, `index/`).

### LLM/RAG-specific conventions

- Keep retrieval deterministic and transparent (return docs + scores).
- Keep prompt constraints explicit: grounded answer, refusal when context insufficient.
- Avoid introducing hallucination-prone prompt wording.
- Preserve citation behavior where available (`build_citation`).

## 8) Modification Guardrails for Agents

- Do not commit generated binaries or cache artifacts (`__pycache__`, large media outputs) unless task explicitly asks.
- Do not commit generated eval/index artifacts from `outputs/` or `index/`; these are local runtime artifacts.
- Avoid breaking public script entrypoints: `build_index.py`, `run_cli.py`, `app.py`, `eval_rag.py`.
- If changing config keys/defaults, update both code and this AGENTS.md.
- Prefer minimal, localized edits over broad refactors.
- Keep backward compatibility for both DashScope and Ollama code paths.
- If the user has a long evaluation running, avoid touching `outputs/` and avoid heavy concurrent GPU/CPU jobs unless needed.

## 9) Pre-Completion Checklist

Before finishing a code change, an agent should:

1. Re-run the relevant command(s) from sections 3-5.
2. Confirm index/eval/output paths still resolve correctly.
3. Check imports and type hints for edited functions.
4. Verify error messages are actionable.
5. Summarize exactly what changed and how it was validated.

## 10) Multimodal Architecture Roadmap

This section records the intended system structure for the graduation project so later work
stays aligned with the thesis plan and current implementation.

### Current baseline: text RAG guide

Flow:

`user question -> text retriever -> prompt builder -> LLM -> TTS`

Primary modules:

- `src/retriever.py`
- `src/prompt.py`
- `src/llm.py`
- `src/tts.py`
- `app.py`
- `run_cli.py`

Role in project:

- Stable baseline for museum guide QA
- Main reference path for grounded answering and refusal behavior

### Scheme A: pipeline-based multimodal QA (implemented direction)

Flow:

`uploaded image -> image embedding / FAISS search -> candidate artifact -> linked text query -> text KB retrieval -> LLM answer -> TTS`

Primary modules:

- `build_image_index.py`
- `src/image_embedder.py`
- `src/image_index.py`
- `src/image_retriever.py`
- `app.py`

Current repo status:

- Image indexing is available
- Image upload entry is available in Streamlit
- Low-confidence rejection is implemented with `min_score` and `min_gap`
- Final answer generation still depends on text knowledge retrieval
- `eval_scheme_a_cross_image.py` tests same-artifact different-image retrieval generalization

Strengths:

- Stronger controllability and explainability
- Easier to trace mistakes to retrieval stage
- Better fit for museum scenarios that prioritize factual grounding

Risks:

- If image retrieval picks the wrong artifact, downstream QA also drifts
- Performance depends on image coverage and image quality

### Scheme B: end-to-end multimodal QA (experimental branch)

Target flow:

`uploaded image + user question -> multimodal LLM -> answer -> optional TTS`

Recommended grounded variant:

`uploaded image + user question -> multimodal LLM extracts clues or draft recognition -> retrieve text KB -> multimodal LLM answers with image + retrieved context -> optional TTS`

Current repo status:

- `src/llm.py` now exposes a true image-bearing `call_multimodal_llm(...)` path
- `src/prompt.py` now includes dedicated multimodal prompts for direct and grounded modes
- `app.py` now exposes Scheme B as a selectable image QA experiment path
- `eval_scheme_b.py` provides direct, grounded, and side-by-side offline evaluation

Expected strengths:

- More natural interaction
- Potentially better handling of open-ended visual questions

Expected risks:

- More hallucination risk if not grounded by retrieval
- Harder refusal calibration
- Harder to explain failures in thesis and demo

### Recommended product and experiment strategy

Keep both branches instead of replacing one with the other.

- Default demo path: Scheme A
- Experimental comparison path: Scheme B
- Thesis conclusion should compare reliability, flexibility, and latency instead of assuming one path is universally better

### Planned comparison metrics

- Top-1 artifact recognition accuracy
- Top-3 artifact hit rate
- Refusal accuracy on low-quality or out-of-knowledge-base images
- Answer correctness on artifact facts
- Groundedness / whether answer stays within retrievable evidence
- End-to-end response time
- User experience clarity in Streamlit demo

### Planned generation-quality evaluation upgrade

Current rule-based QA scoring is useful for closed factual fields, but it is too
strict for Chinese explanatory answers. Future work should keep traditional
metrics as reproducible references while adding stronger semantic judging:

- Keep ROUGE-L and Coverage for text overlap / evidence coverage.
- Optionally report BLEU, METEOR, CIDEr, or SPICE as legacy comparison metrics,
  but do not use them as the only conclusion for Chinese museum-guide answers.
- Keep CLIPScore for image-text relevance in caption-style tasks.
- Add an LLM-as-Judge evaluator for generated descriptions and open QA answers,
  scoring factuality, groundedness, completeness, fluency, hallucination, and
  overall quality against GT descriptions and retrieved evidence.
- Prefer a judge model that is different from the generation model, and manually
  audit a sample of judged cases for thesis credibility.

### Shared multimodal evaluation dataset

To support both caption-generation-style evaluation and visual QA evaluation, the repo now
uses a shared artifact-level dataset generated by `prepare_multimodal_eval_dataset.py`.

Dataset design principles:

- Split by artifact instead of by image to avoid leakage across train/val/test.
- Reuse current text KB as the source of `reference_description` and `qa_pairs`.
- Keep both artifact-level JSONL files and image-level expanded JSONL files.
- Preserve `detail_url`, `source_urls`, and `image_url` so later manual auditing remains possible.

Recommended usage:

- Caption generation:
  compare model outputs against `reference_description`
- Visual QA:
  run questions from `qa_pairs` against the model and score answer correctness

### Implementation priority

1. Continue improving the text knowledge base quality because both branches depend on it.
2. Stabilize Scheme A evaluation, including image recognition accuracy and refusal cases.
3. Use `eval_scheme_b.py` to compare Scheme B direct and grounded modes against Scheme A.
4. Run side-by-side experiments and record representative success/failure cases for the thesis.
5. Only explore lightweight local fine-tuning or LoRA after the two-branch comparison is runnable.

### Change management note

When implementing future multimodal features, preserve the distinction between:

- image retrieval + text RAG
- true end-to-end multimodal LLM QA

Do not describe Scheme A as end-to-end multimodal LLM QA in docs, code comments, or thesis text.
