# AGENTS.md

Guidance for autonomous coding agents working in this repository.
This file is the operational source of truth for build/test/style expectations.

## 1) Repository Snapshot

- Language: Python 3.x
- App types: CLI (`run_cli.py`), Streamlit UI (`app.py`), offline eval (`eval_rag.py`)
- Core architecture: RAG with FAISS retrieval + LLM generation + TTS output
- Dependencies are managed via `requirements.txt` (no Poetry/Pipenv/Conda lock files here)
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

### Build vector index (required before retrieval flows)

```bash
python build_index.py
```

Expected artifact:

- `index/exhibits.index`

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

- `outputs/eval_results.csv`
- `outputs/eval_summary.txt`

## 4) Lint / Format Commands

There is currently **no enforced linter/formatter config** in repo (no `pyproject.toml`, `setup.cfg`, `.flake8`, etc.).

When making changes, use these safe local checks if tools are available:

```bash
python -m compileall src app.py run_cli.py build_index.py eval_rag.py
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
python build_index.py
python eval_rag.py
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
- DashScope: `DASHSCOPE_API_KEY`, `QWEN_MODEL`, `TEMPERATURE` (yaml: `dashscope.*`)
- Ollama: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_TEMPERATURE` (yaml: `ollama.*`)
- Embedding / retrieval: `EMBED_MODEL_NAME`, `DATA_PATH`, `INDEX_PATH`, `TOP_K`, `THRESHOLD`, `MARGIN`
- TTS: `VOICE`, `OUTPUT_DIR`

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
- Keep output artifact paths consistent (`outputs/`, `index/`).

### LLM/RAG-specific conventions

- Keep retrieval deterministic and transparent (return docs + scores).
- Keep prompt constraints explicit: grounded answer, refusal when context insufficient.
- Avoid introducing hallucination-prone prompt wording.
- Preserve citation behavior where available (`build_citation`).

## 8) Modification Guardrails for Agents

- Do not commit generated binaries or cache artifacts (`__pycache__`, large media outputs) unless task explicitly asks.
- Avoid breaking public script entrypoints: `build_index.py`, `run_cli.py`, `app.py`, `eval_rag.py`.
- If changing config keys/defaults, update both code and this AGENTS.md.
- Prefer minimal, localized edits over broad refactors.
- Keep backward compatibility for both DashScope and Ollama code paths.

## 9) Pre-Completion Checklist

Before finishing a code change, an agent should:

1. Re-run the relevant command(s) from sections 3-5.
2. Confirm index/eval/output paths still resolve correctly.
3. Check imports and type hints for edited functions.
4. Verify error messages are actionable.
5. Summarize exactly what changed and how it was validated.
