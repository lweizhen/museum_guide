# src/AGENTS.md

Local guidance for agents editing the `src/` package.

The root `AGENTS.md` is the source of truth for setup, evaluation commands,
configuration, style, and project-level architecture. This file intentionally
stays small and only records source-package boundaries so future agents do not
have to reconcile competing manuals.

## Scope

This file applies only to Python modules inside `src/`.

Standard executable scripts live under `scripts/`:

```text
scripts/build/
scripts/eval/
scripts/judge/
scripts/data_tools/
```

The repository root keeps compatibility wrappers such as `build_index.py`,
`eval_scheme_b.py`, and `judge_guide_quality.py`. Prefer documenting and
maintaining the `scripts/` paths; keep wrappers thin.

The multimodal LoRA dataset is closed-set only. Images of the same artifact are
split across `train` / `test`; single-image artifacts duplicate the same image
into both splits. Keep this
closed-set split contract unless the root `AGENTS.md` and README are updated
first.

## Package Map

Core text RAG path:

```text
config -> kb -> embedder -> retriever -> prompt -> llm -> tts
```

Scheme A image-retrieval path:

```text
image_embedder -> image_index -> image_retriever -> text RAG modules
```

Progress/reporting helpers:

```text
progress -> eval scripts
```

## Public Interfaces To Preserve

Keep these function contracts stable unless the calling scripts are updated in
the same change:

- `src.retriever.retrieve(...)`
- `src.prompt.build_prompt(...)`
- `src.prompt.build_citation(...)`
- `src.llm.call_llm(...)`
- `src.llm.call_multimodal_llm(...)`
- `src.tts.text_to_mp3(...)`
- `src.image_retriever.retrieve_image(...)`
- `src.progress.progress_iter(...)`

Private helpers with a leading underscore may be refactored freely, but keep
their return shape stable if multiple modules depend on them.

## Module Responsibilities

- `config.py`: load `config.yaml`, environment variables, and defaults. New
  config keys must also be reflected in root `AGENTS.md` and README if they are
  user-facing.
- `kb.py`: load and split the text knowledge base from the configured data path.
- `embedder.py`: load the SentenceTransformer text embedding model lazily and
  return normalized `float32` vectors.
- `retriever.py`: combine exact artifact-name matching with FAISS vector
  retrieval; keep retrieval results transparent as `(doc, score)` tuples.
- `prompt.py`: build grounded guide prompts for text RAG and multimodal Scheme B
  modes. Do not weaken refusal and grounding constraints.
- `llm.py`: provide provider-neutral LLM calls for OpenAI-compatible,
  DashScope, and Ollama backends. Never hardcode API keys.
- `tts.py`: convert generated Chinese text to MP3 via Edge TTS.
- `image_embedder.py`, `image_index.py`, `image_retriever.py`: implement CLIP
  image embeddings, FAISS image index access, and Scheme A artifact candidate
  retrieval.
- `progress.py`: lightweight terminal progress display shared by offline eval
  scripts. Keep it dependency-free.

## Editing Rules

- Prefer relative imports inside `src`, for example `from .config import ...`.
- Keep module-level lazy singletons lazy; eager model/index loading can make CLI
  and Streamlit startup much heavier.
- Use UTF-8 for all text I/O.
- Keep user-facing errors actionable and in Chinese where the surrounding path is
  already Chinese.
- Do not write to `outputs/` or `index/` from inside reusable `src` helpers
  unless the helper already explicitly owns that responsibility.
- Preserve Scheme A and Scheme B as separate concepts:
  Scheme A is image retrieval plus text RAG.
  Scheme B is image-bearing multimodal LLM QA.
