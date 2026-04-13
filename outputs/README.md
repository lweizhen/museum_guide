# outputs directory

This directory stores generated runtime artifacts. Files are intentionally split
by workflow stage so full experiments, smoke tests, metrics, and audio outputs do
not overwrite or visually bury each other.

## Layout

- `raw/`: raw evaluation outputs produced directly by evaluation scripts.
- `judged/`: LLM-as-Judge post-processing outputs.
- `metrics/`: ROUGE-L, BLEU, and semantic-similarity post-processing outputs.
- `smoke/`: small test-run artifacts used to verify scripts before full runs.
- `media/`: generated TTS audio files.
- `archive/`: older one-off extracted text or legacy experiment artifacts.

## Main Files For Thesis Experiments

Scheme A QA:

- `raw/eval_scheme_a_qa_results.csv`
- `raw/eval_scheme_a_qa_summary.txt`
- `metrics/eval_scheme_a_qa_metrics.csv`
- `metrics/eval_scheme_a_qa_metrics_summary.txt`

Scheme B:

- `raw/eval_scheme_b_results.csv`
- `raw/eval_scheme_b_summary.txt`
- `judged/eval_scheme_b_judged_results.csv`
- `judged/eval_scheme_b_judged_summary.txt`
- `metrics/eval_scheme_b_metrics_by_mode.csv`
- `metrics/eval_scheme_b_metrics_by_mode_summary.txt`

## Current Default Writers

- `eval_rag.py` -> `raw/`
- `eval_scheme_a*.py` -> `raw/`
- `eval_scheme_b.py` -> `raw/`
- `judge_scheme_b_results.py` -> `judged/`
- `eval_metrics.py` -> `metrics/`
- TTS outputs -> `media/`

## Recommended Workflow

```bash
python eval_scheme_b.py --mode both --stop-on-error
python judge_scheme_b_results.py
python eval_metrics.py --input outputs/judged/eval_scheme_b_judged_results.csv --group-cols mode
```

For Scheme A QA:

```bash
python eval_scheme_a_qa.py --with-llm
python eval_metrics.py --input outputs/raw/eval_scheme_a_qa_results.csv
```

