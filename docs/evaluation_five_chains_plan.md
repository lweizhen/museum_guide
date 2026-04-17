# Unified Five-Chain Evaluation Plan

This document records the next evaluation refactor. The project currently has separate evaluators for Scheme A, Scheme B direct, Scheme B grounded, and Qwen2.5-VL + LoRA. These evaluators do not all use the same task granularity. The refactor goal is to use one shared image-QA test set, one base multimodal model family, one result schema, and one metric pipeline.

## 1. Main Experiment Dataset

The main horizontal comparison should use:

- `data/multimodal_eval/test_images.jsonl`

Reason:

- One row represents one test image.
- Each row contains `qa_pairs`.
- The task format is consistently `image + question -> answer`.
- All five chains can support this format.

`data/multimodal_eval/test_lora.jsonl` should be kept as a supplementary LoRA analysis set. It is useful for checking instruction-level behavior such as identification, captioning, guide style, and QA, but it should not be mixed directly with the main five-chain table.

## 2. Five Chain Names

| Chain | Previous name | Image enters VLM | Uses RAG | Uses LoRA | Meaning |
| --- | --- | --- | --- | --- | --- |
| `retrieval_rag_text` | Scheme A | No | Yes | No | Image retrieval identifies the artifact, then text RAG answers the question. |
| `vl_direct` | Scheme B direct | Yes | No | No | Qwen2.5-VL answers directly from image and question. |
| `vl_rag` | Scheme B grounded | Yes | Yes | No | Qwen2.5-VL answers with retrieved exhibit context. |
| `vl_lora` | LoRA direct | Yes | No | Yes | Qwen2.5-VL + LoRA answers directly. |
| `vl_rag_lora` | New chain | Yes | Yes | Yes | Qwen2.5-VL + LoRA answers with retrieved exhibit context. |

## 3. Implemented Entry Point

The unified evaluator is:

```bash
python scripts/eval/eval_multimodal_chains.py \
  --chains retrieval_rag_text,vl_direct,vl_rag,vl_lora,vl_rag_lora \
  --model-path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --adapter-path outputs/lora/qwen25vl3b_museum
```

Default output:

- `outputs/raw/eval_multimodal_chains_results.csv`
- `outputs/raw/eval_multimodal_chains_summary.txt`

Compatibility wrapper:

```bash
python eval_multimodal_chains.py --chains vl_direct --model-path /path/to/Qwen2.5-VL-3B-Instruct
```

## 4. Recommended Run Strategy

Do not run all five chains at once by default. Run smoke tests first:

```bash
python scripts/eval/eval_multimodal_chains.py \
  --chains vl_direct,vl_rag \
  --model-path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --limit-images 2 \
  --limit-questions 1
```

LoRA smoke test:

```bash
python scripts/eval/eval_multimodal_chains.py \
  --chains vl_lora,vl_rag_lora \
  --model-path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --adapter-path outputs/lora/qwen25vl3b_museum \
  --limit-images 2 \
  --limit-questions 1
```

For long runs, use `--resume`:

```bash
python scripts/eval/eval_multimodal_chains.py \
  --chains vl_rag_lora \
  --model-path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --adapter-path outputs/lora/qwen25vl3b_museum \
  --resume
```

For multiple GPUs, split work by chain or by dataset shard. Each process must write a different output file:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_multimodal_chains.py \
  --chains vl_rag \
  --model-path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --num-shards 2 \
  --shard-index 0 \
  --output outputs/raw/eval_multimodal_chains_vl_rag_s0.csv \
  --summary outputs/raw/eval_multimodal_chains_vl_rag_s0_summary.txt

CUDA_VISIBLE_DEVICES=1 python scripts/eval/eval_multimodal_chains.py \
  --chains vl_rag \
  --model-path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --num-shards 2 \
  --shard-index 1 \
  --output outputs/raw/eval_multimodal_chains_vl_rag_s1.csv \
  --summary outputs/raw/eval_multimodal_chains_vl_rag_s1_summary.txt
```

After all shard files are copied back, concatenate CSV files with the same header before running metrics.

Merge shard files:

```bash
python scripts/eval/merge_eval_csv.py \
  "outputs/raw/eval_multimodal_chains_*_s*.csv" \
  --output outputs/raw/eval_multimodal_chains_results.csv
```

## 5. Output Schema

The unified result CSV uses `chain` as the grouping field and includes:

```text
chain
image_id
artifact_id
target_name
target_era
target_museum
question_idx
question
answer_field
gold_answer
image_path
image_found
image_confident
confidence_reason
recognized_name
recognized_era
recognized_museum
recognized_score
recognized_name_correct
context_found
retrieved_names
retrieved_scores
prediction
auto_correct
auto_score
target_name_mentioned
gold_answer_mentioned
latency_seconds
error
```

## 6. Post-Evaluation Metrics

Unified output can be evaluated with existing metric and guide-quality scripts:

```bash
python scripts/eval/eval_metrics.py \
  --input outputs/raw/eval_multimodal_chains_results.csv \
  --prediction-col prediction \
  --reference-col gold_answer \
  --group-cols chain

python scripts/judge/judge_guide_quality.py \
  --input outputs/raw/eval_multimodal_chains_results.csv \
  --answer-col prediction \
  --reference-col gold_answer \
  --group-cols chain
```

## 7. Evaluation Priorities

P0:

1. Keep `scripts/eval/eval_multimodal_chains.py` runnable for all five chains.
2. Keep `src/hf_qwen_vl.py` as the shared Hugging Face Qwen2.5-VL inference layer.
3. Keep output schema stable.
4. Use `--resume` for remote long-running evaluations.

P1:

1. Connect unified output to `eval_metrics.py`.
2. Connect unified output to semantic judge and guide-quality judge.
3. Generate paper-ready summary tables grouped by `chain`.

P2:

1. Add artificial or low-quality image refusal tests.
2. Add manual sampling to validate judge reliability.
3. Add resource usage and latency statistics.
