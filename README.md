# 基于多模态大模型的文物语音导览助手

本项目面向博物馆文物导览场景，提供两条技术路线：

- 方案 A：图片检索增强 + 文本 RAG + 语音播报
- 方案 B：多模态大模型直答 / grounding 后回答 + 语音播报

项目同时包含：

- Streamlit 网页演示
- 命令行问答入口
- 文本与图像索引构建
- 方案 A / 方案 B 离线评测
- 语义裁判、讲解质量裁判、文本生成指标统计

## 1. 项目结构

当前项目已经按职责重构。标准脚本入口集中在 `scripts/` 下，仓库根目录保留了同名兼容包装脚本，因此旧命令仍然可用。

```text
museum_guide/
  app.py
  run_cli.py
  src/
  scripts/
    build/          # 构建知识库、文本索引、图片索引、评测数据集
    eval/           # 各类离线评测
    judge/          # 语义裁判、讲解质量裁判
    data_tools/     # 数据清洗与外部数据处理工具
  data/
  index/
  outputs/
    raw/            # 原始评测输出
    judged/         # 裁判结果
    metrics/        # ROUGE / BLEU / 语义相似度等
    media/          # TTS 音频
    archive/        # 历史材料
```

## 2. 运行环境

### 2.1 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 2.2 模型与配置

配置优先级如下：

1. 根目录 `config.yaml`
2. 环境变量
3. `src/config.py` 默认值

常用配置项：

- `llm_provider`
- `ollama.model`
- `ollama.multimodal_model`
- `dashscope.model`
- `dashscope.multimodal_model`
- `judge.*`
- `data_path`
- `index_path`
- `image_*`
- `output_dir`

建议不要把真实 API Key 提交到 Git。

Windows PowerShell 设置环境变量：

```powershell
$env:DASHSCOPE_API_KEY="your_api_key"
```

Linux / AutoDL：

```bash
export DASHSCOPE_API_KEY="your_api_key"
```

## 3. 构建流程

### 3.1 生成统一文本知识库

标准入口：

```bash
python scripts/build/prepare_combined_kb.py
```

兼容旧命令：

```bash
python prepare_combined_kb.py
```

输出：

- `data/exhibits_combined.txt`

### 3.2 构建文本向量索引

```bash
python scripts/build/build_index.py
```

兼容旧命令：

```bash
python build_index.py
```

输出：

- `index/exhibits.index`

### 3.3 构建图片索引

```bash
python scripts/build/build_image_index.py
```

兼容旧命令：

```bash
python build_image_index.py
```

输出：

- `index/exhibits_images.index`
- `index/exhibits_images_meta.json`

### 3.4 构建闭集 LoRA / 多模态评测数据集

该数据集只用于“同一文物不同图片”的闭集微调与闭集测试。多图文物保证 `train` 和 `test` 都有图片，额外图片优先划入 `train`；只有一张图片的文物会将同一张图同时写入 `train` 和 `test`。

`train_lora.jsonl` / `test_lora.jsonl` 面向 `vl_rag_lora` 链路进行构造：除识别样本外，讲解、描述和问答样本都会在 instruction 中加入同一文物的参考资料上下文，训练目标是让 LoRA 学会“看图 + 结合 RAG 上下文 + 按导览口吻回答”。因此，新 adapter 更适合用于 `vl_rag_lora`，不一定会让裸 `vl_lora` 直接问答同步变好。

```bash
python scripts/build/prepare_multimodal_eval_dataset.py
```

兼容命令：

```bash
python prepare_multimodal_eval_dataset.py
```

输出：

- `data/multimodal_eval/artifacts.jsonl`
- `data/multimodal_eval/train_images.jsonl`
- `data/multimodal_eval/val_images.jsonl`
- `data/multimodal_eval/test_images.jsonl`
- `data/multimodal_eval/train_lora.jsonl`
- `data/multimodal_eval/val_lora.jsonl`
- `data/multimodal_eval/test_lora.jsonl`
- `data/multimodal_eval/summary.json`


### 3.5 Qwen2.5-VL LoRA 微调

该部分面向 AutoDL 单卡 4090 的闭集微调实验。微调使用 Hugging Face 格式的 Qwen2.5-VL-3B-Instruct，不使用 Ollama 模型文件。

将 LoRA 样本转成 Qwen2.5-VL messages 格式：

```bash
python scripts/finetune/prepare_qwen25vl_lora_data.py --input data/multimodal_eval/train_lora.jsonl --output data/multimodal_eval/train_qwen25vl_messages.jsonl
```

训练 QLoRA adapter：

```bash
python scripts/finetune/train_qwen25vl_lora.py --model-path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct --train-file data/multimodal_eval/train_lora.jsonl --output-dir outputs/lora/qwen25vl3b_museum --epochs 2 --batch-size 1 --gradient-accumulation-steps 8
```

评测微调后模型：

```bash
python scripts/finetune/eval_qwen25vl_lora.py --model-path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct --adapter-path outputs/lora/qwen25vl3b_museum --test-file data/multimodal_eval/test_lora.jsonl
```

输出：

- `outputs/lora/qwen25vl3b_museum/`
- `outputs/finetune/eval_qwen25vl_lora_results.csv`
- `outputs/finetune/eval_qwen25vl_lora_summary.txt`
## 4. 交互入口

### 4.1 Streamlit 页面

```bash
streamlit run app.py
```

### 4.2 命令行入口

```bash
python run_cli.py
```

## 5. 评测命令

## 5.1 文本 RAG 基线评测

```bash
python scripts/eval/eval_rag.py
```

兼容旧命令：

```bash
python eval_rag.py
```

输出：

- `outputs/raw/eval_results.csv`
- `outputs/raw/eval_summary.txt`

## 5.2 方案 A 评测

### 方案 A 图像问答评测

```bash
python scripts/eval/eval_scheme_a_qa.py --with-llm
```

兼容旧命令：

```bash
python eval_scheme_a_qa.py --with-llm
```

### 方案 A 图像描述评测

```bash
python scripts/eval/eval_scheme_a_caption.py --with-llm
```

### 方案 A 跨图检索评测

```bash
python scripts/eval/eval_scheme_a_cross_image.py
```

### 小样本调试

```bash
python scripts/eval/eval_scheme_a_qa.py --limit-images 20 --limit-questions 3 --with-llm
```

## 5.3 方案 B 评测

### 小样本烟雾测试

```bash
python scripts/eval/eval_scheme_b.py --mode grounded --limit-images 3 --limit-questions 1 --stop-on-error
```

### 跑完整答案生成

```bash
python scripts/eval/eval_scheme_b.py --mode both --stop-on-error
```

如需只跑一部分：

```bash
python scripts/eval/eval_scheme_b.py --mode both --limit-images 200 --stop-on-error
```

输出：

- `outputs/raw/eval_scheme_b_results.csv`
- `outputs/raw/eval_scheme_b_summary.txt`
- `outputs/raw/eval_scheme_b_breakdown.json`

## 5.4 方案 B 语义裁判

建议先跑答案生成，再单独跑裁判：

```bash
python scripts/judge/judge_scheme_b_results.py --input outputs/raw/eval_scheme_b_results.csv
```

兼容旧命令：

```bash
python judge_scheme_b_results.py --input outputs/raw/eval_scheme_b_results.csv
```

输出：

- `outputs/judged/eval_scheme_b_judged_results.csv`
- `outputs/judged/eval_scheme_b_judged_summary.txt`
- `outputs/judged/eval_scheme_b_judged_breakdown.json`

## 5.5 讲解质量裁判

### 方案 A

```bash
python scripts/judge/judge_guide_quality.py --input outputs/raw/eval_scheme_a_qa_results.csv
```

### 方案 B

```bash
python scripts/judge/judge_guide_quality.py --input outputs/judged/eval_scheme_b_judged_results.csv --group-cols mode
```

输出：

- `outputs/judged/<input_stem>_guide_quality.csv`
- `outputs/judged/<input_stem>_guide_quality_summary.txt`
- `outputs/judged/<input_stem>_guide_quality_breakdown.json`

## 5.6 文本生成指标

```bash
python scripts/eval/eval_metrics.py --input outputs/judged/eval_scheme_b_judged_results.csv --group-cols mode
```

兼容旧命令：

```bash
python eval_metrics.py --input outputs/judged/eval_scheme_b_judged_results.csv --group-cols mode
```

输出：

- `outputs/metrics/<input_stem>_metrics.csv`
- `outputs/metrics/<input_stem>_metrics_summary.txt`
- `outputs/metrics/<input_stem>_metrics_breakdown.json`

## 6. AutoDL / 远程 GPU 运行

## 6.1 首次进入实例

进入项目目录后建议执行：

```bash
cd /root/autodl-tmp/museum_guide
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export HF_HOME=/root/autodl-tmp/hf_cache
```

如果环境支持网络加速：

```bash
source /etc/network_turbo
```

## 6.2 每次重新连接 SSH

如果实例磁盘和虚拟环境还在，通常只需要：

```bash
cd /root/autodl-tmp/museum_guide
source .venv/bin/activate
export HF_HOME=/root/autodl-tmp/hf_cache
```

如果 `.venv` 不存在，再重新创建并安装依赖。

## 6.3 拉取或更新代码

首次：

```bash
git clone <your_repo_url>
cd museum_guide
```

更新：

```bash
git pull origin main
```

如果 HTTPS 拉取不稳定，可以改用压缩包上传或 SSH key。

## 6.4 结果拉回本地

示例：

```powershell
scp -P <port> root@<host>:/root/autodl-tmp/museum_guide/outputs/raw/eval_scheme_a_qa_results.csv D:\STUDY\graduation_design\museum_guide\outputs\raw\
```

拉整个 `outputs` 前，先确认本地目标目录和远程结果不会误覆盖。

## 7. 当前技术路线说明

### 方案 A

流程：

```text
图片 -> CLIP 编码 -> 图片索引检索 -> 候选文物 -> 文本知识库检索 -> 文本大模型生成 -> TTS
```

特点：

- 稳定
- 可解释
- 易定位错误来源
- 更适合作为默认演示路径

### 方案 B

流程：

```text
图片 + 问题 -> 多模态大模型
```

或 grounded 版本：

```text
图片 + 问题 + 检索上下文 -> 多模态大模型
```

特点：

- 交互自然
- direct 模式容易幻觉
- grounded 模式明显优于 direct
- 更适合作为实验对比路径

## 8. 项目修复优先级

当前项目已经可运行，但还有几项明确的修复优先级。后续开发建议按以下顺序推进：

### P0：优先修复

1. 清理代码、日志、摘要文件中的乱码和编码污染
2. 对齐 README、`config.yaml`、`src/config.py` 与 `src/llm.py` 的 provider 支持说明
3. 提升裁判脚本 JSON 输出的稳定性，减少 `invalid_json`

### P1：中期修复

4. 统一 `outputs/raw`、`outputs/judged`、`outputs/metrics` 的命名规则
5. 给构建和评测脚本增加最小 smoke test / regression test
6. 增加运行耗时、资源占用、GPU 依赖程度等工程指标统计

### P2：后续增强

7. 增加知识库外图片、模糊图、遮挡图等负样本拒答评测
8. 增加人工抽样评审，校验大模型裁判结果
9. 进一步拆分大型评测脚本，降低耦合度

## 9. 兼容性说明

本次重构后：

- 推荐使用 `scripts/...` 下的标准入口
- 仓库根目录同名脚本仍然保留，可继续使用原命令
- `app.py`、`run_cli.py`、`src/` 接口未改动

也就是说，旧命令不会因为这次目录整理而失效。

## Unified Five-Chain Evaluation

The main horizontal experiment uses `data/multimodal_eval/test_images.jsonl` and the Hugging Face version of `Qwen2.5-VL-3B-Instruct`.

Supported chains:

- `retrieval_rag_text`: image retrieval + text RAG + Qwen2.5-VL text generation
- `vl_direct`: Qwen2.5-VL direct image QA
- `vl_rag`: Qwen2.5-VL + RAG
- `vl_lora`: Qwen2.5-VL + LoRA
- `vl_rag_lora`: Qwen2.5-VL + RAG + LoRA

Smoke test:

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

Long runs should be split by chain and use `--resume`:

```bash
python scripts/eval/eval_multimodal_chains.py \
  --chains vl_rag_lora \
  --model-path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --adapter-path outputs/lora/qwen25vl3b_museum \
  --resume
```

Multi-GPU sharding example, one process writes one output file. Do not let multiple processes write the same CSV:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_multimodal_chains.py \
  --chains vl_direct \
  --model-path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --num-shards 2 \
  --shard-index 0 \
  --output outputs/raw/eval_multimodal_chains_vl_direct_s0.csv \
  --summary outputs/raw/eval_multimodal_chains_vl_direct_s0_summary.txt

CUDA_VISIBLE_DEVICES=1 python scripts/eval/eval_multimodal_chains.py \
  --chains vl_direct \
  --model-path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --num-shards 2 \
  --shard-index 1 \
  --output outputs/raw/eval_multimodal_chains_vl_direct_s1.csv \
  --summary outputs/raw/eval_multimodal_chains_vl_direct_s1_summary.txt
```

Merge shard files before metrics if you used multiple GPUs:

```bash
python scripts/eval/merge_eval_csv.py \
  "outputs/raw/eval_multimodal_chains_*_s*.csv" \
  --output outputs/raw/eval_multimodal_chains_results.csv
```

Post-evaluation metrics for unified output:

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

Default output:

- `outputs/raw/eval_multimodal_chains_results.csv`
- `outputs/raw/eval_multimodal_chains_summary.txt`

Detailed plan: `docs/evaluation_five_chains_plan.md`.
