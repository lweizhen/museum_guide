# 文物语音导览问答系统

本项目面向博物馆文物导览场景，围绕“图片识别、知识检索、回答生成、语音播报”构建了一个可演示、可评测、可微调的中文文物导览系统。

当前仓库主要支持两条方案：

- **方案 A**：图像检索增强文本问答。先做图片检索，再做文本检索与文本 RAG 回答生成。
- **方案 B4**：`vl_rag_lora`。使用 Qwen2.5-VL + 图像检索 + 文本检索 + LoRA 适配，并在当前版本中加入**按问题类型自动切换提示词**的机制。

当前推荐的多模态 LoRA 平衡版本为：

- `qkvo + autoswitch prompt`

---

## 1. 项目特点

- 面向中文文物导览问答任务
- 支持 Streamlit 网页演示和命令行调试
- 支持文本 RAG 与图像检索
- 支持统一五链路评测
- 支持 Qwen2.5-VL LoRA 微调与消融实验
- 支持本地网页通过 SSH 隧道调用远程 GPU 推理服务

---

## 2. 当前推荐结论

基于当前实验结果，可优先这样理解：

- 原始五链路主实验中，导览质量最优的是 `retrieval_rag_text`
- 原始五链路主实验中，多模态方案最优的是 `vl_rag`
- `vl_rag_lora` 后续优化里，当前最平衡的版本是 `qkvo + autoswitch prompt`

`autoswitch prompt` 的作用是：

- 对名称、时代、材质、馆藏单位、功能用途、文化价值、历史背景等**字段型问题**，优先走“先直接回答，再简要补充”的短答模式
- 对“请介绍这件文物”“请生成一段导览讲解”等**讲解型请求**，优先走更自然的导览讲解模式

这样可以减少“字段问题被讲成整段导览、导致答非所问”的情况。

---

## 3. 目录结构

```text
museum_guide/
  app.py
  run_cli.py
  config.yaml
  config.yaml.example
  requirements.txt
  README.md
  src/
    config.py
    retriever.py
    image_retriever.py
    image_embedder.py
    llm.py
    hf_qwen_vl.py
    prompt.py
    remote_vl.py
    tts.py
  scripts/
    build/
    eval/
    judge/
    finetune/
    serve/
  data/
  index/
  outputs/
    raw/
    judged/
    metrics/
    media/
    archive/
```

---

## 4. 环境准备

### 4.1 安装依赖

在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
```

### 4.2 配置说明

配置优先级：

1. `config.yaml`
2. 环境变量
3. `src/config.py` 默认值

常见配置项包括：

- `llm_provider`
- `ollama.model`
- `ollama.multimodal_model`
- `dashscope.model`
- `dashscope.multimodal_model`
- `judge.*`
- `remote_vl.*`
- `data_path`
- `index_path`
- `output_dir`

如果新增配置项，需要同步更新：

- `config.yaml.example`
- `src/config.py`
- `README.md`
- `AGENTS.md`

---

## 5. 数据与索引构建

### 5.1 构建统一知识库

```bash
python scripts/build/prepare_combined_kb.py
```

输出：

- `data/exhibits_combined.txt`

### 5.2 构建文本索引

```bash
python scripts/build/build_index.py
```

输出：

- `index/exhibits.index`

### 5.3 构建图片索引

```bash
python scripts/build/build_image_index.py
```

输出：

- `index/exhibits_images.index`
- `index/exhibits_images_meta.json`

### 5.4 构建闭集 LoRA / 多模态评测数据集

```bash
python scripts/build/prepare_multimodal_eval_dataset.py
```

说明：

- 同一文物的图片在 `train` / `test` 内划分
- 单图文物会同时出现在 `train` 和 `test`，这是闭集适配设定，不用于开放集泛化结论
- LoRA 样本优先对齐 `vl_rag_lora` 链路
- 除 `identify` 外，其余样本默认包含 `grounding_context`
- 导览增强样本包含 `overview`、`highlight`、`story` 三类讲解任务

如果要使用 LLM 生成更自然的训练回答，可执行：

```bash
python scripts/build/prepare_multimodal_eval_dataset.py --answer-generator llm
```

如果担心一次运行过长，可分批执行，例如：

```bash
python scripts/build/prepare_multimodal_eval_dataset.py --answer-generator llm --llm-max-calls 200
```

默认缓存文件：

- `data/multimodal_eval/llm_generation_cache.jsonl`

---

## 6. 本地使用方法

### 6.1 启动命令行

```bash
python run_cli.py
```

### 6.2 启动网页

```bash
streamlit run app.py
```

当前页面提供两个主要入口：

- 方案 A：图像检索增强文本问答（远程 GPU）
- 方案 B4：Qwen2.5-VL + RAG + LoRA（自动切换提示词，远程 GPU）

---

## 7. 本地网页调用远程 GPU 服务

### 7.1 远程环境安装依赖

```bash
python -m pip install -r requirements.txt
```

### 7.2 启动远程 GPU 推理服务

```bash
python scripts/serve/serve_qwen25vl_lora_api.py \
  --model-path /path/to/Qwen2.5-VL-3B-Instruct \
  --adapter-path outputs/lora/qwen25vl3b_museum_grounded \
  --host 127.0.0.1 \
  --port 8000
```

如果要换成新的 LoRA 结果，只需要把 `--adapter-path` 指向新的适配器目录。

### 7.3 建立 SSH 隧道

在本地终端执行：

```bash
ssh -L 8000:127.0.0.1:8000 -p 你的端口 root@你的远程地址
```

例如：

```bash
ssh -L 8000:127.0.0.1:8000 -p 21870 root@connect.cqa1.seetacloud.com
```

建立隧道后，再在本地运行：

```bash
streamlit run app.py
```

### 7.4 当前推荐网页展示版本

- 方案 A 作为工程基线展示
- 方案 B4 优先使用 `qkvo + autoswitch prompt`

---

## 8. Ollama 启动方法

如果本地或远程使用 `ollama` 作为文本模型或裁判模型提供方，推荐按下面方式启动。

### 8.1 设置模型目录

```bash
export OLLAMA_MODELS=/root/autodl-tmp/ollama_models
```

### 8.2 启动服务

```bash
nohup ollama serve > ollama.log 2>&1 &
```

### 8.3 检查服务状态

```bash
ollama list
ollama ps
```

项目中常见模型包括：

- `qwen2.5:7b`
- `qwen2.5vl:7b`
- `qwen2.5:0.5b`

---

## 9. LoRA 微调方法

### 9.1 数据准备

```bash
python scripts/finetune/prepare_qwen25vl_lora_data.py
```

### 9.2 基础训练命令

```bash
python scripts/finetune/train_qwen25vl_lora.py --model-path /path/to/Qwen2.5-VL-3B-Instruct
```

### 9.3 评测 LoRA 适配器

```bash
python scripts/finetune/eval_qwen25vl_lora.py \
  --model-path /path/to/Qwen2.5-VL-3B-Instruct \
  --adapter-path outputs/lora/qwen25vl3b_museum
```

### 9.4 当前推荐的消融顺序

第一步：模块范围消融

- `qv`
- `qkvo`
- `qkvo_ffn`

第二步：在最优模块上做 rank 消融

- `r=8`
- `r=16`
- `r=32`

第三步：在最优结构上做 epoch 消融

- `epoch=1`
- `epoch=2`
- `epoch=3`

当前已有结论：

- `qkvo` 是当前最稳的 target modules 设置
- `qkvo_ffn` 没有形成稳定收益，并且推理更慢

---

## 10. 统一五链路主实验评测

统一主实验使用：

```bash
python scripts/eval/eval_multimodal_chains.py
```

固定链路名：

- `retrieval_rag_text`
- `vl_direct`
- `vl_rag`
- `vl_lora`
- `vl_rag_lora`

主测试集：

- `data/multimodal_eval/test_images.jsonl`

### 10.1 跑全部五条链路

```bash
python scripts/eval/eval_multimodal_chains.py \
  --chains retrieval_rag_text vl_direct vl_rag vl_lora vl_rag_lora \
  --model-path /path/to/Qwen2.5-VL-3B-Instruct \
  --adapter-path outputs/lora/qwen25vl3b_museum_grounded \
  --output outputs/raw/eval_multimodal_chains_results.csv \
  --summary outputs/raw/eval_multimodal_chains_results_summary.txt \
  --resume \
  --stop-on-error
```

### 10.2 单独跑某一条链路

例如只跑 `vl_rag_lora`：

```bash
python scripts/eval/eval_multimodal_chains.py \
  --chains vl_rag_lora \
  --model-path /path/to/Qwen2.5-VL-3B-Instruct \
  --adapter-path outputs/lora/qwen25vl3b_museum_grounded \
  --output outputs/raw/eval_vl_rag_lora.csv \
  --summary outputs/raw/eval_vl_rag_lora_summary.txt \
  --resume \
  --stop-on-error
```

### 10.3 分片评测

支持参数：

- `--num-shards`
- `--shard-index`

规则：

- 并行任务必须写入不同 CSV
- 不要让多个进程同时写同一个输出文件
- 跑完后用下面脚本合并：

```bash
python scripts/eval/merge_eval_csv.py
```

---

## 11. Legacy 评测命令

### 11.1 文本 RAG

```bash
python scripts/eval/eval_rag.py
```

### 11.2 方案 A 图像问答

```bash
python scripts/eval/eval_scheme_a_qa.py --with-llm
```

### 11.3 方案 A 图像描述

```bash
python scripts/eval/eval_scheme_a_caption.py --with-llm
```

### 11.4 方案 A 跨图检索

```bash
python scripts/eval/eval_scheme_a_cross_image.py
```

### 11.5 方案 B 评测

```bash
python scripts/eval/eval_scheme_b.py --mode both --stop-on-error
```

### 11.6 方案 B 语义裁判

```bash
python scripts/judge/judge_scheme_b_results.py --input outputs/raw/eval_scheme_b_results.csv
```

---

## 12. 质量裁判与生成指标

### 12.1 讲解质量裁判

```bash
python scripts/judge/judge_guide_quality.py --input outputs/raw/eval_scheme_a_qa_results.csv
python scripts/judge/judge_guide_quality.py --input outputs/judged/eval_scheme_b_judged_results.csv --group-cols mode
```

如果是统一五链路结果，例如：

```bash
python scripts/judge/judge_guide_quality.py \
  --input outputs/raw/eval_multimodal_chains_results.csv \
  --answer-col prediction \
  --reference-col gold_answer \
  --group-cols chain
```

### 12.2 文本生成指标

```bash
python scripts/eval/eval_metrics.py --input outputs/judged/eval_scheme_b_judged_results.csv --group-cols mode
```

统一五链路下建议按 `chain` 分组。

### 12.3 当前值得保留的主要结论

- `retrieval_rag_text` 是原始五链路里导览质量最优的方案
- `vl_rag` 是原始五链路里最强的多模态方案
- `qkvo + autoswitch prompt` 是 `vl_rag_lora` 后续优化中当前最平衡的版本

---

## 13. 远程结果拉回本地

假设本地项目目录为：

```powershell
D:\STUDY\graduation_design\museum_guide
```

先进入目录：

```powershell
cd D:\STUDY\graduation_design\museum_guide
```

拉回原始结果：

```powershell
scp -P 21870 root@connect.cqa1.seetacloud.com:/root/autodl-tmp/museum_guide/outputs/raw/eval_vl_rag_lora_qkvo_llm_v2_autoswitch.csv outputs/raw/
scp -P 21870 root@connect.cqa1.seetacloud.com:/root/autodl-tmp/museum_guide/outputs/raw/eval_vl_rag_lora_qkvo_llm_v2_autoswitch_summary.txt outputs/raw/
```

拉回质量裁判结果：

```powershell
scp -P 21870 root@connect.cqa1.seetacloud.com:/root/autodl-tmp/museum_guide/outputs/judged/eval_vl_rag_lora_qkvo_llm_v2_autoswitch_guide_quality.csv outputs/judged/
scp -P 21870 root@connect.cqa1.seetacloud.com:/root/autodl-tmp/museum_guide/outputs/judged/eval_vl_rag_lora_qkvo_llm_v2_autoswitch_guide_quality_summary.txt outputs/judged/
```

---

## 14. 最小验证

### 14.1 编译检查

```bash
python -m compileall src app.py run_cli.py build_index.py build_image_index.py eval_rag.py eval_scheme_a.py eval_scheme_a_caption.py eval_scheme_a_cross_image.py eval_scheme_a_qa.py eval_scheme_b.py judge_scheme_b_results.py judge_guide_quality.py eval_metrics.py prepare_combined_kb.py prepare_multimodal_eval_dataset.py
```

### 14.2 构建烟雾测试

```bash
python scripts/build/prepare_combined_kb.py
```

### 14.3 评测烟雾测试

```bash
python scripts/eval/eval_scheme_b.py --mode grounded --limit-images 2 --limit-questions 1 --dry-run
```

---

## 15. 当前推荐使用顺序

如果你第一次完整跑项目，推荐按下面顺序：

1. 安装依赖
2. 准备 `config.yaml`
3. 构建统一知识库
4. 构建文本索引
5. 构建图片索引
6. 构建多模态评测数据
7. 启动远程 GPU 服务
8. 建立 SSH 隧道并运行 `app.py`
9. 跑统一五链路主实验
10. 跑讲解质量裁判
11. 如需继续优化，再做 LoRA 微调与消融

如果你已经在做 LoRA 优化，推荐顺序是：

1. 保留 `qkvo` 作为当前最佳 LoRA 结构
2. 使用 `autoswitch prompt` 作为当前最平衡的推理版本
3. 页面展示默认优先使用 `autoswitch`
4. 论文主实验仍保留原始五链路结果，把 `autoswitch` 作为后续优化结果补充说明

---

## 16. 说明

- 当前项目还没有完整单元测试体系
- 主要依赖脚本级烟雾测试和统一评测脚本
- 不建议把新的实验结果写到未约定目录
- 如果需要清理历史结果，优先移动到 `outputs/archive/`，不要直接删除仍可能用于论文的中间结果
