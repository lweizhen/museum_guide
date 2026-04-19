# 基于多模态大模型的文物语音导览助手

本项目面向博物馆文物导览场景，研究并实现了一个集成图像识别、知识库检索、大模型问答、语音播报和网页交互的智能导览原型系统。系统支持用户通过文字或图片提问，并围绕文物名称、时代、材质、馆藏单位、功能用途、文化价值和历史背景等内容生成讲解式回答。

项目当前包含两类核心技术路线：

- 方案 A：图像检索增强 + 文本 RAG。系统先通过图片索引识别候选文物，再检索文本知识库，最后由文本大模型生成回答。
- 方案 B：多模态大模型问答。系统将图片直接输入视觉语言模型，并对 direct、RAG、LoRA 和 RAG+LoRA 等链路进行对比。

当前论文实验主线采用“工程基线 + 多模态大模型消融实验”的组织方式：

- `retrieval_rag_text`：图像检索 + 文本 RAG，作为工程稳定基线。
- `vl_direct`：Qwen2.5-VL 直接图像问答，作为多模态大模型基础组。
- `vl_rag`：Qwen2.5-VL + 检索增强上下文，用于观察 RAG 的作用。
- `vl_lora`：Qwen2.5-VL + LoRA 微调，用于观察 LoRA 的作用。
- `vl_rag_lora`：Qwen2.5-VL + 检索增强上下文 + LoRA 微调，用于观察 RAG 与 LoRA 结合后的效果。

## 1. 项目功能

项目目前支持以下功能：

- 文本问答：用户输入文物相关问题，系统从知识库检索相关文物信息并生成回答。
- 图片问答：用户上传文物图片，系统先识别候选文物，再结合用户问题生成讲解。
- 语音播报：系统可将生成回答转换为语音，用于语音导览场景。
- 网页演示：通过 Streamlit 提供可交互页面。
- 命令行问答：通过 CLI 快速测试文本问答链路。
- 知识库构建：支持将基础文物资料和博物馆中国数据合并为统一知识库。
- 文本索引构建：使用向量模型和 FAISS 构建文本检索索引。
- 图片索引构建：使用图像编码模型和 FAISS 构建图片检索索引。
- 多模态评测：支持方案 A、方案 B 和统一五链路实验评测。
- LoRA 微调：支持基于 Hugging Face 版 Qwen2.5-VL-3B-Instruct 进行闭集 LoRA 微调。
- 自动指标评测：支持 ROUGE-L、BLEU、语义相似度、自动正确率和平均自动得分。
- 大模型裁判：支持语义裁判和导览讲解质量裁判。

## 2. 项目目录

当前项目已经按职责整理。标准脚本入口位于 `scripts/` 目录下，仓库根目录仍保留同名兼容包装脚本，因此新旧命令都可以使用。后续维护时优先使用 `scripts/` 下的标准路径。

```text
museum_guide/
  app.py                         # Streamlit 网页入口
  run_cli.py                     # 命令行问答入口
  config.yaml                    # 本地配置文件，不建议提交真实密钥
  config.yaml.example            # 配置模板
  requirements.txt               # Python 依赖
  src/                           # 核心代码
    config.py                    # 配置读取
    retriever.py                 # 文本检索
    image_retriever.py           # 图片检索
    llm.py                       # LLM 调用封装
    hf_qwen_vl.py                # Hugging Face Qwen2.5-VL 推理封装
    prompt.py                    # 提示词模板
    tts.py                       # 语音合成
  scripts/
    build/                       # 知识库、索引、评测数据集构建
    eval/                        # 离线评测脚本
    judge/                       # 大模型裁判脚本
    finetune/                    # Qwen2.5-VL LoRA 微调脚本
    data_tools/                  # 数据清洗与外部数据处理工具
  data/
    exhibits.txt                 # 手工整理核心文物知识
    exhibits_museumschina.txt    # 博物馆中国数据
    exhibits_combined.txt        # 合并后的统一知识库
    multimodal_eval/             # 多模态评测与 LoRA 数据
  index/
    exhibits.index               # 文本索引
    exhibits_images.index        # 图片索引
    exhibits_images_meta.json    # 图片索引元数据
  outputs/
    raw/                         # 原始评测结果
    metrics/                     # ROUGE、BLEU、语义相似度等指标
    judged/                      # 大模型裁判结果
    lora/                        # LoRA adapter，本地备份即可，不建议提交 GitHub
    media/                       # TTS 音频
    archive/                     # 历史材料
  docs/                          # 实验计划与说明文档
```

## 3. 环境安装

建议使用 Python 虚拟环境。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 下激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如果使用 Hugging Face 模型或 sentence-transformers 模型，建议将缓存放到数据盘，避免系统盘占满：

```bash
export HF_HOME=/root/autodl-tmp/hf_home
export HF_HUB_CACHE=/root/autodl-tmp/hf_home/hub
export SENTENCE_TRANSFORMERS_HOME=/root/autodl-tmp/hf_home
```

如果使用 AutoDL 网络加速：

```bash
source /etc/network_turbo
```

## 4. 配置说明

配置读取优先级如下：

1. `config.yaml`
2. 环境变量
3. `src/config.py` 默认值

常用配置项包括：

- `llm_provider`：文本大模型 provider。
- `ollama.model`：Ollama 文本模型。
- `ollama.multimodal_model`：Ollama 多模态模型。
- `dashscope.model`：DashScope 文本模型。
- `dashscope.multimodal_model`：DashScope 多模态模型。
- `judge.*`：裁判模型配置。
- `data_path`：知识库路径。
- `index_path`：文本索引路径。
- `image_index_path`：图片索引路径。
- `image_meta_path`：图片索引元数据路径。
- `output_dir`：输出目录。

不要把真实 API Key 提交到 GitHub。推荐使用环境变量：

```bash
export DASHSCOPE_API_KEY="your_api_key"
```

Windows PowerShell：

```powershell
$env:DASHSCOPE_API_KEY="your_api_key"
```

## 5. 构建流程

所有命令默认在仓库根目录执行。

### 5.1 构建统一知识库

```bash
python scripts/build/prepare_combined_kb.py
```

输出：

```text
data/exhibits_combined.txt
```

兼容命令：

```bash
python prepare_combined_kb.py
```

### 5.2 构建文本索引

```bash
python scripts/build/build_index.py
```

输出：

```text
index/exhibits.index
```

兼容命令：

```bash
python build_index.py
```

### 5.3 构建图片索引

```bash
python scripts/build/build_image_index.py
```

输出：

```text
index/exhibits_images.index
index/exhibits_images_meta.json
```

兼容命令：

```bash
python build_image_index.py
```

### 5.4 构建闭集多模态评测与 LoRA 数据集

```bash
python scripts/build/prepare_multimodal_eval_dataset.py
```

输出：

```text
data/multimodal_eval/artifacts.jsonl
data/multimodal_eval/train_images.jsonl
data/multimodal_eval/val_images.jsonl
data/multimodal_eval/test_images.jsonl
data/multimodal_eval/train_lora.jsonl
data/multimodal_eval/val_lora.jsonl
data/multimodal_eval/test_lora.jsonl
data/multimodal_eval/summary.json
```

当前数据集采用闭集设定：文物集合固定，训练集和测试集覆盖同一批文物，但尽量使用不同图片划分。对于只有一张图片的文物，同一张图会同时写入训练集和测试集，这是为了保证闭集文物识别任务中所有文物都有训练和测试样本。

`train_lora.jsonl` 面向 `vl_rag_lora` 链路构建，除识别样本外，讲解、描述和问答样本会在 instruction 中加入同一文物的参考资料上下文。当前版本还加入了总览讲解、看点讲解和故事化讲解样本，用于尝试提升导览口吻。

## 6. 交互运行

### 6.1 启动网页页面

```bash
streamlit run app.py
```

### 6.2 启动命令行问答

```bash
python run_cli.py
```

## 7. 传统评测脚本

这一部分保留了早期方案 A 和方案 B 的独立评测脚本，主要用于单独排查链路问题。论文主实验建议优先使用第 8 节的统一五链路评测。

### 7.1 文本 RAG 评测

```bash
python scripts/eval/eval_rag.py
```

兼容命令：

```bash
python eval_rag.py
```

### 7.2 方案 A 图像问答评测

```bash
python scripts/eval/eval_scheme_a_qa.py --with-llm
```

小样本调试：

```bash
python scripts/eval/eval_scheme_a_qa.py --limit-images 20 --limit-questions 3 --with-llm
```

### 7.3 方案 A 图像描述评测

```bash
python scripts/eval/eval_scheme_a_caption.py --with-llm
```

### 7.4 方案 A 跨图检索评测

```bash
python scripts/eval/eval_scheme_a_cross_image.py
```

### 7.5 方案 B 评测

```bash
python scripts/eval/eval_scheme_b.py --mode both --stop-on-error
```

小样本调试：

```bash
python scripts/eval/eval_scheme_b.py --mode grounded --limit-images 3 --limit-questions 1 --stop-on-error
```

### 7.6 方案 B 语义裁判

```bash
python scripts/judge/judge_scheme_b_results.py --input outputs/raw/eval_scheme_b_results.csv
```

### 7.7 讲解质量裁判

方案 A：

```bash
python scripts/judge/judge_guide_quality.py --input outputs/raw/eval_scheme_a_qa_results.csv
```

方案 B：

```bash
python scripts/judge/judge_guide_quality.py --input outputs/judged/eval_scheme_b_judged_results.csv --group-cols mode
```

### 7.8 文本生成指标

```bash
python scripts/eval/eval_metrics.py --input outputs/judged/eval_scheme_b_judged_results.csv --group-cols mode
```

## 8. 统一五链路评测与消融实验

论文主实验建议使用 `scripts/eval/eval_multimodal_chains.py`。它使用统一测试集 `data/multimodal_eval/test_images.jsonl`，并固定使用 Hugging Face 版 `Qwen2.5-VL-3B-Instruct` 进行横向对比。

实验设计上，`retrieval_rag_text` 不是 Qwen2.5-VL 的消融项，而是传统图像检索增强 RAG 工程基线。`vl_direct`、`vl_rag`、`vl_lora` 和 `vl_rag_lora` 才构成围绕 Qwen2.5-VL 的消融实验：以 `vl_direct` 为基础组，分别加入 RAG、LoRA，以及同时加入 RAG 和 LoRA。

### 8.1 支持的五条链路

| 链路名称 | 实验角色 | 含义 | 是否使用图片检索 | 是否使用多模态大模型 | 是否使用 RAG | 是否使用 LoRA |
|---|---|---|---:|---:|---:|---:|
| `retrieval_rag_text` | 工程基线 | 图像检索 + 文本 RAG | 是 | 否 | 是 | 否 |
| `vl_direct` | 消融基础组 | Qwen2.5-VL 直接问答 | 否 | 是 | 否 | 否 |
| `vl_rag` | RAG 消融组 | Qwen2.5-VL + RAG | 是 | 是 | 是 | 否 |
| `vl_lora` | LoRA 消融组 | Qwen2.5-VL + LoRA | 否 | 是 | 否 | 是 |
| `vl_rag_lora` | RAG+LoRA 组合组 | Qwen2.5-VL + RAG + LoRA | 是 | 是 | 是 | 是 |

### 8.2 烟雾测试

```bash
python scripts/eval/eval_multimodal_chains.py \
  --chains vl_direct,vl_rag \
  --model-path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --limit-images 2 \
  --limit-questions 1 \
  --stop-on-error
```

LoRA 链路烟雾测试：

```bash
python scripts/eval/eval_multimodal_chains.py \
  --chains vl_lora,vl_rag_lora \
  --model-path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --adapter-path outputs/lora/qwen25vl3b_museum_grounded \
  --limit-images 2 \
  --limit-questions 1 \
  --stop-on-error
```

### 8.3 单链路正式评测

建议长任务按链路分开跑，便于断点续跑和结果检查。

```bash
python scripts/eval/eval_multimodal_chains.py \
  --chains vl_rag_lora \
  --model-path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --adapter-path outputs/lora/qwen25vl3b_museum_grounded \
  --output outputs/raw/eval_multimodal_chains_vl_rag_lora_grounded.csv \
  --summary outputs/raw/eval_multimodal_chains_vl_rag_lora_grounded_summary.txt \
  --resume \
  --stop-on-error
```

### 8.4 五链路合并评测

如果五条链路分别跑出了单独 CSV，可以用合并脚本整理：

```bash
python scripts/eval/merge_eval_csv.py \
  "outputs/raw/eval_multimodal_chains_*.csv" \
  --output outputs/raw/eval_multimodal_chains_results.csv
```

如果多 GPU 分片运行，必须让每个进程写入不同 CSV，不能多个进程同时写同一个文件。

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

### 8.5 五链路指标计算

```bash
python scripts/eval/eval_metrics.py \
  --input outputs/raw/eval_multimodal_chains_results.csv \
  --prediction-col prediction \
  --reference-col gold_answer \
  --group-cols chain
```

### 8.6 五链路讲解质量裁判

```bash
python scripts/judge/judge_guide_quality.py \
  --input outputs/raw/eval_multimodal_chains_results.csv \
  --answer-col prediction \
  --reference-col gold_answer \
  --group-cols chain
```

如果已经有部分裁判结果，建议不要直接覆盖旧文件，先确认输入和输出路径。裁判脚本会输出：

```text
outputs/judged/<input_stem>_guide_quality.csv
outputs/judged/<input_stem>_guide_quality_summary.txt
outputs/judged/<input_stem>_guide_quality_breakdown.json
```

## 9. Qwen2.5-VL LoRA 微调

LoRA 微调使用 Hugging Face 模型，不使用 Ollama 模型文件。推荐在 AutoDL 或其他 GPU 环境中运行。

### 9.1 训练

```bash
python scripts/finetune/train_qwen25vl_lora.py \
  --model-path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --train-file data/multimodal_eval/train_lora.jsonl \
  --output-dir outputs/lora/qwen25vl3b_museum_grounded \
  --epochs 2 \
  --batch-size 1 \
  --gradient-accumulation-steps 8
```

### 9.2 评测 LoRA

```bash
python scripts/eval/eval_multimodal_chains.py \
  --chains vl_rag_lora \
  --model-path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --adapter-path outputs/lora/qwen25vl3b_museum_grounded \
  --output outputs/raw/eval_multimodal_chains_vl_rag_lora_grounded.csv \
  --summary outputs/raw/eval_multimodal_chains_vl_rag_lora_grounded_summary.txt \
  --resume \
  --stop-on-error
```

### 9.3 Adapter 备份

LoRA adapter 文件较大，不建议提交 GitHub。建议使用 `scp` 拉回本地备份：

```powershell
scp -P 21870 -r root@connect.cqa1.seetacloud.com:/root/autodl-tmp/museum_guide/outputs/lora/qwen25vl3b_museum_grounded .\outputs\lora\
```

## 10. AutoDL 常用流程

### 10.1 每次进入 SSH 后

```bash
cd /root/autodl-tmp/museum_guide
source .venv/bin/activate
export HF_HOME=/root/autodl-tmp/hf_home
export HF_HUB_CACHE=/root/autodl-tmp/hf_home/hub
export SENTENCE_TRANSFORMERS_HOME=/root/autodl-tmp/hf_home
```

如果要使用 Ollama 裁判模型：

```bash
export OLLAMA_MODELS=/root/autodl-tmp/ollama_models
nohup ollama serve > ollama.log 2>&1 &
```

检查 Ollama：

```bash
ollama list
ollama ps
```

### 10.2 更新代码

```bash
cd /root/autodl-tmp/museum_guide
git pull origin main
```

如果 GitHub 连接不稳定，可以先启用加速：

```bash
source /etc/network_turbo
git config --global http.version HTTP/1.1
git pull origin main
```

### 10.3 将远程结果拉回本地

在本地 PowerShell 执行：

```powershell
cd D:\STUDY\graduation_design\museum_guide

scp -P 21870 root@connect.cqa1.seetacloud.com:/root/autodl-tmp/museum_guide/outputs/raw/eval_multimodal_chains_vl_rag_lora_grounded.csv .\outputs\raw\

scp -P 21870 root@connect.cqa1.seetacloud.com:/root/autodl-tmp/museum_guide/outputs/metrics/eval_multimodal_chains_vl_rag_lora_grounded_metrics_summary.txt .\outputs\metrics\

scp -P 21870 root@connect.cqa1.seetacloud.com:/root/autodl-tmp/museum_guide/outputs/judged/eval_multimodal_chains_vl_rag_lora_grounded_guide_quality_summary.txt .\outputs\judged\
```

## 11. 当前主要实验结果

以下结果来自统一五链路实验。除 `vl_rag_lora` 外，其余链路使用五链路总表结果；`vl_rag_lora` 使用后续重新训练后表现最好的 `qwen25vl3b_museum_grounded` 版本。

实验分析时建议将 `retrieval_rag_text` 作为工程基线，将 `vl_direct`、`vl_rag`、`vl_lora` 和 `vl_rag_lora` 作为多模态大模型路线的消融实验。这样可以分别回答两个问题：第一，传统图像检索增强 RAG 是否仍然具备工程优势；第二，在 Qwen2.5-VL 路线中，RAG、LoRA 以及二者结合分别带来什么增益。

### 11.1 自动正确率与文本生成指标

| 链路 | 自动正确率 | 平均自动得分 | ROUGE-L | BLEU-4 | 语义相似度 |
|---|---:|---:|---:|---:|---:|
| `retrieval_rag_text` | 0.9420 | 0.8582 | 0.5166 | 0.3633 | 0.7119 |
| `vl_direct` | 0.1489 | 0.3912 | 0.1244 | 0.0744 | 0.5112 |
| `vl_rag` | 0.8972 | 0.8241 | 0.4668 | 0.3156 | 0.6966 |
| `vl_lora` | 0.5514 | 0.6163 | 0.5113 | 0.2453 | 0.7804 |
| `vl_rag_lora` 最优版 | 0.9841 | 0.9613 | 0.8978 | 0.5283 | 0.9635 |

从自动指标看，`vl_rag_lora` 最优版表现最好。它在自动正确率、平均自动得分、ROUGE-L、BLEU-4 和语义相似度上均取得最高结果，说明 RAG 与 LoRA 结合后，模型能够更好地适应闭集文物知识问答任务。

`vl_direct` 表现最差，自动正确率只有 0.1489。这说明仅依赖多模态大模型直接看图问答，难以稳定回答具体文物知识问题。文物导览是知识密集型任务，模型内部知识不足以替代外部知识库。

从消融角度看，`vl_rag` 相比 `vl_direct` 有明显提升，说明检索增强上下文对多模态大模型非常关键。`vl_lora` 相比 `vl_direct` 有一定提升，但不如加入 RAG 的链路，说明 LoRA 微调能够增强闭集适配能力，却不能替代知识库 grounding。`vl_rag_lora` 进一步超过 `vl_rag` 和 `vl_lora`，说明 RAG 与 LoRA 在当前闭集任务中具有互补性。

### 11.2 导览质量裁判指标

| 链路 | 通过率 | 事实性 | 依据性 | 导览风格 | 清晰度 | 完整性 | 综合分 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `retrieval_rag_text` | 0.9227 | 4.6881 | 4.9755 | 3.6988 | 4.7099 | 4.5953 | 3.9204 |
| `vl_direct` | 0.3282 | 2.5706 | 3.0902 | 3.0136 | 3.3959 | 2.7381 | 2.9970 |
| `vl_rag` | 0.9160 | 4.5859 | 4.9449 | 3.6518 | 4.6602 | 4.5257 | 3.9121 |
| `vl_lora` | 0.4890 | 3.1227 | 3.5050 | 2.8079 | 3.4306 | 2.9500 | 3.1030 |
| `vl_rag_lora` 最优版 | 0.8599 | 4.6698 | 4.9453 | 3.1690 | 4.2010 | 4.0401 | 3.8507 |

从导览质量裁判看，`retrieval_rag_text` 和 `vl_rag` 仍然更稳。它们在讲解完整性、清晰度和导览风格上略高，说明基础 RAG 链路生成的回答更规整，也更接近稳定导览文本。

在 Qwen2.5-VL 消融组内部，`vl_rag` 的质量裁判结果显著优于 `vl_direct`，再次说明 RAG 是关键增益来源。`vl_lora` 虽然比 `vl_direct` 更贴近任务，但整体仍弱于 `vl_rag`。`vl_rag_lora` 最优版在事实性和依据性上很高，但导览风格和吸引力没有超过基础 RAG 链路。这说明当前 LoRA 微调主要增强了闭集知识问答适配能力，对“像导览员一样讲解”的提升还不明显。后续如果要专门提升导览风格，需要构建更强的导览式问题测试集和更高质量的讲解训练样本。

### 11.3 实验结论

综合实验结果可以得到以下结论：

1. `retrieval_rag_text` 是工程稳定基线。它具有较高事实正确率和最高质量裁判通过率，链路可解释、错误来源清晰，适合作为默认演示和系统兜底方案。
2. `vl_direct` 是多模态大模型消融基础组，表现明显落后，说明文物导览任务高度依赖外部知识库，多模态大模型直接问答容易产生泛化回答或事实错误。
3. `vl_rag` 相比 `vl_direct` 大幅提升，证明 RAG 是提升文物问答可靠性的关键机制。
4. `vl_lora` 相比 `vl_direct` 有一定提升，但仍明显弱于 RAG 链路，说明 LoRA 微调可以增强闭集适配能力，但不能替代知识库。
5. `vl_rag_lora` 最优版在自动正确率、文本相似度和语义相似度上取得最高结果，说明 RAG 与 LoRA 结合后效果最好，是当前多模态大模型消融实验中的最优组合。

因此，论文中可以将 `vl_rag_lora` 作为综合性能最强的研究方案，将 `retrieval_rag_text` 作为工程稳定性最好的基线方案。

## 12. 输出目录约定

请尽量遵守以下输出目录约定，避免结果文件越来越乱：

```text
outputs/raw/       # 原始评测结果
outputs/metrics/   # 自动指标结果
outputs/judged/    # 大模型裁判结果
outputs/lora/      # LoRA adapter 本地备份
outputs/media/     # TTS 音频
outputs/archive/   # 历史材料
```

不建议将大型 adapter、模型权重、临时缓存提交到 GitHub。

## 13. 当前维护优先级

### P0：优先修复

1. 清理输出文件和历史日志中的乱码显示问题。
2. 继续提升大模型裁判 JSON 输出稳定性，减少 `invalid_json`。
3. 对齐 README、`config.yaml.example`、`src/config.py` 和实际 provider 支持范围。

### P1：中期完善

1. 为构建脚本和评测脚本增加最小 smoke test。
2. 增加运行耗时、GPU 占用、平均延迟等工程指标统计。
3. 进一步统一不同评测脚本的输出命名。

### P2：后续增强

1. 增加知识库外图片、模糊图片、遮挡图片等拒答评测。
2. 增加人工抽样核验，用于校准大模型裁判结果。
3. 构建专门的导览口吻测试集，评估模型是否真的更像导览员。

## 14. 本地网页调用远程 GPU 导览服务

由于本地电脑不适合加载微调后的 Qwen2.5-VL，也可能缺少完整图片检索和文本生成环境，网页演示采用“本地页面 + SSH 隧道 + 远程 GPU 导览服务”的方式。页面中提供两个方案入口：

- 方案 A：图像检索增强文本问答。该方案在远程 GPU 环境中完成图片检索、文本检索和文本大模型回答生成。
- 方案 B4：Qwen2.5-VL + RAG + LoRA。该方案在远程 GPU 环境中完成图片检索、文本检索，并调用微调后的多模态大模型生成回答。

### 14.1 远程 GPU 启动多模态推理服务

在远程 GPU 服务器中执行：

```bash
cd /root/autodl-tmp/museum_guide
source .venv/bin/activate
python -m pip install -r requirements.txt

python scripts/serve/serve_qwen25vl_lora_api.py \
  --model-path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --adapter-path outputs/lora/qwen25vl3b_museum_grounded \
  --host 127.0.0.1 \
  --port 8000
```

如果使用的是其他 LoRA adapter，请把 `--adapter-path` 替换为实际目录，例如：

```bash
--adapter-path outputs/lora/qwen25vl3b_museum
```

### 14.2 本地建立 SSH 隧道

在本地 PowerShell 另开一个窗口执行：

```powershell
ssh -p 21870 -L 8000:127.0.0.1:8000 root@connect.cqa1.seetacloud.com
```

该窗口保持连接即可。建立隧道后，本地网页访问 `http://127.0.0.1:8000` 时，实际会转发到远程 GPU 的推理服务。

### 14.3 本地启动网页

在本地项目目录执行：

```powershell
cd D:\STUDY\graduation_design\museum_guide
streamlit run app.py
```

打开页面后选择：

- `方案 A：图像检索增强文本问答（远程 GPU）`
- `方案 B4：Qwen2.5-VL + RAG + LoRA（远程 GPU）`

两个方案都需要确认远程服务已启动，并且 SSH 隧道窗口没有关闭。
