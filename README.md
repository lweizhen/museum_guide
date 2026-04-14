# Museum Guide：基于多模态大模型的语音导览助手

本项目面向博物馆导览场景，构建一个可以“看图识文物、按问题讲解、必要时语音播报”的智能导览助手。系统当前包含文本 RAG 问答、图片检索式多模态问答、端到端多模态大模型问答、离线评测和指标分析等模块。

项目目前保留两条多模态路线，便于毕业设计中做对比实验：

- **方案 A：图像检索 + 文本 RAG**，先用 CLIP/FAISS 识别图片对应文物，再检索文本知识库并让 LLM 生成回答。
- **方案 B：多模态大模型问答**，直接把图片和问题交给多模态模型，也支持加入检索上下文的 grounded 模式。

## 功能概览

| 功能 | 说明 | 主要入口 |
| --- | --- | --- |
| 文本 RAG 问答 | 根据用户文本问题检索知识库并生成讲解 | `run_cli.py`、`app.py`、`eval_rag.py` |
| 方案 A 图片问答 | 图片检索识别文物，再结合文本 RAG 回答 | `build_image_index.py`、`eval_scheme_a_qa.py` |
| 方案 A 图片描述评测 | 根据图片识别结果生成文物描述并评估 | `eval_scheme_a_caption.py` |
| 方案 A 跨图检索评测 | 测试同一文物不同图片的识别泛化能力 | `eval_scheme_a_cross_image.py` |
| 方案 B 多模态问答 | 调用本地/远程多模态模型回答图片问题 | `eval_scheme_b.py`、`src/llm.py` |
| 大模型裁判 | 对已生成答案做语义评分，避免边生成边评分太慢 | `judge_scheme_b_results.py` |
| 文本指标评测 | 计算 ROUGE-L、BLEU、语义相似度等指标 | `eval_metrics.py` |
| 数据集构建 | 生成统一多模态训练/验证/测试集 | `prepare_multimodal_eval_dataset.py` |
| 语音播报 | 将讲解文本转成语音文件 | `src/tts.py` |

## 两套多模态方案

### 方案 A：图像检索 + 文本 RAG

流程：

```text
上传图片 -> 图像向量编码 -> FAISS 图片检索 -> 候选文物 -> 文本知识库检索 -> LLM 生成回答
```

优点是可解释性强、事实约束更稳定，适合博物馆导览这类重视准确性的场景。缺点是如果第一步图片检索识别错了，后续回答也会跟着偏。

### 方案 B：多模态大模型问答

直接回答模式：

```text
上传图片 + 用户问题 -> 多模态 LLM -> 回答
```

检索增强模式：

```text
上传图片 -> 检索候选文物和文本上下文 -> 图片 + 问题 + 上下文 -> 多模态 LLM -> 回答
```

方案 B 的交互更自然，但更依赖多模态模型本身能力，也更慢。当前更推荐把它作为实验对比路线，而不是替代方案 A。

## 项目结构

```text
museum_guide/
  app.py                         # Streamlit 可视化应用
  run_cli.py                     # 命令行问答入口
  build_index.py                 # 构建文本向量索引
  build_image_index.py           # 构建图片向量索引
  prepare_combined_kb.py         # 合并文本知识库
  prepare_multimodal_eval_dataset.py
                                 # 构建多模态评测/训练数据集
  eval_rag.py                    # 文本 RAG 评测
  eval_scheme_a.py               # 方案 A 图片检索评测
  eval_scheme_a_qa.py            # 方案 A 图片问答评测
  eval_scheme_a_caption.py       # 方案 A 图片描述评测
  eval_scheme_a_cross_image.py   # 方案 A 跨图片识别评测
  eval_scheme_b.py               # 方案 B 多模态问答评测
  judge_scheme_b_results.py      # 对方案 B 结果做独立裁判评分
  eval_metrics.py                # 计算 ROUGE/BLEU/语义相似度
  src/                           # 核心源码
  data/                          # 知识库、测试问题、数据集
  index/                         # FAISS 索引和图片元数据
  outputs/                       # 运行结果输出目录
```

`outputs/` 已经按用途整理：

```text
outputs/
  raw/       # 原始评测结果
  judged/    # 大模型裁判后的结果
  metrics/   # ROUGE、BLEU、语义相似度等指标
  smoke/     # 小样本冒烟测试结果
  media/     # TTS 语音输出
  archive/   # 旧版中期/开题等提取文本归档
```

## 环境安装

在项目根目录执行：

```bash
python -m pip install -r requirements.txt
```

如果使用 Ollama 本地模型，需要先安装 Ollama，并拉取模型。例如：

```bash
ollama pull qwen2.5vl:7b
```

如果只跑文本 RAG 或方案 A 的检索流程，可以不租 GPU。若要跑方案 B 的全量多模态评测，建议使用 4090 或同级 GPU。

## 配置说明

本地配置文件为 `config.yaml`，示例文件为 `config.yaml.example`。`config.yaml` 可能包含 API Key，因此不要提交到 GitHub。

配置读取优先级：

```text
config.yaml -> 环境变量 -> 代码默认值
```

常用配置项：

| 配置项 | 说明 |
| --- | --- |
| `llm_provider` | 文本 LLM 提供方，可选 `ollama`、`dashscope`、`openai` |
| `ollama.model` | Ollama 使用的模型，例如 `qwen2.5vl:7b` |
| `dashscope.model` | DashScope 文本模型 |
| `judge.*` | 大模型裁判相关配置 |
| `data_path` | 文本知识库路径，通常是 `data/exhibits_combined.txt` |
| `index_path` | 文本向量索引路径 |
| `image_*` | 图片索引、图片缓存和 CLIP 模型配置 |
| `output_dir` | TTS 音频输出目录，默认 `outputs/media` |

推荐把敏感 Key 放到环境变量里。

Windows PowerShell：

```powershell
$env:DASHSCOPE_API_KEY="your_dashscope_api_key"
```

Linux / AutoDL：

```bash
export DASHSCOPE_API_KEY="your_dashscope_api_key"
```

## 本地运行流程

首次运行或更新知识库后，先构建知识库和索引：

```bash
python prepare_combined_kb.py
python build_index.py
python build_image_index.py
```

启动 Streamlit 应用：

```bash
streamlit run app.py
```

启动命令行问答：

```bash
python run_cli.py
```

## 评测流程

### 文本 RAG 评测

```bash
python eval_rag.py
```

输出：

```text
outputs/raw/eval_results.csv
outputs/raw/eval_summary.txt
```

### 方案 A 评测

图片问答评测：

```bash
python eval_scheme_a_qa.py --with-llm
```

图片描述评测：

```bash
python eval_scheme_a_caption.py --with-llm
```

跨图片检索评测：

```bash
python eval_scheme_a_cross_image.py
```

小样本测试可以加限制参数：

```bash
python eval_scheme_a_qa.py --limit-images 20 --limit-questions 3 --with-llm
```

### 方案 B 评测

先跑小样本确认模型和配置没有问题：

```bash
python eval_scheme_b.py --mode grounded --limit-images 3 --limit-questions 1 --stop-on-error
```

跑 direct 和 grounded 两种模式：

```bash
python eval_scheme_b.py --mode both --stop-on-error
```

如果全量太慢，可以只跑一部分：

```bash
python eval_scheme_b.py --mode both --limit-images 200 --stop-on-error
```

### 分开运行大模型裁判

方案 B 全量生成很慢，不建议边生成边裁判。推荐先生成答案，再单独评分：

```bash
python eval_scheme_b.py --mode both --stop-on-error
python judge_scheme_b_results.py --input outputs/raw/eval_scheme_b_results.csv
```

输出：

```text
outputs/judged/eval_scheme_b_judged_results.csv
outputs/judged/eval_scheme_b_judged_summary.txt
outputs/judged/eval_scheme_b_judged_breakdown.json
```

### 计算 ROUGE / BLEU / 语义相似度

方案 B：

```bash
python eval_metrics.py --input outputs/judged/eval_scheme_b_judged_results.csv --group-cols mode
```

方案 A：

```bash
python eval_metrics.py --input outputs/raw/eval_scheme_a_qa_results.csv
```

输出位于：

```text
outputs/metrics/
```

## 指标说明

当前评测指标分为三类：

| 指标 | 作用 |
| --- | --- |
| 检索指标 | 判断是否找到了正确文物，例如 Top-1 命中率、Top-K 命中率 |
| 规则指标 | 判断答案是否提到正确文物、是否拒答合理、是否使用检索证据 |
| 生成质量指标 | 判断生成文本和参考答案的相似度，例如 ROUGE-L、BLEU、语义相似度 |

其中 BLEU 和 ROUGE 更偏文本重合，中文开放式讲解里分数可能偏低；语义相似度和大模型裁判更适合评价“意思是否对”。毕业论文中建议同时报告传统指标和语义指标，不只依赖某一个分数。

## 远程 GPU 服务器运行步骤

以下步骤适用于 AutoDL 这类远程 GPU 服务器。命令中的 `ssh`、`scp`、`tmux`、`nohup` 是 Linux/服务器工具名，需要保留英文原样。

### 1. 登录远程服务器

把 `<端口>` 和 `<服务器地址>` 替换成 AutoDL 页面给你的信息：

```bash
ssh -p <端口> root@<服务器地址>
```

例如格式类似：

```bash
ssh -p 21870 root@connect.xxx.seetacloud.com
```

### 2. 进入项目目录

```bash
cd /root/autodl-tmp/museum_guide
```

如果项目还没拉取：

```bash
cd /root/autodl-tmp
git clone https://github.com/lweizhen/museum_guide.git
cd museum_guide
```

### 3. 激活虚拟环境

```bash
source .venv/bin/activate
```

如果提示 `.venv` 不存在，就重新创建：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

只要 `.venv` 还在数据盘里，之后重新 SSH 登录一般不需要重新安装依赖，只需要再次 `source .venv/bin/activate`。

### 4. 设置网络和缓存目录

```bash
source /etc/network_turbo
export HF_HOME=/root/autodl-tmp/hf_home
export HF_ENDPOINT=https://hf-mirror.com
export OLLAMA_MODELS=/root/autodl-tmp/ollama_models
```

如果要使用 DashScope 裁判模型：

```bash
export DASHSCOPE_API_KEY="your_dashscope_api_key"
```

### 5. 启动 Ollama

```bash
ollama serve > ollama.log 2>&1 &
sleep 3
ollama list
```

如果模型不存在，拉取模型：

```bash
ollama pull qwen2.5vl:7b
```

确认 GPU 是否被使用：

```bash
nvidia-smi
ollama ps
```

### 6. 跑小样本测试

```bash
python prepare_combined_kb.py
python build_index.py
python build_image_index.py
python eval_scheme_b.py --mode grounded --limit-images 3 --limit-questions 1 --stop-on-error
```

### 7. 跑全量评测

如果安装了 `tmux`：

```bash
tmux new -s evalb
python eval_scheme_b.py --mode both --stop-on-error
```

退出但不中断任务：

```text
先按 Ctrl+B，再按 D
```

重新进入：

```bash
tmux attach -t evalb
```

如果没有 `tmux`，可以用 `nohup`：

```bash
nohup python eval_scheme_b.py --mode both --stop-on-error > evalb.log 2>&1 &
tail -f evalb.log
```

### 8. 全量生成后单独跑裁判

```bash
python judge_scheme_b_results.py --input outputs/raw/eval_scheme_b_results.csv
```

再计算文本指标：

```bash
python eval_metrics.py --input outputs/judged/eval_scheme_b_judged_results.csv --group-cols mode
```

### 9. 把远程输出目录拉回本地

在本地电脑的 PowerShell 或终端执行下面命令，不是在远程服务器里执行：

```bash
scp -P <端口> -r root@<服务器地址>:/root/autodl-tmp/museum_guide/outputs .\
```

如果你的 SSH 命令是：

```bash
ssh -p 21870 root@connect.cqa1.seetacloud.com
```

那么拉取命令就是：

```bash
scp -P 21870 -r root@connect.cqa1.seetacloud.com:/root/autodl-tmp/museum_guide/outputs .\
```

## 当前实验经验

目前的实验经验大致是：

- 方案 A 更稳定、可解释，适合作为主系统和展示路线。
- 方案 B 的直接回答模式容易只看图泛泛回答，命中率和事实稳定性不如检索增强模式。
- 方案 B 的检索增强模式效果更好，但速度明显更慢，适合作为对比实验。
- 大模型裁判最好和答案生成分开跑，否则全量评测会非常慢。
- 传统指标可以保留，但中文讲解任务更应该结合语义相似度和人工样例分析。

## 后续计划

建议后续优先做这些工作：

1. 继续清洗和扩展文本知识库，提高两个方案的事实基础。
2. 为方案 A 增加更多同一文物的不同图片，测试跨图片识别泛化。
3. 对方案 B 的直接回答模式和检索增强模式做代表性成功、失败案例分析。
4. 引入 CLIPScore 或更强的图文相关性指标，用于图片描述类任务。
5. 如果要微调，优先微调讲解问答风格，不建议只用少量文物图片微调识别能力。

## 注意事项

- 不要提交 `config.yaml` 中的 API Key。
- 不要提交大型缓存、模型文件、音频输出和临时评测结果。
- 修改输出路径时，需要同步更新 README、`AGENTS.md` 和相关脚本默认参数。
- 如果正在跑长时间评测，不要同时运行其他重 GPU/CPU 任务，避免干扰结果。
