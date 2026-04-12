# Museum Guide

面向博物馆文物讲解场景的本地优先 RAG / 多模态问答项目。当前项目包含一个稳定的文本 RAG 基线、一个基于图像检索的多模态方案 A，以及一个真正把图片传给多模态大模型的方案 B 实验分支。

项目当前更适合的定位是：用可解释的方案 A 做主展示，用方案 B 做对照实验，最后在论文中比较两者的准确率、稳定性、延迟和可解释性。

## 当前功能

| 功能 | 状态 | 入口 |
| --- | --- | --- |
| 文本问答 RAG | 已实现 | `run_cli.py`、`app.py`、`eval_rag.py` |
| 文物图像检索 | 已实现 | `build_image_index.py`、`src/image_retriever.py` |
| 方案 A：图片识别 + 文本 RAG | 已实现 | `app.py`、`eval_scheme_a_qa.py` |
| 方案 A：图片标题/讲解生成评测 | 已实现 | `eval_scheme_a_caption.py` |
| 方案 A：同文物不同图片泛化评测 | 已实现 | `eval_scheme_a_cross_image.py` |
| 方案 B：端到端多模态问答 | 已实现实验入口 | `app.py`、`eval_scheme_b.py` |
| 统一多模态评测集生成 | 已实现 | `prepare_multimodal_eval_dataset.py` |
| 统一知识库合并 | 已实现 | `prepare_combined_kb.py` |
| TTS 语音输出 | 已实现 | `src/tts.py` |

## 两个方案

### 方案 A：检索增强式多模态问答

流程：

```text
上传图片 -> CLIP 图像向量检索 -> 识别候选文物 -> 构造文本查询 -> FAISS 文本检索 -> LLM 生成讲解
```

优点是可解释、可控、容易定位错误。如果答案错了，通常可以拆成三类看：图像识别错、文本知识库没召回、LLM 生成偏了。这个方案是当前推荐的主演示路径。

### 方案 B：多模态大模型问答

直接模式：

```text
上传图片 + 问题 -> 多模态 LLM -> 答案
```

知识库增强模式：

```text
上传图片 -> 图像检索候选文物 -> 文本检索上下文 -> 图片 + 问题 + 上下文 -> 多模态 LLM -> 答案
```

方案 B 更自然，但本地 4GB 显存机器上速度和稳定性受限，也更容易幻觉。它现在适合作为毕业设计中的对照实验，而不是完全替代方案 A。

## 项目结构

```text
.
├─ app.py                              # Streamlit 图形界面，支持文本 RAG、方案 A、方案 B
├─ run_cli.py                          # 命令行文本问答
├─ build_index.py                      # 构建文本 FAISS 索引
├─ build_image_index.py                # 构建图片 FAISS 索引
├─ prepare_combined_kb.py              # 合并小型人工知识库和 MuseumsChina 知识库
├─ prepare_multimodal_eval_dataset.py  # 构建统一多模态评测集
├─ eval_rag.py                         # 文本 RAG 评测
├─ eval_scheme_a.py                    # 方案 A 图片索引自测
├─ eval_scheme_a_qa.py                 # 方案 A 图片问答评测
├─ eval_scheme_a_caption.py            # 方案 A 图片讲解生成评测
├─ eval_scheme_a_cross_image.py        # 方案 A 同文物不同图片泛化评测
├─ eval_scheme_b.py                    # 方案 B 多模态问答评测
├─ src/
│  ├─ config.py                        # 配置读取
│  ├─ embedder.py                      # 文本向量
│  ├─ retriever.py                     # 文本检索
│  ├─ image_embedder.py                # 图像向量
│  ├─ image_index.py                   # 图片数据与索引辅助
│  ├─ image_retriever.py               # 图片检索
│  ├─ prompt.py                        # 文本/多模态提示词模板
│  ├─ llm.py                           # OpenAI/DashScope/Ollama/多模态调用
│  ├─ tts.py                           # 语音合成
│  └─ eval_utils.py                    # 评测公共工具
├─ data/
│  ├─ exhibits.txt                     # 早期人工整理核心文物知识库
│  ├─ exhibits_museumschina.txt        # 博物中国爬取知识库
│  ├─ exhibits_combined.txt            # 默认统一知识库，由 prepare_combined_kb.py 生成
│  ├─ test_questions.jsonl             # 文本 RAG 测试集
│  └─ multimodal_eval/                 # 多模态 train/val/test 数据集
├─ index/                              # 文本和图片索引
└─ outputs/                            # 评测结果输出
```

## 环境安装

建议在项目根目录执行：

```bash
python -m pip install -r requirements.txt
```

如果使用 Ollama 本地模型，需要先安装 Ollama 并下载对应模型，例如：

```bash
ollama pull qwen2.5vl:3b
```

你的 3050 4GB 显存更适合把本地多模态模型作为实验对照。正式跑大批量方案 B 评测或 LoRA 微调时，更建议租用 16GB 以上显存的云 GPU。

## 配置

项目配置优先级：

```text
config.yaml -> 环境变量 -> src/config.py 默认值
```

常用配置项：

| 配置 | 说明 |
| --- | --- |
| `llm_provider` | `openai`、`dashscope` 或 `ollama` |
| `openai.model` | OpenAI 兼容模型名 |
| `dashscope.model` | DashScope / Qwen 模型名 |
| `ollama.model` | Ollama 本地模型名 |
| `data_path` | 文本知识库路径，当前默认 `data/exhibits_combined.txt` |
| `index_path` | 文本向量索引路径 |
| `image_index.index_path` | 图片向量索引路径 |
| `image_index.min_score` | 图片识别最低相似度阈值 |
| `image_index.min_gap` | 图片识别 top1/top2 最小差距阈值 |

不要把真实 API Key 提交到公开仓库。建议本地使用 `config.yaml`，共享代码时使用 `config.yaml.example`。

## 首次运行

1. 合并知识库：

```bash
python prepare_combined_kb.py
```

2. 构建文本索引：

```bash
python build_index.py
```
3. 构建图片索引：

```bash
python build_image_index.py
```

4. 启动 Streamlit：

```bash
streamlit run app.py
```

## 命令行问答

```bash
python run_cli.py
```

## Streamlit 使用方式

`app.py` 里可以选择不同问答模式：

| 模式 | 用途 |
| --- | --- |
| 文本 RAG | 只输入文字问题，从文本知识库检索后回答 |
| 方案 A | 上传文物图片，先识别文物，再结合问题进行文本 RAG 回答 |
| 方案 B 直接模式 | 上传图片和问题，直接交给多模态模型回答 |
| 方案 B 知识库增强模式 | 先用图片检索获取文物候选和文本上下文，再让多模态模型回答 |

方案 A 和方案 B 的切换主要发生在 `app.py` 的界面分支中。`src/` 里的模块更多是公共能力：检索、提示词、LLM 调用、图片向量等，不强行绑定某一个方案。

## 测试集与评测

### 文本 RAG 测试集

文件：

```text
data/test_questions.jsonl
```

用于 `eval_rag.py`，测试文本检索和回答是否命中文物、是否拒答、是否基于知识库。

运行：

```bash
python eval_rag.py
```

输出：

```text
outputs/eval_results.csv
outputs/eval_summary.txt
```

### 多模态统一评测集

生成：

```bash
python prepare_multimodal_eval_dataset.py
```

输出：

```text
data/multimodal_eval/artifacts.jsonl
data/multimodal_eval/train.jsonl
data/multimodal_eval/val.jsonl
data/multimodal_eval/test.jsonl
data/multimodal_eval/train_images.jsonl
data/multimodal_eval/val_images.jsonl
data/multimodal_eval/test_images.jsonl
data/multimodal_eval/summary.json
```

这个数据集按文物划分 train/val/test，避免同一文物同时出现在训练集和测试集中。图片级文件适合视觉问答和图像检索评测，文物级文件适合描述生成或后续微调数据构造。

### 方案 A 图片索引自测

```bash
python eval_scheme_a.py
```

这更像“图片索引是否能找回自己”的自检，不是严格独立测试。

### 方案 A 图片问答评测

```bash
python eval_scheme_a_qa.py
```

常用小规模调试：

```bash
python eval_scheme_a_qa.py --limit-images 20 --limit-questions 3
```

### 方案 A 同文物不同图片泛化评测

```bash
python eval_scheme_a_cross_image.py
```

这个脚本会用每个文物的一张图作为临时图库，再用同一文物的其他图片检索它，适合观察“换一张图还认不认得出来”。

### 方案 A 图片讲解生成评测

```bash
python eval_scheme_a_caption.py
```

可选调用 LLM：

```bash
python eval_scheme_a_caption.py --with-llm
```

### 方案 B 多模态问答评测

直接模式：

```bash
python eval_scheme_b.py --mode direct --limit-images 5 --limit-questions 2
```

知识库增强模式：

```bash
python eval_scheme_b.py --mode grounded --limit-images 5 --limit-questions 2
```

两种模式对比：

```bash
python eval_scheme_b.py --mode both --limit-images 5 --limit-questions 2 --max-calls 10
```

只检查数据读取和方案 B grounded 检索链路，不调用多模态模型：

```bash
python eval_scheme_b.py --mode grounded --limit-images 2 --limit-questions 1 --dry-run
```

输出：

```text
outputs/eval_scheme_b_results.csv
outputs/eval_scheme_b_summary.txt
outputs/eval_scheme_b_breakdown.json
```

## 提示词文件为什么有多套模板

`src/prompt.py` 里保留多套模板，是因为项目现在不止一种问答链路：

| 模板 | 用途 |
| --- | --- |
| `build_prompt` | 文本 RAG 与方案 A 最终回答 |
| `build_multimodal_direct_prompt` | 方案 B 直接图片问答 |
| `build_multimodal_grounded_prompt` | 方案 B 图片 + 知识库增强回答 |

这样做的好处是实验边界清楚：方案 A 追求可控和可解释，方案 B 追求端到端多模态能力，两者可以公平对比。

## 当前推荐实验路线

1. 先保证 `prepare_combined_kb.py`、`build_index.py`、`build_image_index.py` 都能正常运行。
2. 用 `eval_rag.py` 稳定文本 RAG 基线。
3. 用 `eval_scheme_a_cross_image.py` 和 `eval_scheme_a_qa.py` 评估方案 A。
4. 用 `eval_scheme_b.py --mode both` 小规模评估方案 B。
5. 论文中重点比较方案 A 和方案 B 的准确率、拒答能力、可解释性和延迟。

## LoRA 微调建议

如果后续微调，当前更建议先微调“讲解问答风格”，不要急着微调文物图片识别。原因是每个文物只有少量图片，视觉微调很容易过拟合；而问答风格数据可以从知识库构造更多样本，收益更稳定。

推荐数据划分：

```text
train：讲解风格、问答风格、拒答风格样本
val：调参数和早停
test：完全不参与训练，用于最终报告
```

评分可以结合规则指标和大模型裁判：规则指标负责可复现，大模型裁判负责语义覆盖和讲解质量。

## 后续评测升级记录

当前 QA 自动评分主要依赖字符串匹配、包含关系和 token 覆盖率，适合名称、时代、材质、馆藏单位等封闭事实题，但不适合评价中文长段讲解。后续建议把讲解生成质量评测升级为“传统指标 + 图文相关性 + 大模型裁判”的组合：

- 传统文本指标：保留 ROUGE-L、Coverage，可选 BLEU、METEOR、CIDEr、SPICE 作为论文对照，但不把它们作为唯一结论。
- 图文相关性指标：保留 CLIPScore，用来判断生成描述和图片是否视觉相关。
- 语义裁判指标：引入 LLM-as-Judge，对 GT 描述、知识库证据和模型生成内容进行评分，维度建议包括 factuality、groundedness、completeness、fluency、hallucination、overall。
- 可靠性建议：裁判模型尽量不要和生成模型完全相同，并抽样人工复核一部分案例，避免裁判偏差。

## 常用验证命令

```bash
python -m compileall src app.py run_cli.py build_index.py eval_rag.py eval_scheme_b.py eval_scheme_a_cross_image.py prepare_combined_kb.py
python prepare_combined_kb.py
python eval_scheme_b.py --mode grounded --limit-images 2 --limit-questions 1 --dry-run
```

如果改了知识库内容或 `data_path`，需要重新运行：

```bash
python build_index.py
```
