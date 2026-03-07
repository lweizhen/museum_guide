# Museum Guide (RAG + LLM + TTS)

一个基于 **RAG（Retrieval-Augmented Generation）**
的博物馆智能讲解系统。 系统通过向量检索获取文物知识，再调用
**大语言模型生成讲解文本**，并支持 **语音播报**。

支持两种 LLM 方式：

-   DashScope API（Qwen）
-   Ollama 本地模型

------------------------------------------------------------------------

## 快速开始

```bash
# 1. 安装依赖
python -m pip install -r requirements.txt

# 2. 配置 LLM（编辑 config.yaml，选择 dashscope 或 ollama）

# 3. 构建索引
python build_index.py

# 4. 运行系统
python run_cli.py          # 命令行版
streamlit run app.py       # 网页版
```

------------------------------------------------------------------------

# 1. 安装依赖

``` bash
python -m pip install -r requirements.txt
```

------------------------------------------------------------------------

# 2. 配置 LLM（二选一）

系统支持两种 LLM 方式，可通过 `config.yaml` 或环境变量配置。

## 方式 A：使用 config.yaml（推荐）

编辑项目根目录的 `config.yaml` 文件：

**选择 DashScope API：**

```yaml
llm_provider: "dashscope"

dashscope:
  api_key: "你的API密钥"
  model: "qwen3.5-plus"
  temperature: 0.3
```

**选择 Ollama 本地模型：**

```yaml
llm_provider: "ollama"

ollama:
  base_url: "http://localhost:11434"
  model: "qwen2.5:3b"
  temperature: 0.3
```

## 方式 B：使用环境变量

**配置 DashScope API：**

Windows（Anaconda Prompt）：

``` bash
set DASHSCOPE_API_KEY=你的key
set LLM_PROVIDER=dashscope
```

**配置 Ollama 本地模型：**

1. 安装 Ollama：https://ollama.com

2. 下载模型（推荐）：

``` bash
ollama pull qwen2.5:3b
```

3. 设置环境变量：

``` bash
set LLM_PROVIDER=ollama
set OLLAMA_MODEL=qwen2.5:3b
```

------------------------------------------------------------------------

# 3. 构建向量索引

首次运行需要构建向量数据库：

``` bash
python build_index.py
```

生成文件：

    index/exhibits.index

------------------------------------------------------------------------

# 4. 命令行运行

``` bash
python run_cli.py
```

示例：

    请输入问题：介绍一下贾湖骨笛

系统流程：

用户问题\
↓\
向量检索（RAG）\
↓\
LLM生成讲解\
↓\
TTS生成语音

------------------------------------------------------------------------

# 5. 网页运行（Streamlit）

``` bash
streamlit run app.py
```

浏览器访问：

    http://localhost:8501

------------------------------------------------------------------------

# 6. RAG 效果评测

运行评测脚本：

``` bash
python eval_rag.py
```

评测结果输出到：

    outputs/eval_results.csv
    outputs/eval_summary.txt

主要指标：

-   Retrieval Top1 Accuracy
-   Retrieval TopK Accuracy
-   Negative Refuse Accuracy
-   Generation Mention Rate
-   Generation Grounded Rate

------------------------------------------------------------------------

# 7. 项目结构

    museum_guide
    │
    ├─ data
    │   ├─ exhibits.txt
    │   └─ test_questions.jsonl
    │
    ├─ index
    │   └─ exhibits.index
    │
    ├─ outputs
    │   ├─ eval_results.csv
    │   ├─ eval_summary.txt
    │   └─ output.mp3
    │
    ├─ src
    │   ├─ config.py
    │   ├─ embedder.py
    │   ├─ kb.py
    │   ├─ retriever.py
    │   ├─ llm.py
    │   ├─ prompt.py
    │   └─ tts.py
    │
    ├─ build_index.py
    ├─ run_cli.py
    ├─ eval_rag.py
    ├─ app.py
    ├─ config.yaml
    ├─ requirements.txt
    └─ README.md

------------------------------------------------------------------------

# 8. 系统架构

用户问题\
│\
▼\
Embedding (MiniLM)\
│\
▼\
FAISS 向量检索\
│\
▼\
RAG Prompt 构造\
│\
▼\
LLM (Qwen / Ollama)\
│\
▼\
生成讲解文本\
│\
▼\
TTS 语音播报

------------------------------------------------------------------------

# 9. 技术栈

-   Python
-   SentenceTransformers
-   FAISS
-   DashScope (Qwen API)
-   Ollama
-   Streamlit
-   Edge TTS

------------------------------------------------------------------------

# 10. 实验结果（示例）

    Retrieval Top1 Acc: 0.767
    Retrieval TopK  Acc: 0.833
    Negative Refuse Acc: 1.000
    Generation Mention Rate: 0.833
    Generation Grounded Rate: 1.000

说明系统在小规模文物数据集上能够较好完成 **检索 + 生成讲解** 任务。

------------------------------------------------------------------------

# 11. 常见问题

**Q: 如何切换 LLM 提供者？**

A: 编辑 `config.yaml` 中的 `llm_provider` 字段，可选值为 `dashscope` 或 `ollama`。

**Q: DashScope API 报错怎么办？**

A: 检查以下几点：
- API Key 是否正确配置
- 网络是否能访问 DashScope 服务
- 账户余额是否充足

**Q: Ollama 连接失败怎么办？**

A: 确认：
- Ollama 服务是否已启动（运行 `ollama serve`）
- 模型是否已下载（运行 `ollama list` 查看）
- 端口 11434 是否被占用

**Q: 向量索引需要重建吗？**

A: 以下情况需要重建：
- 修改了 `data/exhibits.txt` 知识库内容
- 更换了嵌入模型（`embedding.model_name`）
- 首次运行

**Q: 如何修改检索参数？**

A: 编辑 `config.yaml` 中的 `retrieval` 部分：
- `top_k`: 返回的文档数量
- `threshold`: 最低相似度阈值
- `margin`: 动态 Top-k 的相对阈值

**Q: config.yaml 会提交到 Git 吗？**

A: 建议将 `config.yaml` 添加到 `.gitignore`，避免泄露 API 密钥。
