# src/ AGENTS.md

本文件描述 `src/` 包内各模块的职责、依赖关系和修改约定，供自动化代码智能体参考。

## 1) 包概述

`src/` 是项目的核心 Python 包，实现 RAG 管线的全部阶段：

```
config  -->  embedder  -->  retriever  -->  prompt  -->  llm  -->  tts
              |                |
              v                v
             kb           FAISS index
```

所有模块使用包内相对导入（`from .config import ...`）。
外部入口脚本（`run_cli.py`、`app.py`、`eval_rag.py`、`build_index.py`）通过 `from src.xxx import ...` 调用。

## 2) 模块职责

### config.py

- 集中管理所有可配置参数，读取优先级：`config.yaml` > 环境变量 > 代码默认值。
- 核心函数 `_get(yaml_key_path, env_key, default)` 实现三级回退逻辑。
- `_load_yaml_config()` — 懒加载项目根目录的 `config.yaml`（需 `pyyaml`，缺失时静默回退）。
- 关键常量：`EMBED_MODEL_NAME`、`DATA_PATH`、`INDEX_PATH`、`TOP_K`、`THRESHOLD`、`MARGIN`。
- LLM 相关：`LLM_PROVIDER`（`openai` / `dashscope` / `ollama`）、`OPENAI_*` 系列、`QWEN_MODEL`、`TEMPERATURE`、`OLLAMA_*` 系列。
- TTS 相关：`VOICE`、`OUTPUT_DIR`。
- 提供 `get_api_key()` 获取 DashScope 密钥，`get_openai_api_key()` 获取 OpenAI 兼容接口密钥，缺失时抛出 `RuntimeError`。
- **修改约定**：增删配置项后，同步更新 `config.yaml`、根目录 `AGENTS.md` 第 6 节和本文件。

### embedder.py

- 封装 SentenceTransformer 嵌入模型的加载与文本编码。
- `get_embed_model()` — 单例懒加载，返回 `SentenceTransformer` 实例。
- `encode_texts(texts)` — 将文本列表编码为 L2 归一化的 `np.ndarray`（float32, shape `(n, dim)`）。
- 依赖：`config.EMBED_MODEL_NAME`、`numpy`、`faiss`、`sentence_transformers`。

### kb.py

- 知识库文档加载。
- `load_docs()` — 从 `DATA_PATH` 读取文本，按双换行 `\n\n` 切分文档块，返回 `list[str]`。
- 空库时抛出 `RuntimeError`。
- 依赖：`config.DATA_PATH`。

### retriever.py

- 向量检索 + 规则匹配混合检索器。
- `get_index()` / `get_docs()` — 单例懒加载 FAISS 索引和文档列表。
- `retrieve(query, top_k, threshold, margin)` — 核心检索函数，返回 `list[tuple[str, float]]`。
  - 规则优先：query 包含展品名时直接命中（得分 1.0）。
  - 向量检索：FAISS inner-product search。
  - 动态阈值：短 query（<=8 字符）放宽阈值至 `min(threshold, 0.55)`。
  - 相对阈值（动态 Top-k）：只保留得分 >= `max_score - margin` 的结果。
- 内部辅助 `_extract_name(doc)` — 从文档中提取 `【展品名称】` 字段。
- 依赖：`config`、`embedder`、`kb`、`faiss`、`numpy`。

### prompt.py

- Prompt 工程模块，构造发送给 LLM 的提示文本。
- `clip(s, n=450)` — 截断长文本并追加省略号。
- `extract_meta(block_text)` — 正则提取展品名称和所属时代。
- `build_prompt(query, contexts)` — 将检索结果组装为讲解员 prompt，限定回答规则（口语化、150 字以内、不编造、中文）。
- `build_citation(contexts)` — 生成 `资料来源：xxx、yyy` 格式的引用标注，去重保序。
- 依赖：`re`（标准库）。

### llm.py

- LLM 调用统一入口。
- `call_llm(prompt)` — 根据 `LLM_PROVIDER` 分发到对应后端（`openai` / `dashscope` / `ollama`）。
- `_call_openai(prompt)` — 通过标准 HTTP POST 调用 OpenAI 兼容的 `/chat/completions` 端点，支持自定义 base_url。
- `_call_dashscope(prompt)` — DashScope API 调用，自动判断多模态/文本模型（`_is_multimodal_model`）。
- `_call_ollama(prompt)` — 通过 HTTP POST 调用本地 Ollama `/api/generate`，stream=False。
- 三条路径均返回 `str`，失败时抛出 `RuntimeError` 并附带可操作的诊断信息。
- 依赖：`config`、`dashscope`、`json`、`urllib`（标准库）。

### tts.py

- 文本转语音模块。
- `text_to_mp3(text, filename="output.mp3")` — 使用 Edge TTS 异步生成 MP3 文件，返回输出路径。
- 内部 `_gen_mp3(text, out_path)` 为 async 协程，由 `asyncio.run()` 驱动。
- 自动创建输出目录（`exist_ok=True`）。
- 依赖：`config.VOICE`、`config.OUTPUT_DIR`、`edge_tts`、`asyncio`、`os`。

## 3) 模块依赖图

```
config.py          (无包内依赖，所有其他模块的基础)
  ^
  |
embedder.py        (依赖 config)
  ^
  |
kb.py              (依赖 config)
  ^
  |
retriever.py       (依赖 config, embedder, kb)
  
prompt.py          (无包内依赖，仅用标准库 re)

llm.py             (依赖 config)

tts.py             (依赖 config)
```

## 4) 修改约定

- **不要破坏公共 API**：`retrieve()`、`call_llm()`、`build_prompt()`、`build_citation()`、`text_to_mp3()`、`encode_texts()` 是外部入口脚本直接调用的函数，修改签名需同步更新调用方。
- **单例模式**：`embedder._embed_model`、`retriever._index`、`retriever._docs` 采用模块级单例懒加载，不要改为非懒加载或多实例模式（会影响性能和内存）。
- **内部函数**：以 `_` 开头的函数（`_call_openai`、`_call_dashscope`、`_call_ollama`、`_extract_name`、`_is_multimodal_model`、`_gen_mp3`、`_get`、`_load_yaml_config`）为内部实现，修改时无需更新外部调用方，但需保证返回类型不变。
- **配置变更**：任何新增/删除/重命名配置项，必须同步更新 `config.py`、根目录 `AGENTS.md` 第 6 节、以及本文件第 2 节。
- **中文用户消息**：`RuntimeError` 等异常中的中文提示面向终端用户，修改时保持中文、保持可操作性（说明具体该检查什么）。
- **编码规范**：遵循根目录 `AGENTS.md` 第 7 节的全部代码风格规则（PEP 8、type hints、snake_case 等）。

## 5) 常见修改场景指引

| 场景 | 涉及模块 | 注意事项 |
|---|---|---|
| 新增 LLM 后端 | `config.py`, `llm.py` | 在 `call_llm` 添加分支，`config.py` 添加对应环境变量 |
| 调整检索策略 | `retriever.py` | 保持返回类型 `list[tuple[str, float]]` 不变 |
| 修改 prompt 模板 | `prompt.py` | 注意保留"不编造"和"中文回答"约束 |
| 更换 TTS 引擎 | `tts.py`, `config.py` | 保持 `text_to_mp3()` 签名和返回值不变 |
| 更换嵌入模型 | `config.py` | 修改 `EMBED_MODEL_NAME` 默认值，需重建索引 |
| 新增知识库字段 | `kb.py`, `prompt.py`, `retriever.py` | `_extract_name` 和 `extract_meta` 可能需同步更新 |
