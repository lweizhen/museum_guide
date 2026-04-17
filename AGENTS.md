# AGENTS.md

本文件是仓库级操作说明，供自动化编码代理使用。目录级 `AGENTS.md`
可以补充更窄范围规则，但不应与本文件冲突。

## 1. 仓库概览

- 语言：Python 3.x
- 交互入口：
  - `app.py`：Streamlit 页面
  - `run_cli.py`：命令行入口
- 核心架构：文本 RAG + FAISS + LLM + TTS
- 多模态分支：
  - 方案 A：图像检索增强 + 文本 RAG
  - 方案 B：多模态大模型 direct / grounded 对比
- 依赖管理：`requirements.txt`

## 2. 当前目录结构

标准脚本入口已经归类到 `scripts/` 下：

```text
scripts/
  build/
  eval/
  judge/
  data_tools/
```

仓库根目录保留了同名兼容包装脚本，因此下列两种命令都应可用：

```bash
python scripts/eval/eval_scheme_b.py
python eval_scheme_b.py
```

优先文档化和维护 `scripts/` 下的标准路径，顶层脚本仅视为兼容层。

## 3. 关键命令

所有命令默认在仓库根目录执行。

### 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 构建统一知识库

```bash
python scripts/build/prepare_combined_kb.py
```

输出：

- `data/exhibits_combined.txt`

### 构建文本索引

```bash
python scripts/build/build_index.py
```

输出：

- `index/exhibits.index`

### 构建图片索引

```bash
python scripts/build/build_image_index.py
```

输出：

- `index/exhibits_images.index`
- `index/exhibits_images_meta.json`


### 构建闭集 LoRA / 多模态评测数据集

同一文物的图片在 `train` / `test` 内划分；单图文物会把同一张图同时写入 `train` 和 `test`。该数据集类型在 `summary.json` 中标记为 `closed_set_lora`。

```bash
python scripts/build/prepare_multimodal_eval_dataset.py
```

### Qwen2.5-VL LoRA 微调

标准入口位于 `scripts/finetune/`，顶层保留兼容包装脚本。微调使用 Hugging Face 模型，不使用 Ollama 模型文件。

```bash
python scripts/finetune/prepare_qwen25vl_lora_data.py
python scripts/finetune/train_qwen25vl_lora.py --model-path /path/to/Qwen2.5-VL-3B-Instruct
python scripts/finetune/eval_qwen25vl_lora.py --model-path /path/to/Qwen2.5-VL-3B-Instruct --adapter-path outputs/lora/qwen25vl3b_museum
```
### 启动页面

```bash
streamlit run app.py
```

### 启动命令行

```bash
python run_cli.py
```

### 文本 RAG 评测

```bash
python scripts/eval/eval_rag.py
```

### 方案 A 图像问答评测

```bash
python scripts/eval/eval_scheme_a_qa.py --with-llm
```

### 方案 A 图像描述评测

```bash
python scripts/eval/eval_scheme_a_caption.py --with-llm
```

### 方案 A 跨图检索评测

```bash
python scripts/eval/eval_scheme_a_cross_image.py
```

### 方案 B 评测

```bash
python scripts/eval/eval_scheme_b.py --mode both --stop-on-error
```

### 方案 B 语义裁判

```bash
python scripts/judge/judge_scheme_b_results.py --input outputs/raw/eval_scheme_b_results.csv
```

### 讲解质量裁判

```bash
python scripts/judge/judge_guide_quality.py --input outputs/raw/eval_scheme_a_qa_results.csv
python scripts/judge/judge_guide_quality.py --input outputs/judged/eval_scheme_b_judged_results.csv --group-cols mode
```

### 文本生成指标

```bash
python scripts/eval/eval_metrics.py --input outputs/judged/eval_scheme_b_judged_results.csv --group-cols mode
```

## 4. 输出目录约定

```text
outputs/
  raw/       # 原始评测结果
  judged/    # 裁判结果
  metrics/   # ROUGE / BLEU / 语义相似度等
  media/     # TTS 音频
  archive/   # 历史材料
```

代理修改代码时不要把新输出写到未约定目录。

## 5. 配置约定

配置加载优先级：

1. `config.yaml`
2. 环境变量
3. `src/config.py` 默认值

常见配置项：

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

要求：

- 不要硬编码 API Key
- 新增配置项时，同时更新：
  - `config.yaml.example`
  - `src/config.py`
  - `README.md`
  - 本文件

## 6. 代码风格与修改约束

- 保持 Python 代码可读性，遵循 PEP 8
- 新改函数尽量补类型注解
- `src/` 内部优先使用相对导入
- 用户可见中文提示保持清晰、直接
- 不要静默吞异常
- 方案 A 与方案 B 必须概念分离：
  - 方案 A 不是端到端多模态大模型
  - 方案 B 才是图像直接参与的大模型问答

## 7. 运行与验证

当前仓库没有完整单元测试体系，主要依赖脚本烟雾验证。

建议最小验证命令：

```bash
python -m compileall src app.py run_cli.py build_index.py build_image_index.py eval_rag.py eval_scheme_a.py eval_scheme_a_caption.py eval_scheme_a_cross_image.py eval_scheme_a_qa.py eval_scheme_b.py judge_scheme_b_results.py judge_guide_quality.py eval_metrics.py prepare_combined_kb.py prepare_multimodal_eval_dataset.py
```

可选烟雾测试：

```bash
python scripts/build/prepare_combined_kb.py
python scripts/eval/eval_scheme_b.py --mode grounded --limit-images 2 --limit-questions 1 --dry-run
```

## 8. 项目修复优先级

这是当前项目维护的明确优先级，代理在整理工程时应优先服从。

### P0

1. 清理乱码、编码污染、异常字符串
2. 对齐 README / 配置 / 实际 provider 支持范围
3. 提升 judge 输出 JSON 稳定性

### P1

4. 统一 `outputs` 内各类结果文件命名
5. 增加最小 smoke test / regression test
6. 增加性能与资源占用统计

### P2

7. 增加知识库外与低质量图片拒答评测
8. 增加人工抽样核验自动裁判
9. 继续拆分大型评测脚本，降低耦合

## 9. 代理完成任务前检查

完成改动前至少检查：

1. 标准路径和兼容路径是否都还可用
2. 文档是否同步更新
3. 关键输出路径是否未被改坏
4. 是否引入新的硬编码路径或密钥
5. 是否做了最小可执行验证
