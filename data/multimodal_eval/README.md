# Multimodal Evaluation Dataset

这份目录用于承载后续两条多模态评测路线共用的数据底座：

- `描述生成`：输入图片，输出文物介绍，和 `reference_description` 做对比
- `视觉问答`：输入图片和问题，和 `qa_pairs` 中的答案做对比

## 生成方式

在项目根目录运行：

```bash
python prepare_multimodal_eval_dataset.py
```

可选：

```bash
python prepare_multimodal_eval_dataset.py --limit-artifacts 20
```

脚本会读取：

- 当前文本知识库：`config.yaml` / `src/config.py` 指向的 `DATA_PATH`
- 当前图片索引元数据：`index/exhibits_images_meta.json`

## 输出文件

- `artifacts.jsonl`
  - 文物级样本，一条记录对应一件文物，包含多张图片、参考描述和 QA 对
- `train.jsonl` / `val.jsonl` / `test.jsonl`
  - 文物级划分结果
- `train_images.jsonl` / `val_images.jsonl` / `test_images.jsonl`
  - 图片级展开结果，便于描述生成或单图 QA 评测
- `summary.json`
  - 样本数量、划分数量、知识库匹配方式和未匹配样本统计

## 主要字段

- `artifact_id`
- `artifact_name`
- `era`
- `museum`
- `category`
- `detail_url`
- `source_urls`
- `reference_description`
- `reference_facts`
- `qa_pairs`
- `images`

其中：

- `reference_description`
  - 由知识库中的 `功能用途 / 历史意义 / 文化价值 / 纹饰与造型 / 历史背景 / 补充信息 / 故事传说`
    自动拼接而成，适合做描述生成评测参考文本
- `qa_pairs`
  - 由知识库中的结构化字段自动生成，适合做问答准确率评测
- `images`
  - 每张图片都保留 `image_path` 和 `image_url`

## 划分规则

- 按 `文物级` 划分，不按图片级随机拆分
- 这样可以避免同一件文物的不同图片同时出现在训练集和测试集里
- 默认比例：
  - `train`: 70%
  - `val`: 10%
  - `test`: 20%

## 建议用法

### 1. 描述生成评测

输入：

- `*_images.jsonl` 中的 `image_path`

参考输出：

- `reference_description`

推荐指标：

- `CLIPScore`
- `ROUGE / CIDEr / SPICE` 作为辅助
- `大模型评审` 用于判断是否有幻觉、是否提到关键事实

### 2. 视觉问答评测

输入：

- `image_path`
- `qa_pairs[].question`

参考答案：

- `qa_pairs[].answer`

推荐指标：

- 选择题：`Accuracy`
- 开放式问答：`EM / F1 / 关键词命中 / 大模型评审`

## 注意

- 这份数据集是 `评测数据底座`，不等同于高质量训练语料
- 如果后续要做真正的微调训练，建议再单独补人工清洗和更严格的参考描述质量控制
