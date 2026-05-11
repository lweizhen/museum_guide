"""项目配置中心。

本文件集中读取模型、知识库、索引、TTS、远程服务和裁判模型等配置。
读取顺序是：`config.yaml` 优先，其次环境变量，最后使用代码默认值。
这样本地演示、远程 GPU 评测和论文实验可以共用同一套配置入口。
"""

from __future__ import annotations

import os
from pathlib import Path

# ======================
# 加载项目级 config.yaml
# ======================
# 优先级：config.yaml > 环境变量 > 代码默认值

_yaml_cfg: dict = {}

def _load_yaml_config() -> dict:
    """从项目根目录尝试加载 `config.yaml`。

    返回空字典表示没有找到配置文件，后续会自动回退到环境变量或默认值。
    """
    global _yaml_cfg
    if _yaml_cfg:
        return _yaml_cfg

    # 向上查找 config.yaml：先从 src/ 的父目录（即项目根）找
    candidates = [
        Path(__file__).resolve().parent.parent / "config.yaml",
        Path.cwd() / "config.yaml",
    ]
    for p in candidates:
        if p.is_file():
            try:
                import yaml
                with open(p, "r", encoding="utf-8") as f:
                    _yaml_cfg = yaml.safe_load(f) or {}
                break
            except ImportError:
                # PyYAML 未安装，回退到纯环境变量模式
                break
            except Exception:
                break
    return _yaml_cfg


def _get(yaml_key_path: str, env_key: str, default: str) -> str:
    """
    按统一优先级获取配置值。

    参数：
    - `yaml_key_path`：`config.yaml` 中的嵌套键路径，例如 `dashscope.api_key`
    - `env_key`：对应的环境变量名
    - `default`：前两者都不存在时使用的默认值
    """
    # 1) 尝试从 yaml 读取
    cfg = _load_yaml_config()
    parts = yaml_key_path.split(".")
    node = cfg
    for part in parts:
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            node = None
            break
    if node is not None and str(node).strip():
        return str(node).strip()

    # 2) 尝试环境变量
    env_val = os.getenv(env_key)
    if env_val is not None and env_val.strip():
        return env_val.strip()

    # 3) 默认值
    return default


# ======================
# Embedding / 数据 / 索引
# ======================
EMBED_MODEL_NAME = _get("embedding.model_name", "EMBED_MODEL_NAME",
                        "paraphrase-multilingual-MiniLM-L12-v2")

DATA_PATH = _get("data_path", "DATA_PATH", "data/exhibits_combined.txt")
INDEX_PATH = _get("index_path", "INDEX_PATH", "index/exhibits.index")

IMAGE_CSV_PATH = _get(
    "image_index.csv_path",
    "IMAGE_CSV_PATH",
    "bowuguozhongguo_names_filtered.csv",
)
IMAGE_CACHE_DIR = _get(
    "image_index.image_cache_dir",
    "IMAGE_CACHE_DIR",
    "data/museumschina_images",
)
IMAGE_INDEX_PATH = _get(
    "image_index.index_path",
    "IMAGE_INDEX_PATH",
    "index/exhibits_images.index",
)
IMAGE_META_PATH = _get(
    "image_index.meta_path",
    "IMAGE_META_PATH",
    "index/exhibits_images_meta.json",
)
IMAGE_MODEL_NAME = _get(
    "image_index.model_name",
    "IMAGE_MODEL_NAME",
    "clip-ViT-B-32",
)
IMAGE_TOP_K = int(_get("image_index.top_k", "IMAGE_TOP_K", "5"))
IMAGE_MIN_SCORE = float(_get("image_index.min_score", "IMAGE_MIN_SCORE", "0.55"))
IMAGE_MIN_GAP = float(_get("image_index.min_gap", "IMAGE_MIN_GAP", "0.03"))
IMAGE_MAX_IMAGES_PER_ITEM = int(
    _get("image_index.max_images_per_item", "IMAGE_MAX_IMAGES_PER_ITEM", "3")
)

TOP_K = int(_get("retrieval.top_k", "TOP_K", "5"))
THRESHOLD = float(_get("retrieval.threshold", "THRESHOLD", "0.5"))
MARGIN = float(_get("retrieval.margin", "MARGIN", "0.08"))  # 动态 Top-k

# ======================
# LLM Provider 选择
#   - dashscope: 走 DashScope API
#   - ollama:    走本地 Ollama
# ======================
LLM_PROVIDER = _get("llm_provider", "LLM_PROVIDER", "dashscope").lower()

# ======================
# DashScope（API）配置
# ======================
QWEN_MODEL = _get("dashscope.model", "QWEN_MODEL", "qwen3.5-plus")
QWEN_MULTIMODAL_MODEL = _get(
    "dashscope.multimodal_model",
    "QWEN_MULTIMODAL_MODEL",
    QWEN_MODEL,
)
TEMPERATURE = float(_get("dashscope.temperature", "TEMPERATURE", "0.3"))


def get_api_key() -> str:
    """
    获取 DashScope API Key。

    DashScope 分支调用通义千问 API 时会使用该函数；如果未配置则抛出
    明确异常，避免后续请求失败时难以定位问题。
    """
    key = _get("dashscope.api_key", "DASHSCOPE_API_KEY", "")
    if not key:
        raise RuntimeError(
            "未配置 DashScope API Key，请在 config.yaml 的 dashscope.api_key "
            "或环境变量 DASHSCOPE_API_KEY 中设置。"
        )
    return key


# ======================
# TTS
# ======================
VOICE = _get("tts.voice", "VOICE", "zh-CN-XiaoxiaoNeural")
OUTPUT_DIR = _get("tts.output_dir", "OUTPUT_DIR", "outputs/media")

# ======================
# Ollama（本地）配置
# ======================
OLLAMA_BASE_URL = _get("ollama.base_url", "OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = _get("ollama.model", "OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_MULTIMODAL_MODEL = _get(
    "ollama.multimodal_model",
    "OLLAMA_MULTIMODAL_MODEL",
    OLLAMA_MODEL,
)
OLLAMA_TEMPERATURE = float(_get("ollama.temperature", "OLLAMA_TEMPERATURE", str(TEMPERATURE)))
OLLAMA_TIMEOUT_SECONDS = int(
    _get("ollama.timeout_seconds", "OLLAMA_TIMEOUT_SECONDS", "600")
)

# ======================
# Judge LLM / semantic evaluation
# ======================
# provider:
#   - same: use the normal text LLM path
#   - dashscope: use DashScope text generation with judge.model
#   - ollama: use Ollama text generation with judge.model
#   - openai: use an OpenAI-compatible chat/completions endpoint
JUDGE_PROVIDER = _get("judge.provider", "JUDGE_PROVIDER", "same").lower()
JUDGE_MODEL = _get("judge.model", "JUDGE_MODEL", "")
JUDGE_BASE_URL = _get("judge.base_url", "JUDGE_BASE_URL", "https://api.openai.com/v1")
JUDGE_API_KEY = _get("judge.api_key", "JUDGE_API_KEY", "")
JUDGE_TEMPERATURE = float(_get("judge.temperature", "JUDGE_TEMPERATURE", "0.0"))
JUDGE_TIMEOUT_SECONDS = int(_get("judge.timeout_seconds", "JUDGE_TIMEOUT_SECONDS", "120"))

# ======================
# Remote Qwen2.5-VL service
# ======================
REMOTE_VL_BASE_URL = _get(
    "remote_vl.base_url",
    "REMOTE_VL_BASE_URL",
    "http://127.0.0.1:8000",
)
REMOTE_VL_TIMEOUT_SECONDS = int(
    _get("remote_vl.timeout_seconds", "REMOTE_VL_TIMEOUT_SECONDS", "600")
)
