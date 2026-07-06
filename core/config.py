"""
核心配置模块 — 统一的全局配置管理
使用 Pydantic Settings 提供类型安全的配置读取
"""
import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# 加载 .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """应用全局配置"""

    # ── 项目路径 ──────────────────────────────────────────
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"
    logs_dir: Path = PROJECT_ROOT / "logs"
    prompts_dir: Path = PROJECT_ROOT / "prompts"
    config_dir: Path = PROJECT_ROOT / "config"

    # ── LLM 配置 ─────────────────────────────────────────
    api_key: str = os.environ.get("API_KEY", "")
    base_url: str = os.environ.get("BASE_URL", "")
    model_id: str = os.environ.get("MODEL_ID", "qwen3.7-max")
    embedding_model: str = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-v4")

    # ── RAG 配置 ─────────────────────────────────────────
    chunk_size: int = 200
    chunk_overlap: int = 20
    rag_top_k: int = 3
    bm25_top_k: int = 5
    rrf_k: int = 60
    vector_weight: float = 0.7
    bm25_weight: float = 0.3

    # ── 会话配置 ──────────────────────────────────────────
    session_timeout: int = 3600  # 会话超时时间（秒）
    max_context_messages: int = 10  # 单轮保留的最大消息数
    max_tool_calls: int = 5  # 单次工具调用上限

    # ── 记忆配置 ──────────────────────────────────────────
    memory_db_path: str = str(PROJECT_ROOT / "chroma_db" / "memory.sqlite")
    memory_collection: str = "long_term_memory"
    memory_max_results: int = 5

    # ── 评估配置 ──────────────────────────────────────────
    eval_batch_size: int = 10
    eval_metrics: List[str] = ["recall", "precision", "mrr", "ndcg"]

    # ── 服务器配置 ────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8080
    reload: bool = True
    log_level: str = "info"
    cors_origins: List[str] = ["*"]

    # ── 知识库配置 ────────────────────────────────────────
    knowledge_base_dir: str = str(PROJECT_ROOT / "data" / "knowledge")
    supported_file_types: List[str] = ["txt", "pdf", "md", "docx"]

    class Config:
        env_file = str(PROJECT_ROOT / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"  # 允许额外的环境变量（兼容旧的.env配置）


# 全局单例配置
settings = Settings()
