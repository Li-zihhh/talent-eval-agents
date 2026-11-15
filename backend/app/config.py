"""应用配置模块 - 集中管理所有环境变量和服务配置参数。

配置项来源：.env 文件（基础设置）和代码内默认值（开发环境）。
使用 Pydantic Settings 实现类型安全的环境变量加载。
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用全局配置，所有配置项均支持通过环境变量或 .env 文件覆盖。"""

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent.parent / ".env"),
        extra="ignore",
    )

    # ── 数据存储 ──
    database_url: str = "postgresql+psycopg://talent:talent_dev_password@127.0.0.1:15432/talent_docs"
    redis_url: str = "redis://127.0.0.1:16379/0"

    # ── 对象存储 (MinIO / S3) ──
    s3_endpoint_url: str = "http://127.0.0.1:19000"
    s3_public_endpoint_url: str = "http://127.0.0.1:19000"
    s3_access_key: str = "talent_minio"
    s3_secret_key: str = "talent_minio_password"
    s3_bucket: str = "talent-documents"

    # ── 跨域配置 ──
    cors_origins: str = "http://localhost:15173,http://127.0.0.1:15173"

    # ── 文档解析服务 ──
    mineru_url: str = "http://127.0.0.1:18001"

    # ── 模型服务 ──
    dashscope_api_key: str = ""
    chat_model: str = "qwen-plus"
    judge_model: str = ""
    judge_api_key: str = ""
    judge_base_url: str = ""
    embedding_model: str = "text-embedding-v3"
    embedding_dimension: int = 1024
    rerank_model: str = "gte-rerank-v2"
    rerank_top_n: int = 20

    # ── 向量数据库 (Milvus) ──
    milvus_uri: str = "http://127.0.0.1:19531"
    milvus_token: str = "root:Milvus"
    milvus_collection: str = "talent_evidence_v2"

    # ── 日志 ──
    log_dir: str = "logs"
    service_name: str = "backend"


@lru_cache
def get_settings() -> Settings:
    """获取缓存的全局配置单例。
    使用 lru_cache 确保 Settings 实例在整个进程生命周期内只初始化一次。
    """
    return Settings()