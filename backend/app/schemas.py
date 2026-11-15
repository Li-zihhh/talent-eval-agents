"""Pydantic 数据模型定义 - 用于 API 请求/响应的序列化与校验。

本模块定义了文档创建等核心数据结构的校验 Schema。
"""

from pydantic import BaseModel, Field


class DocumentCreate(BaseModel):
    """文档创建请求的数据校验模型。"""
    candidate_id: str = Field(min_length=1, max_length=64)
    tenant_id: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=255)
    document_type: str = Field(min_length=1, max_length=64)
    source: str = "manual_upload"
    permission_scope: str = "hr_private"
    confidentiality_level: str = "internal"