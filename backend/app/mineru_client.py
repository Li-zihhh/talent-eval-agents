"""MinerU 文档解析客户端模块 - 将 PDF/图片文件发送到 MinerU 服务进行结构化解析。

MinerU (Miner Unstructured) 是一个专业的文档解析引擎，
支持将 PDF、图片等非结构化文档转换为 Markdown 和结构化 JSON。
"""

import json
import logging
from pathlib import Path

import httpx

from app.config import get_settings
from app.document_parsers import ParseResult


logger = logging.getLogger(__name__)


def parse_with_mineru(path: Path) -> ParseResult:
    """调用 MinerU 服务解析 PDF/图片文档。
    
    通过 HTTP POST 将文件发送到 MinerU 的 /file_parse 接口，
    超时时间 900 秒以支持大文件解析。
    
    Args:
        path: 待解析文件的本地路径。
        
    Returns:
        ParseResult 对象，包含解析后的 Markdown 文本和结构化内容。
        
    Raises:
        RuntimeError: 解析服务返回非成功状态时抛出。
    """
    logger.info("mineru_parse_started file=%s size_bytes=%s", path.name, path.stat().st_size)
    with path.open("rb") as handle:
        response = httpx.post(
            f"{get_settings().mineru_url}/file_parse",
            files={"files": (path.name, handle, "application/octet-stream")},
            data={"backend": "pipeline", "return_md": "true", "return_content_list": "true", "response_format_zip": "false"},
            timeout=900,
        )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "completed" or not payload.get("results"):
        logger.error("mineru_parse_invalid_response task_id=%s status=%s error=%s", payload.get("task_id"), payload.get("status"), payload.get("error"))
        raise RuntimeError(payload.get("error") or "MinerU 未返回解析结果")
    name, result = next(iter(payload["results"].items()))
    content_list = result.get("content_list", "[]")
    structured = json.loads(content_list) if isinstance(content_list, str) else content_list
    result_value = ParseResult(
        "mineru_pipeline_3.4.4",
        result.get("md_content", ""),
        {"task_id": payload.get("task_id"), "source_name": name, "backend": payload.get("backend"), "version": payload.get("version")},
        {"content_list": structured},
    )
    logger.info("mineru_parse_completed task_id=%s source_name=%s text_chars=%s", payload.get("task_id"), name, len(result_value.text))
    return result_value