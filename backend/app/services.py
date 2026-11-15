"""核心业务服务模块 - 提供文档上传、解析和员工数据导入等核心业务逻辑。

主要函数：
    upload_document: 上传文档文件、创建数据库记录、触发异步解析流水线
    parse_version: 执行文档版本解析（自动选择解析引擎）
    import_employees: 批量导入员工档案到 EmployeeProfile 表
    import_jobs: 批量导入岗位描述到 JobDescription 表
"""

import hashlib
import json
import logging
import mimetypes
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.document_parsers import parse_document
from app.models import Document, DocumentVersion, EmployeeProfile, FileObject, ParseArtifact, ParseJob, ParseStatus
from app.mineru_client import parse_with_mineru
from app.object_store import ObjectStore
from app.router import ParserKind, choose_parser
from app.storage import build_object_key


logger = logging.getLogger(__name__)


def upload_document(db: Session, store: ObjectStore, *, filename: str, content: bytes, candidate_id: str, title: str, document_type: str, employee_id: UUID | None = None, knowledge_base_id: UUID | None = None, permission_scope: str = "hr_private", tenant_id: str = "course-demo") -> Document:
    """上传文档文件并创建完整的数据库记录链。

    执行步骤：
        1. 计算文件 SHA256 摘要。
        2. 创建 Document 记录并生成物料编号 (MAT-xxxxxx)。
        3. 上传文件到对象存储 (MinIO)。
        4. 创建 FileObject + DocumentVersion 记录。
        5. 异步触发解析流水线。

    Args:
        db: 数据库会话。
        store: 对象存储客户端。
        filename: 原始文件名。
        content: 文件二进制内容。
        candidate_id: 关联的候选人 ID（员工编号）。
        title: 文档标题。
        document_type: 文档类型（如 resume、certificate）。
        employee_id: 关联的员工 UUID（可选）。
        knowledge_base_id: 关联的知识库 UUID（可选）。
        permission_scope: 权限范围。
        tenant_id: 租户 ID。

    Returns:
        创建的 Document 对象。
    """
    logger.info(
        "document_upload_started filename=%s size_bytes=%s employee_id=%s knowledge_base_id=%s document_type=%s",
        filename,
        len(content),
        employee_id,
        knowledge_base_id,
        document_type,
    )
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    digest = hashlib.sha256(content).hexdigest()
    employee = db.get(EmployeeProfile, employee_id) if employee_id else None
    employee_no = employee.employee_no if employee else candidate_id
    document = Document(candidate_id=employee_no, employee_id=employee_id, knowledge_base_id=knowledge_base_id, tenant_id=tenant_id, title=title, document_type=document_type, permission_scope=permission_scope)
    db.add(document)
    db.flush()
    document.material_no = f"MAT-{str(document.id).split('-')[0].upper()}"
    key = build_object_key(tenant_id, str(document.id), 1, filename)
    store.put_bytes(key, content, mime)
    file_object = FileObject(bucket_name=store.settings.s3_bucket, object_key=key, original_name=filename, mime_type=mime, size_bytes=len(content), sha256=digest)
    db.add(file_object)
    db.flush()
    version = DocumentVersion(document_id=document.id, file_object_id=file_object.id, version_no=1, is_current=True)
    db.add(version)
    db.commit()
    db.refresh(document)
    logger.info(
        "document_upload_completed document_id=%s material_no=%s object_key=%s sha256=%s",
        document.id,
        document.material_no,
        key,
        digest,
    )
    return document


def parse_version(db: Session, store: ObjectStore, version_id: UUID, job_id: UUID | None = None) -> ParseJob:
    """执行文档版本的解析任务。

    根据文件类型自动选择解析引擎（PDF->MinerU, XLSX->Native 等），
    将解析结果保存为 ParseArtifact，并更新 ParseJob 状态。

    Args:
        db: 数据库会话。
        store: 对象存储客户端（用于下载文件）。
        version_id: 要解析的文档版本 ID。
        job_id: 已有的解析任务 ID（可选，用于重试）。

    Returns:
        解析任务对象 ParseJob。
    """
    version = db.get(DocumentVersion, version_id)
    if not version:
        raise ValueError("文档版本不存在")
    file_object = db.get(FileObject, version.file_object_id)
    parser_kind = choose_parser(file_object.original_name)
    logger.info(
        "parse_job_started job_id=%s version_id=%s filename=%s parser_kind=%s",
        job_id,
        version_id,
        file_object.original_name,
        parser_kind.value,
    )
    job = db.get(ParseJob, job_id) if job_id else None
    if job is None:
        job = ParseJob(document_version_id=version.id, parser_name=parser_kind.value)
        db.add(job)
    job.parser_name = parser_kind.value
    job.status = ParseStatus.RUNNING
    job.progress = 10
    job.started_at = datetime.now(UTC)
    db.commit()
    try:
        content = store.get_bytes(file_object.object_key)
        suffix = Path(file_object.original_name).suffix
        # Windows 下 NamedTemporaryFile 保持打开的句柄，解析器无法再次打开同名文件，
        # 因此先写入一个已关闭的临时文件，解析结束后再显式删除。
        handle, temp_name = tempfile.mkstemp(suffix=suffix)
        temp_path = Path(temp_name)
        try:
            with os.fdopen(handle, "wb") as temp:
                temp.write(content)
            if parser_kind == ParserKind.MINERU:
                result = parse_with_mineru(temp_path)
            else:
                try:
                    result = parse_document(temp_path)
                except Exception as exc:
                    if suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg"}:
                        raise
                    logger.warning(
                        "basic_parser_failed_fallback_mineru job_id=%s filename=%s reason=%s",
                        job.id,
                        file_object.original_name,
                        exc,
                    )
                    result = parse_with_mineru(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)
        markdown = result.text.encode("utf-8")
        structured = json.dumps(result.structured or result.metadata, ensure_ascii=False, indent=2).encode("utf-8")
        base = f"artifacts/{job.id}"
        for artifact_type, key, body, content_type in [
            ("markdown", f"{base}/content.md", markdown, "text/markdown; charset=utf-8"),
            ("structured_json", f"{base}/structured.json", structured, "application/json"),
        ]:
            store.put_bytes(key, body, content_type)
            db.add(ParseArtifact(parse_job_id=job.id, artifact_type=artifact_type, bucket_name=store.settings.s3_bucket, object_key=key, content_type=content_type, size_bytes=len(body)))
            logger.info("parse_artifact_saved job_id=%s artifact_type=%s object_key=%s size_bytes=%s", job.id, artifact_type, key, len(body))
        job.parser_name = result.parser_name
        job.status = ParseStatus.SUCCEEDED
        job.progress = 100
        logger.info("parse_job_succeeded job_id=%s parser_name=%s text_chars=%s", job.id, result.parser_name, len(result.text))
    except Exception as exc:
        job.status = ParseStatus.FAILED
        job.error_code = "PARSE_FAILED"
        job.error_message = str(exc)[:2000]
        job.progress = 100
        logger.exception("parse_job_failed job_id=%s version_id=%s error=%s", job.id, version_id, exc)
    job.finished_at = datetime.now(UTC)
    db.commit()
    db.refresh(job)
    logger.info("parse_job_finished job_id=%s status=%s progress=%s", job.id, job.status, job.progress)
    return job


def current_version(db: Session, document_id: UUID) -> DocumentVersion | None:
    return db.scalar(select(DocumentVersion).where(DocumentVersion.document_id == document_id, DocumentVersion.is_current.is_(True)))