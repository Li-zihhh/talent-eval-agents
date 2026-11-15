"""示例数据填充模块 - 导入演示用的岗位描述 (JobDescription) 数据。

在开发环境中用于初始化数据库中的示例岗位数据，
支持幂等插入（重复键则更新）。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import JobDescription


def seed_job_descriptions(
    records: Sequence[dict[str, Any]],
    *,
    session_factory: Callable[[], Session] = SessionLocal,
) -> dict[str, int]:
    """将岗位描述记录批量写入数据库。
    
    如果 job_code 已存在则执行更新，否则执行插入。
    
    Args:
        records: 岗位描述字典列表。
        session_factory: 数据库会话工厂。
        
    Returns:
        包含 inserted 和 updated 计数的字典。
    """
    inserted = 0
    updated = 0
    with session_factory() as db:
        for record in records:
            job = db.scalar(
                select(JobDescription).where(JobDescription.job_code == record["job_code"])
            )
            if job is None:
                db.add(JobDescription(**record))
                inserted += 1
                continue
            for field in ("tenant_id", "name", "content", "version", "status"):
                setattr(job, field, record[field])
            updated += 1
        db.commit()
    return {"inserted": inserted, "updated": updated}