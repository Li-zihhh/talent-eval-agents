"""后台 Worker 模块 - 独立的 Redis 队列消费者，处理异步任务。

消费的任务类型：
    - document.parse: 文档解析任务
    - pipeline.continue: 流水线后续阶段（分块/索引）

Worker 从 Redis 队列中拉取任务并执行，与 FastAPI 主进程解耦。
"""

import logging
from uuid import UUID

from redis import Redis
from redis.exceptions import TimeoutError

from app.config import get_settings
from app.database import SessionLocal
from app.document_pipeline import continue_document_pipeline
from app.evidence_index_service import run_index_job
from app.logging_config import configure_logging
from app.object_store import ObjectStore
from app.services import parse_version


logger = logging.getLogger(__name__)


def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_dir, "worker")
    redis = Redis.from_url(settings.redis_url, decode_responses=True, socket_timeout=15)
    queues = ["talent:parse:queue", "talent:index:queue"]
    logger.info("worker_started queues=%s", ",".join(queues))
    while True:
        try:
            item = redis.blpop(queues, timeout=5)
        except TimeoutError:
            continue
        except Exception:
            logger.exception("worker_queue_read_failed queue=talent:parse:queue")
            raise
        if not item:
            continue
        queue_name, payload = item
        job_id, version_id = payload.split(":", 1)
        logger.info("worker_job_received queue=%s job_id=%s version_id=%s", queue_name, job_id, version_id)
        with SessionLocal() as db:
            if queue_name == "talent:index:queue":
                job = run_index_job(db, UUID(job_id), UUID(version_id))
                logger.info("worker_index_job_completed job_id=%s status=%s indexed_count=%s", job.id, job.status, job.indexed_count)
            else:
                job = parse_version(db, ObjectStore(), UUID(version_id), UUID(job_id))
                logger.info("worker_parse_job_completed job_id=%s status=%s parser=%s", job.id, job.status, job.parser_name)
                if job.status.value == "succeeded":
                    index_job = continue_document_pipeline(
                        db,
                        store=ObjectStore(),
                        redis_client=redis,
                        version_id=UUID(version_id),
                        settings=settings,
                    )
                    logger.info(
                        "worker_pipeline_continued version_id=%s index_job_id=%s status=%s",
                        version_id,
                        index_job.id,
                        index_job.status,
                    )


if __name__ == "__main__":
    run()