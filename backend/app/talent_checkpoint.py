"""LangGraph 检查点持久化模块 - 使用 PostgreSQL 存储 LangGraph 工作流的状态快照。

支持 LangGraph 的中断与恢复机制，工作流执行状态保存到 PostgreSQL，
服务重启后可继续执行未完成的任务。
"""

from __future__ import annotations

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool


def checkpoint_dsn(database_url: str) -> str:
    """将 SQLAlchemy 格式的数据库 URL 转换为 psycopg 兼容的连接串。

    Args:
        database_url: SQLAlchemy 连接字符串（postgresql+psycopg://）。

    Returns:
        psycopg 兼容的连接字符串（postgresql://）。
    """
    return database_url.replace("postgresql+psycopg://", "postgresql://", 1)


class PostgresCheckpointRuntime:
    """管理 FastAPI 进程内的异步连接池与 LangGraph checkpointer。

    在服务启动时初始化连接池和检查点表，服务关闭时释放资源。
    """

    def __init__(self, database_url: str) -> None:
        self._pool = AsyncConnectionPool(
            conninfo=checkpoint_dsn(database_url),
            open=False,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
        )
        self.saver = AsyncPostgresSaver(self._pool)

    async def start(self) -> AsyncPostgresSaver:
        await self._pool.open()
        await self.saver.setup()
        return self.saver

    async def close(self) -> None:
        await self._pool.close()