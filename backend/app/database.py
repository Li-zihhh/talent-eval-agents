"""数据库连接模块 - 管理 SQLAlchemy Engine 和会话工厂。

使用 PostgreSQL 作为主数据库，通过 SQLAlchemy 2.0 的
create_engine + sessionmaker 模式管理连接池。
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import get_settings

# 全局数据库引擎，启用连接池预检 (pool_pre_ping)
engine = create_engine(get_settings().database_url, pool_pre_ping=True)

# 会话工厂，用于按需创建独立事务会话
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def get_db():
    """FastAPI 依赖注入：为每次 HTTP 请求提供一个独立的数据库会话。
    请求结束时自动关闭会话，确保连接归还连接池。
    """
    with SessionLocal() as session:
        yield session