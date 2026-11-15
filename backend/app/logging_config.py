"""日志配置模块 - 提供按日期切分的文件日志和终端日志输出。

核心组件：
    DailyFileHandler: 自定义 Handler，按日期自动切换日志文件。
    configure_logging: 一次调用即可配置根 Logger 的所有 Handler。
"""

import logging
import re
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path

# 日志格式模板
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class DailyFileHandler(logging.Handler):
    """按日期自动切换输出文件的日志处理器。
    
    每天生成一个独立的日志文件，文件名格式：{prefix}-{YYYY-MM-DD}.log。
    跨天时自动关闭旧文件、创建新文件。
    """

    def __init__(
        self,
        log_dir: Path,
        filename_prefix: str,
        date_provider: Callable[[], date] = date.today,
    ) -> None:
        super().__init__()
        self.log_dir = log_dir
        self.filename_prefix = filename_prefix
        self.date_provider = date_provider
        self._active_date: date | None = None
        self._file_handler: logging.FileHandler | None = None
        self.talent_logging_handler = True  # 标记为自定义 Handler，便于清理

    def _ensure_file_handler(self) -> logging.FileHandler:
        """确保当前日期的文件 Handler 可用。日期变更时自动重建。"""
        current_date = self.date_provider()
        if self._file_handler is not None and current_date == self._active_date:
            return self._file_handler
        if self._file_handler is not None:
            self._file_handler.close()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"{self.filename_prefix}-{current_date.isoformat()}.log"
        self._file_handler = logging.FileHandler(path, encoding="utf-8")
        self._file_handler.setFormatter(self.formatter)
        self._active_date = current_date
        return self._file_handler

    def setFormatter(self, formatter: logging.Formatter | None) -> None:
        """设置日志格式化器，同时同步到内部的 FileHandler。"""
        super().setFormatter(formatter)
        if self._file_handler is not None:
            self._file_handler.setFormatter(formatter)

    def emit(self, record: logging.LogRecord) -> None:
        """将日志记录写入当天对应的文件。"""
        try:
            self._ensure_file_handler().emit(record)
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        """关闭当前文件 Handler 并释放资源。"""
        if self._file_handler is not None:
            self._file_handler.close()
            self._file_handler = None
        super().close()


def configure_logging(
    log_dir: str | Path,
    service_name: str,
    *,
    current_date: date | None = None,
) -> None:
    """配置根 Logger：输出到终端（INFO）及按日期切分的文件日志（INFO + ERROR）。
    
    会先移除所有已注册的 talent_logging_handler，避免重复注册。
    
    Args:
        log_dir: 日志文件存放目录。
        service_name: 服务名称，用于日志文件名前缀。
        current_date: 可选，用于锁定日期（测试场景）。
    """
    directory = Path(log_dir)
    safe_service_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", service_name).strip("-") or "service"
    date_provider = (lambda: current_date) if current_date is not None else date.today
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 清理先前可能重复注册的 Handler
    for handler in list(root_logger.handlers):
        if getattr(handler, "talent_logging_handler", False):
            root_logger.removeHandler(handler)
            handler.close()

    # 终端输出 Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.talent_logging_handler = True

    # 按日切分的一般日志
    service_handler = DailyFileHandler(directory, safe_service_name, date_provider)
    service_handler.setLevel(logging.INFO)
    service_handler.setFormatter(formatter)

    # 按日切分的错误日志
    error_handler = DailyFileHandler(directory, f"{safe_service_name}-error", date_provider)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(service_handler)
    root_logger.addHandler(error_handler)