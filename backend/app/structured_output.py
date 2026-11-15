"""结构化输出调用封装。

模型（当前为 qwen-plus）在 function_calling 结构化输出下偶尔返回不可解析的结果，
LangChain 会得到 None 或直接抛解析错误。裸调用会让 `None.model_validate` / `None.xxx`
变成难以理解的崩溃，因此这里统一做三件事：

1. 判空与 schema 校验；
2. 失败时把校验问题反馈给模型，再生成一次；
3. 仍然失败时抛 StructuredOutputError，由调用方决定降级方式。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Sequence, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

REPAIR_INSTRUCTION = "上次输出未通过结构校验，请严格按 schema 重新生成，不要输出解释文字："


class StructuredOutputError(RuntimeError):
    """结构化输出在重试后仍不可用。"""

    def __init__(self, schema: str, attempts: int, reason: str) -> None:
        super().__init__(f"模型未能返回可用的 {schema}（已尝试 {attempts} 次）：{reason}")
        self.schema = schema
        self.attempts = attempts
        self.reason = reason


def summarize_error(error: Exception) -> str:
    """把校验错误压缩成可反馈给模型、也可以写进日志的短文本。"""
    if isinstance(error, ValidationError):
        issues = error.errors(include_input=False, include_url=False, include_context=False)
        return json.dumps(issues, ensure_ascii=False)[:800]
    return str(error)[:800]


def invoke_structured(
    model: Any,
    schema: type[SchemaT],
    messages: Sequence[tuple[str, str]],
    *,
    method: str = "function_calling",
    attempts: int = 2,
) -> SchemaT:
    """调用模型的结构化输出并返回 Pydantic 实例。

    Args:
        model: LangChain 聊天模型。
        schema: 期望的结构化输出模型。
        messages: (role, content) 序列。
        method: with_structured_output 的解析方式。
        attempts: 最多调用次数；1 表示不重试。

    Raises:
        StructuredOutputError: 所有尝试都没有拿到合法结果。
    """
    if attempts < 1:
        raise ValueError("attempts 必须大于等于 1")
    working: list[tuple[str, str]] = list(messages)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = model.with_structured_output(schema, method=method).invoke(working)
            if result is None:
                raise ValueError("模型未返回结构化结果")
            return result if isinstance(result, schema) else schema.model_validate(result)
        except (ValidationError, ValueError) as exc:
            # OutputParserException 继承自 ValueError，同样走修复重试。
            last_error = exc
            logger.warning(
                "event=structured_output_invalid schema=%s attempt=%s max_attempts=%s reason=%s",
                schema.__name__,
                attempt,
                attempts,
                summarize_error(exc),
            )
            if attempt < attempts:
                working = [*messages, ("user", REPAIR_INSTRUCTION + summarize_error(exc))]
    reason = summarize_error(last_error) if last_error is not None else "未知原因"
    raise StructuredOutputError(schema.__name__, attempts, reason) from last_error
