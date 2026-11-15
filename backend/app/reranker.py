"""重排序模块 - 使用阿里云 DashScope 重排序模型对检索结果进行二次排序。

Reranker 接口定义了重排序协议，支持灵活替换实现。
当前默认使用 DashScope 的 gte-rerank-v2 模型。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from dashscope.rerank import TextReRank

from app.config import Settings, get_settings


class Reranker(Protocol):
    """重排序器协议接口，定义 rerank 方法签名。"""

    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> list[dict]:
        ...


@dataclass(frozen=True)
class RerankHit:
    """单条重排序命中结果。"""
    index: int   # 原始文档列表中的索引
    score: float # 重排序相关性分数


class DashScopeReranker:
    """基于阿里云 DashScope TextReRank 服务的重排序器。

    封装了 TextReRank API 调用，对检索结果按查询相关性重新打分排序。
    """

    def __init__(self, *, model: str, api_key: str) -> None:
        self.model = model
        self.api_key = api_key

    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> list[RerankHit]:
        """对文档列表进行重排序。

        Args:
            query: 用户查询文本。
            documents: 待重排序的文档文本列表。
            top_n: 返回前 N 条最相关结果。

        Returns:
            按相关性降序排列的 RerankHit 列表。

        Raises:
            RuntimeError: DashScope API 返回错误时抛出。
        """
        response = TextReRank.call(
            model=self.model,
            query=query,
            documents=list(documents),
            top_n=top_n,
            api_key=self.api_key,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"rerank failed: status={response.status_code} "
                f"code={getattr(response, 'code', None)} message={getattr(response, 'message', None)}"
            )
        return [
            RerankHit(index=item.index, score=float(item.relevance_score))
            for item in response.output.results
        ]


def get_reranker(settings: Settings | None = None) -> DashScopeReranker | None:
    """工厂函数：获取配置好的重排序器实例。

    Args:
        settings: 可选，传入配置覆盖默认值。

    Returns:
        DashScopeReranker 实例，如果 API Key 未配置则返回 None。
    """
    value = settings or get_settings()
    if not value.dashscope_api_key or value.dashscope_api_key == "请用户自行填写":
        return None
    return DashScopeReranker(model=value.rerank_model, api_key=value.dashscope_api_key)