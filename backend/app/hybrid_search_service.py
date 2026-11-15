"""混合检索服务模块 - 组合向量检索（Dense）+ 关键词检索（Sparse/BM25）+ 重排序。

核心流程：
    1. 对查询文本生成向量（Embedding）。
    2. 调用 Milvus 的 hybrid_search，融合 Dense 向量 + BM25 关键词结果 (RRF)。
    3. 对融合结果使用重排序模型进行精排。
    4. 返回最终 Top-N 排序证据列表。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.milvus_store import EvidenceFilter, EvidenceSearchResult


@dataclass(frozen=True)
class RankedEvidence:
    """经过混合检索和重排序后的最终证据条目。"""
    chunk_id: str        # 分块唯一标识
    candidate_id: str    # 候选人标识
    content: str         # 证据文本内容
    rrf_score: float     # RRF 融合分数
    rerank_score: float  # 重排序分数
    metadata: dict[str, Any]  # 元数据（文档类型、来源等）


def rerank_evidence(
    *,
    query: str,
    candidates: list[EvidenceSearchResult],
    reranker: Any,
    top_n: int,
) -> list[RankedEvidence]:
    """对混合检索候选集执行重排序。

    Args:
        query: 用户查询文本。
        candidates: RRF 融合后的候选证据列表。
        reranker: 重排序器实例（如 DashScopeReranker）。
        top_n: 重排序返回的最大条数。

    Returns:
        重排序后的 RankedEvidence 列表。
    """
    if not candidates:
        return []
    hits = reranker.rerank(
        query=query,
        documents=[item.content for item in candidates],
        top_n=min(top_n, len(candidates)),
    )
    ranked: list[RankedEvidence] = []
    for hit in hits:
        item = candidates[hit.index]
        ranked.append(
            RankedEvidence(
                chunk_id=item.chunk_id,
                candidate_id=item.candidate_id,
                content=item.content,
                rrf_score=item.score,
                rerank_score=hit.score,
                metadata=item.metadata,
            )
        )
    return ranked


def hybrid_search_evidence(
    *,
    query: str,
    filters: EvidenceFilter,
    store: Any,
    embedder: Any,
    reranker: Any,
    limit: int = 10,
    ef: int = 80,
    rrf_k: int = 60,
    rerank_top_n: int = 20,
) -> list[RankedEvidence]:
    """执行完整的混合检索流水线：向量 -> RRF 融合 -> 重排序。

    Args:
        query: 用户查询文本。
        filters: 证据过滤条件（租户、权限范围、候选人等）。
        store: Milvus 证据存储实例。
        embedder: 文本向量化实例。
        reranker: 重排序器实例。
        limit: 最终返回的结果数量。
        ef: HNSW 索引的 ef 搜索参数，越大召回越准但越慢。
        rrf_k: RRF 融合算法中的 k 参数。
        rerank_top_n: 进入重排序阶段的候选数。

    Returns:
        排序后的 RankedEvidence 列表，最多 limit 条。
    """
    # 生成查询向量
    query_vector = list(embedder.embed_query(query))
    # Milvus 混合搜索：Dense + BM25 融合
    fused = store.hybrid_search(
        query_text=query,
        query_vector=query_vector,
        filters=filters,
        limit=rerank_top_n,
        ef=ef,
        rrf_k=rrf_k,
    )
    # 重排序
    ranked = rerank_evidence(query=query, candidates=fused, reranker=reranker, top_n=rerank_top_n)
    return ranked[:limit]