"""Milvus 证据向量存储模块 - 管理 Milvus Collection 的创建、增删和混合检索。

核心组件：
    EvidenceRecord: 证据向量的数据模型。
    EvidenceFilter: 检索过滤条件（租户、权限范围、候选人等）。
    MilvusEvidenceStore: 封装 Milvus 的 Collection 生命周期和检索操作。

支持三种检索模式：
    - search: 纯向量检索 (Dense)。
    - sparse_search: BM25 关键词检索 (Sparse)。
    - hybrid_search: Dense + Sparse 融合检索，使用 RRF 算法合并排序。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence

from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker


class MilvusClientProtocol(Protocol):
    """Milvus 客户端协议，定义证据存储所需的最小操作集合。"""
    def has_collection(self, collection_name: str) -> bool: ...

    def create_schema(self, **kwargs: Any) -> Any: ...
    def prepare_index_params(self) -> Any: ...
    def create_collection(self, **kwargs: Any) -> Any: ...
    def upsert(self, **kwargs: Any) -> Any: ...
    def search(self, **kwargs: Any) -> Any: ...
    def hybrid_search(self, **kwargs: Any) -> Any: ...
    def delete(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class EvidenceRecord:
    """单条证据向量记录，包含文本内容、向量和元数据。"""
    chunk_id: str              # 分块全局唯一标识（主键）
    candidate_id: str          # 所属候选人 ID
    document_version_id: str   # 所属文档版本 ID
    tenant_id: str             # 租户标识
    document_type: str         # 文档类型（简历、绩效报告等）
    permission_scope: str      # 权限范围（hr_private 等）
    embedding_model: str       # 生成 embedding 的模型名称
    content_hash: str          # 内容哈希，用于去重/变更检测
    content: str               # 原始文本内容
    embedding: list[float]     # 向量表示
    parent_chunk_id: str | None   # 父级分块 ID（层级分块场景）
    page_start: int | None        # 原始文档起始页码
    page_end: int | None          # 原始文档结束页码
    is_active: bool = True        # 是否活跃（软删除标记）


@dataclass(frozen=True)
class EvidenceFilter:
    """证据检索过滤条件，用于限定检索范围。"""
    tenant_id: str                  # 必填，租户标识
    permission_scopes: list[str]    # 必填，允许的权限范围列表
    candidate_ids: list[str] | None = None      # 可选，限定候选人
    document_types: list[str] | None = None     # 可选，限定文档类型
    document_version_id: str | None = None      # 可选，限定文档版本


@dataclass(frozen=True)
class EvidenceSearchResult:
    """单条检索结果。"""
    chunk_id: str
    candidate_id: str
    content: str
    score: float          # 检索分数（Cosine 距离或 RRF 分数）
    metadata: dict[str, Any]  # 完整的元数据字段


def _literal(value: str) -> str:
    """将字符串转为 Milvus 表达式中使用的 JSON 字面量。"""
    return json.dumps(value, ensure_ascii=False)


def _in_expression(field: str, values: Sequence[str]) -> str:
    """构建 Milvus 的 `field in [...]` 过滤表达式。"""
    return f"{field} in [{', '.join(_literal(value) for value in values)}]"


def build_filter_expression(filters: EvidenceFilter) -> str:
    """根据过滤条件构建 Milvus 标量过滤表达式字符串。

    Args:
        filters: 证据过滤条件。

    Returns:
        Milvus 兼容的过滤表达式。

    Raises:
        ValueError: 缺少必需的 tenant_id 或 permission_scopes。
    """
    if not filters.tenant_id:
        raise ValueError("tenant_id is required")
    if not filters.permission_scopes:
        raise ValueError("permission_scopes must not be empty")
    parts = [
        f"tenant_id == {_literal(filters.tenant_id)}",
        _in_expression("permission_scope", filters.permission_scopes),
    ]
    if filters.candidate_ids:
        parts.append(_in_expression("candidate_id", filters.candidate_ids))
    if filters.document_types:
        parts.append(_in_expression("document_type", filters.document_types))
    if filters.document_version_id:
        parts.append(f"document_version_id == {_literal(filters.document_version_id)}")
    parts.append("is_active == true")
    return " and ".join(parts)


class MilvusEvidenceStore:
    """Milvus 证据向量存储，管理 Collection 的完整生命周期。

    Collection 设计：
        - chunk_id (VARCHAR, 主键): 分块唯一标识
        - embedding (FLOAT_VECTOR): Dense 向量 (HNSW 索引，Cosine 距离)
        - content_sparse (SPARSE_FLOAT_VECTOR): BM25 稀疏向量 (自动生成)
        - 标量字段: candidate_id, tenant_id, document_type, permission_scope 等
        - 支持 Dense + Sparse 混合检索 (RRF 融合)
    """

    # 检索时返回的所有字段
    output_fields = [
        "chunk_id",
        "candidate_id",
        "document_version_id",
        "tenant_id",
        "document_type",
        "permission_scope",
        "embedding_model",
        "content_hash",
        "content",
        "parent_chunk_id",
        "page_start",
        "page_end",
    ]

    def __init__(
        self,
        *,
        client: MilvusClientProtocol,
        collection_name: str,
        dimension: int,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.dimension = dimension

    @classmethod
    def connect(cls, *, uri: str, token: str, collection_name: str, dimension: int) -> "MilvusEvidenceStore":
        """工厂方法：连接 Milvus 并创建 EvidenceStore 实例。"""
        return cls(
            client=MilvusClient(uri=uri, token=token),
            collection_name=collection_name,
            dimension=dimension,
        )

    def ensure_collection(self) -> None:
        """确保 Collection 存在；若不存在则创建，包含 Schema、索引和 BM25 函数。

        创建的索引：
            - embedding: HNSW (M=16, efConstruction=128), Cosine 距离
            - content_sparse: SPARSE_INVERTED_INDEX, BM25 度量
        """
        if self.client.has_collection(self.collection_name):
            return
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=36)
        schema.add_field(field_name="candidate_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="document_version_id", datatype=DataType.VARCHAR, max_length=36)
        schema.add_field(field_name="tenant_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="document_type", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="permission_scope", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="embedding_model", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="content_hash", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},
        )
        schema.add_field(field_name="parent_chunk_id", datatype=DataType.VARCHAR, max_length=36, nullable=True)
        schema.add_field(field_name="page_start", datatype=DataType.INT64, nullable=True)
        schema.add_field(field_name="page_end", datatype=DataType.INT64, nullable=True)
        schema.add_field(field_name="is_active", datatype=DataType.BOOL)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.dimension)
        schema.add_field(field_name="content_sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name="content_bm25",
                function_type=FunctionType.BM25,
                input_field_names=["content"],
                output_field_names=["content_sparse"],
            )
        )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_name="evidence_embedding_hnsw",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 128},
        )
        index_params.add_index(
            field_name="content_sparse",
            index_name="evidence_content_bm25",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level="Bounded",
        )

    def upsert(self, records: Sequence[EvidenceRecord]) -> int:
        """批量插入或更新证据向量记录。

        Args:
            records: 证据记录序列。

        Returns:
            插入的记录数。

        Raises:
            ValueError: 嵌入向量维度与 Collection 不匹配时抛出。
        """
        if not records:
            return 0
        for record in records:
            if len(record.embedding) != self.dimension:
                raise ValueError(
                    f"embedding dimension {len(record.embedding)} does not match collection dimension {self.dimension}"
                )
        self.client.upsert(collection_name=self.collection_name, data=[asdict(record) for record in records])
        return len(records)

    def _to_results(self, rows: Any) -> list[EvidenceSearchResult]:
        """将 Milvus 原始搜索结果转换为 EvidenceSearchResult 列表。"""
        results: list[EvidenceSearchResult] = []
        for hit in rows[0] if rows else []:
            entity = dict(hit.get("entity") or {})
            results.append(
                EvidenceSearchResult(
                    chunk_id=str(hit.get("id") or entity.get("chunk_id", "")),
                    candidate_id=str(entity.get("candidate_id", "")),
                    content=str(entity.get("content", "")),
                    score=float(hit.get("distance", 0.0)),
                    metadata=entity,
                )
            )
        return results

    def search(
        self,
        query_vector: list[float],
        *,
        filters: EvidenceFilter,
        limit: int = 10,
        ef: int = 80,
        consistency_level: str = "Bounded",
    ) -> list[EvidenceSearchResult]:
        """Dense 向量检索：使用 HNSW 索引进行近似最近邻搜索。

        Args:
            query_vector: 查询向量。
            filters: 过滤条件。
            limit: 返回结果数。
            ef: HNSW ef 搜索参数。
            consistency_level: Milvus 一致性级别。

        Returns:
            EvidenceSearchResult 列表。

        Raises:
            ValueError: 查询向量维度不匹配时抛出。
        """
        if len(query_vector) != self.dimension:
            raise ValueError(
                f"query dimension {len(query_vector)} does not match collection dimension {self.dimension}"
            )
        rows = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            anns_field="embedding",
            filter=build_filter_expression(filters),
            limit=limit,
            output_fields=self.output_fields,
            search_params={"metric_type": "COSINE", "params": {"ef": ef}},
            consistency_level=consistency_level,
        )
        return self._to_results(rows)

    def sparse_search(
        self,
        query_text: str,
        *,
        filters: EvidenceFilter,
        limit: int = 10,
        consistency_level: str = "Bounded",
    ) -> list[EvidenceSearchResult]:
        """Sparse 关键词检索：使用 BM25 进行关键词匹配搜索。

        Args:
            query_text: 查询文本（自然语言）。
            filters: 过滤条件。
            limit: 返回结果数。
            consistency_level: Milvus 一致性级别。

        Returns:
            EvidenceSearchResult 列表。
        """
        rows = self.client.search(
            collection_name=self.collection_name,
            data=[query_text],
            anns_field="content_sparse",
            filter=build_filter_expression(filters),
            limit=limit,
            output_fields=self.output_fields,
            search_params={"metric_type": "BM25"},
            consistency_level=consistency_level,
        )
        return self._to_results(rows)

    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        *,
        filters: EvidenceFilter,
        limit: int = 10,
        ef: int = 80,
        rrf_k: int = 60,
        consistency_level: str = "Bounded",
    ) -> list[EvidenceSearchResult]:
        """混合检索：同时执行 Dense 和 Sparse 检索，使用 RRF 算法融合排序。

        RRF (Reciprocal Rank Fusion): 将 Dense 向量检索和 BM25 关键词检索
        的结果通过倒数排名融合，k 参数控制排名衰减速度。

        Args:
            query_text: 查询文本（用于 BM25）。
            query_vector: 查询向量（用于 Dense 检索）。
            filters: 过滤条件。
            limit: 返回结果数。
            ef: HNSW ef 搜索参数。
            rrf_k: RRF 算法参数 k。
            consistency_level: Milvus 一致性级别。

        Returns:
            混合检索后的 EvidenceSearchResult 列表。

        Raises:
            ValueError: 查询向量维度不匹配时抛出。
        """
        if len(query_vector) != self.dimension:
            raise ValueError(
                f"query dimension {len(query_vector)} does not match collection dimension {self.dimension}"
            )
        filter_expression = build_filter_expression(filters)
        dense_req = AnnSearchRequest(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": ef}},
            limit=limit * 2,
            filter=filter_expression,
        )
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="content_sparse",
            param={"metric_type": "BM25"},
            limit=limit * 2,
            filter=filter_expression,
        )
        rows = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=rrf_k),
            limit=limit,
            output_fields=self.output_fields,
            consistency_level=consistency_level,
        )
        return self._to_results(rows)

    def delete_version(self, *, tenant_id: str, document_version_id: str) -> None:
        """删除指定文档版本的所有证据向量记录。

        Args:
            tenant_id: 租户标识。
            document_version_id: 文档版本 ID。
        """
        if not self.client.has_collection(self.collection_name):
            return
        expression = (
            f"tenant_id == {_literal(tenant_id)} and "
            f"document_version_id == {_literal(document_version_id)}"
        )
        self.client.delete(collection_name=self.collection_name, filter=expression)