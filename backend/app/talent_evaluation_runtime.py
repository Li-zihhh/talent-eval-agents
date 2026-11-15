"""人才评估运行时装配模块 - 初始化所有依赖并构建评估子图。

负责将模型、向量存储、重排序器等基础设施注入评估流水线：
    - evaluation graph: 评估维度生成 + 并行证据检索与评估
    - report graph: 评估报告合成

所有依赖（ChatModel、Embedding、Reranker、MilvusStore）在此统一初始化。
"""

from __future__ import annotations

import logging

from app.database import SessionLocal
from app.evidence_citations import load_pack_sources
from app.evidence_pack import model_extractor
from app.evidence_index_service import get_evidence_store
from app.hybrid_search_service import hybrid_search_evidence
from app.milvus_store import EvidenceFilter
from app.model_provider import get_chat_model, get_embedding_model
from app.reranker import get_reranker
from app.talent_evaluation_dispatch import (
    build_branch_evidence_provider,
    build_evidence_branch_worker,
    build_query_plan_candidate_provider,
    build_structured_dimension_generator,
    build_talent_evaluation_dispatch_graph,
)
from app.talent_evaluation_report import (
    build_structured_consistency_reviewer,
    build_structured_dimension_scorer,
    build_structured_report_writer,
    build_talent_evaluation_report_graph,
)
from app.talent_tools import TalentToolService, ToolExecutor, ToolPolicy

logger = logging.getLogger(__name__)


def build_runtime_graph():
    """构建人才评估运行时图，包含评估维度生成、并行证据检索与评估、评估报告合成等子图。"""
    logger.info(
        "event=runtime_graph_build_started function=build_runtime_graph"
    )
    model = get_chat_model(temperature=0)
    embedder = get_embedding_model()
    reranker = get_reranker()
    if model is None:
        raise RuntimeError("模型未配置，请先设置项目 .env 中的 DASHSCOPE_API_KEY")
    if embedder is None:
        raise RuntimeError("Embedding 服务未配置")
    if reranker is None:
        raise RuntimeError("Rerank 服务未配置")
    store = get_evidence_store()

    def search(*, query, candidate_ids, tenant_id, permission_scopes):
        """执行混合搜索，返回与查询最相关的8个的证据包。"""
        logger.info(
            "event=hybrid_search_started function=search candidate_ids=%s "
            "query_chars=%s permission_scope_count=%s limit=8 rerank_top_n=20",
            candidate_ids, len(query), len(permission_scopes),
        )
        results = hybrid_search_evidence(
            query=query,
            filters=EvidenceFilter(
                tenant_id=tenant_id,
                permission_scopes=permission_scopes,
                candidate_ids=candidate_ids,
            ),
            store=store,
            embedder=embedder,
            reranker=reranker,
            limit=8,
            rerank_top_n=20,
        )
        logger.info(
            "event=hybrid_search_completed function=search candidate_ids=%s "
            "hit_count=%s chunk_ids=%s",
            candidate_ids, len(results), [item.chunk_id for item in results],
        )
        return results

    def load_sources(chunks, *, tenant_id, permission_scopes):
        """从数据库加载证据包的来源信息。"""
        logger.info(
            "event=source_loading_started function=load_sources "
            "requested_chunk_count=%s chunk_ids=%s permission_scope_count=%s",
            len(chunks), [item["chunk_id"] for item in chunks],
            len(permission_scopes),
        )
        with SessionLocal() as db:
            sources = load_pack_sources(
                db,
                chunks,
                tenant_id=tenant_id,
                permission_scopes=permission_scopes,
            )
        logger.info(
            "event=source_loading_completed function=load_sources "
            "requested_chunk_count=%s loaded_source_count=%s loaded_chunk_ids=%s",
            len(chunks), len(sources), [item["chunk_id"] for item in sources],
        )
        return sources

    # 装配人才工具服务，作为评估分支 worker 的统一入口。
    # 注入证据提供者，其内部封装了"混合检索 → 来源加载 → LLM 事实提取"
    # 三步流水线：search 负责 Milvus 向量检索 + 重排序，
    # load_sources 负责数据库原文加载 + 权限二次鉴权，
    # extract 负责调用 LLM 从原文提取结构化证据事实（最多重试 3 次）。
    service = TalentToolService(
        session_factory=SessionLocal,
        evidence_provider=build_branch_evidence_provider(
            search=search,
            load_sources=load_sources,
            extract=model_extractor(
                model,
                max_attempts=3,
                backoff_seconds=0.5,
            ),
        ),
    )
    runtime_graph = build_talent_evaluation_dispatch_graph(
        dimension_generator=build_structured_dimension_generator(lambda: model),
        candidate_provider=build_query_plan_candidate_provider(service),
        branch_worker=build_evidence_branch_worker(
            service,
            ToolExecutor(
                tool_policies={
                    "search_candidate_evidence": ToolPolicy(
                        max_attempts=1,
                        timeout_seconds=None,
                    )
                }
            ),
        ),
        max_work_items=24,
    )
    logger.info(
        "event=runtime_graph_build_completed function=build_runtime_graph "
        "max_work_items=24 evidence_total_timeout=disabled "
        "llm_tool_timeout=disabled"
    )
    return runtime_graph


def build_report_runtime_graph():
    """构建人才评估报告运行时图，包含评估维度评分、一致性检查、报告生成等子图。"""
    logger.info(
        "event=report_runtime_graph_build_started "
        "function=build_report_runtime_graph"
    )
    model = get_chat_model(temperature=0)
    if model is None:
        raise RuntimeError("模型未配置，请先设置项目 .env 中的 DASHSCOPE_API_KEY")
    runtime_graph = build_talent_evaluation_report_graph(
        dimension_scorer=build_structured_dimension_scorer(lambda: model),
        consistency_reviewer=build_structured_consistency_reviewer(lambda: model),
        report_writer=build_structured_report_writer(lambda: model),
        max_scoring_concurrency=12,
        max_consistency_concurrency=12,
    )
    logger.info(
        "event=report_runtime_graph_build_completed "
        "function=build_report_runtime_graph"
    )
    return runtime_graph


# Module-level lazy graph singletons.
_graph = None
_report_graph = None


def __getattr__(name: str):
    if name == "graph":
        global _graph
        if _graph is None:
            _graph = build_runtime_graph()
        return _graph
    if name == "report_graph":
        global _report_graph
        if _report_graph is None:
            _report_graph = build_report_runtime_graph()
        return _report_graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")