"""人才决策图模块 - 定义完整决策流程的 LangGraph StateGraph 核心结构。

定义以下核心类型：
    DecisionContext: 决策图的共享状态，贯穿整个工作流
    TalentRequest: 用户请求的结构化表示
    DecisionStatus: 决策流程的状态机

提供图构建函数和节点间消息传递机制。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal

from langchain.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.types import Command
from typing_extensions import TypedDict


DecisionStatus = Literal[
    "received",           # 已接收请求
    "request_ready",      # 请求解析完成
    "candidates_ready",   # 候选人已确定
    "drafting",           # 正在生成评估
    "completed",          # 评估完成
    "no_candidates",      # 未找到候选人
    "failed",             # 评估失败
]


class TalentRequest(TypedDict):
    """用户的人才评估请求。"""
    original_text: str
    task_type: Literal["evaluate_and_recommend"] # 任务类型，当前仅支持评估_and推荐


class EvaluationResult(TypedDict):
    """人才评估结果。"""
    candidate_id: str
    status: Literal["placeholder"]


class TalentDecisionInput(TypedDict):
    """人才评估决策图的输入。"""
    messages: list[AnyMessage]
    request_text: str


class TalentDecisionOutput(TypedDict):
    """人才评估决策图的输出。"""
    status: DecisionStatus
    candidate_ids: list[str]
    evaluations: list[EvaluationResult]
    report: str
    errors: list[str]


class TalentDecisionState(TypedDict, total=False):
    """人才评估决策图的状态。"""
    messages: Annotated[list[AnyMessage], add_messages]
    request_text: str
    request: TalentRequest
    candidate_ids: list[str]
    evidence_refs: Annotated[list[str], list.__add__] # 评估引用的证据参考
    evaluations: Annotated[list[EvaluationResult], list.__add__] # 评估结果
    report: str
    status: DecisionStatus
    errors: Annotated[list[str], list.__add__]
    retry_count: int


@dataclass(frozen=True)
class DecisionContext:
    """ 决策图的共享权限上下文。包含租户 ID、权限范围、运行 ID、对话 ID、轮次号。"""
    tenant_id: str
    permission_scopes: tuple[str, ...]
    run_id: str = ""
    conversation_id: str = ""
    turn_no: int = 0

# 定义候选人提供器类型
CandidateProvider = Callable[[TalentRequest, DecisionContext], list[str]]


def _receive_request(
    state: TalentDecisionState,
    runtime: Runtime[DecisionContext],
) -> Command[Literal["prepare_request"]]:
    """接收请求节点，验证租户 ID 和请求文本。"""
    if not runtime.context.tenant_id.strip():
        raise ValueError("Runtime Context 中的 tenant_id 不能为空")
    request_text = state["request_text"].strip()
    if not request_text:
        raise ValueError("request_text 不能为空")
    return Command(
        update={"request_text": request_text, "status": "received", "errors": []},
        goto="prepare_request",
    )


def _prepare_request(state: TalentDecisionState) -> dict:
    """请求准备节点，将原始请求文本转换为结构化 TalentRequest。"""
    return {
        "request": {
            "original_text": state["request_text"],
            "task_type": "evaluate_and_recommend",
        },
        "status": "request_ready",
    }


def _candidate_node(candidate_provider: CandidateProvider):
    """候选人节点，根据请求和上下文检索候选人。"""
    def retrieve_candidates(
        state: TalentDecisionState,
        runtime: Runtime[DecisionContext],
    ) -> dict:
        # 闭包，将 candidate_provider 作为参数，返回一个函数，用于检索候选人
        candidate_ids = candidate_provider(state["request"], runtime.context)
        return {"candidate_ids": candidate_ids, "status": "candidates_ready"}

    return retrieve_candidates


def _route_candidates(state: TalentDecisionState) -> Literal["evaluate", "no_candidates"]:
    return "evaluate" if state.get("candidate_ids") else "no_candidates"


def _evaluate(state: TalentDecisionState) -> dict:
    """评估节点，根据候选人列表生成评估结果。"""
    return {
        "evaluations": [
            {"candidate_id": candidate_id, "status": "placeholder"}
            for candidate_id in state["candidate_ids"]
        ]
    }


def _compose_report(state: TalentDecisionState) -> dict:
    """报告合成节点，根据评估结果生成报告。"""
    candidate_text = "、".join(item["candidate_id"] for item in state["evaluations"])
    return {"report": f"候选人评估占位结果：{candidate_text}", "status": "drafting"}


def _validate_report(state: TalentDecisionState) -> dict:
    """报告验证节点，检查报告是否为空。"""
    if state.get("report"):
        return {"status": "completed"}
    retries = state.get("retry_count", 0) + 1
    if retries > 1:
        return {"retry_count": retries, "status": "failed", "errors": ["报告为空"]}
    return {"retry_count": retries}


def _route_report(state: TalentDecisionState) -> Literal["retry", "done"]:
    """报告路由节点，根据状态判断是否需要重试或完成。"""
    if state.get("status") in {"completed", "failed"}:
        return "done"
    return "retry"


def _no_candidates(_: TalentDecisionState) -> dict:
    """没有候选节点，处理没有检索到候选的情况。"""
    return {
        "evaluations": [],
        "report": "当前条件下未检索到候选人。",
        "status": "no_candidates",
    }


def build_talent_decision_graph(candidate_provider: CandidateProvider):
    """ 构建人才评估决策图。 """
    builder = StateGraph(
        TalentDecisionState,
        context_schema=DecisionContext,
        input_schema=TalentDecisionInput,
        output_schema=TalentDecisionOutput,
    )
    builder.add_node("receive_request", _receive_request)
    builder.add_node("prepare_request", _prepare_request)
    builder.add_node("retrieve_candidates", _candidate_node(candidate_provider))
    builder.add_node("evaluate", _evaluate)
    builder.add_node("compose_report", _compose_report)
    builder.add_node("validate_report", _validate_report)
    builder.add_node("no_candidates", _no_candidates)
    builder.add_edge(START, "receive_request")
    builder.add_edge("prepare_request", "retrieve_candidates")
    builder.add_conditional_edges("retrieve_candidates", _route_candidates)
    builder.add_edge("evaluate", "compose_report")
    builder.add_edge("compose_report", "validate_report")
    builder.add_conditional_edges(
        "validate_report",
        _route_report,
        {"retry": "compose_report", "done": END},
    )
    builder.add_edge("no_candidates", END)
    return builder.compile()


class MissingReducerState(TypedDict):
    items: list[str]


def build_missing_reducer_demo():
    """缺失 reducer 演示图"""
    builder = StateGraph(MissingReducerState)
    builder.add_node("branch_a", lambda _: {"items": ["A"]})
    builder.add_node("branch_b", lambda _: {"items": ["B"]})
    builder.add_edge(START, "branch_a")
    builder.add_edge(START, "branch_b")
    builder.add_edge("branch_a", END)
    builder.add_edge("branch_b", END)
    return builder.compile()


class ReducerState(TypedDict):
    items: Annotated[list[str], list.__add__]


def build_reducer_demo():
    builder = StateGraph(ReducerState)
    builder.add_node("branch_a", lambda _: {"items": ["A"]})
    builder.add_node("branch_b", lambda _: {"items": ["B"]})
    builder.add_edge(START, "branch_a")
    builder.add_edge(START, "branch_b")
    builder.add_edge("branch_a", END)
    builder.add_edge("branch_b", END)
    return builder.compile()