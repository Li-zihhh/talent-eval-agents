"""人才决策运行时模块 - 组装请求理解、评估和报告三个子图为完整的人才决策图。

将 talent_request_graph、talent_evaluation_runtime (evaluation + report) 
三个独立编译的 LangGraph 子图组合为一个完整的决策引擎。
"""

from __future__ import annotations

from app.talent_decision_engine import build_talent_decision_engine
from app.talent_evaluation_runtime import graph as evaluation_graph
from app.talent_evaluation_runtime import report_graph
from app.talent_request_graph import graph as request_graph


def build_runtime_decision_graph(*, checkpointer=None):
    """构建整合的决策图，供 Agent Server 或 FastAPI run 服务使用。

    将请求理解图、评估图和报告图组装到一个完整的 StateGraph 中，
    支持 checkpointer 用于 LangGraph 的中断/恢复。

    Args:
        checkpointer: LangGraph 检查点保存器（如 PostgresSaver）。

    Returns:
        编译后的完整决策 StateGraph。
    """
    return build_talent_decision_engine(
        request_graph=request_graph,
        evaluation_graph=evaluation_graph,
        report_graph=report_graph,
        checkpointer=checkpointer,
    )


# Agent Server 提供父级 checkpointer，FastAPI run 服务调用时需要传入
# build_runtime_decision_graph with its own checkpointer so interrupt/resume uses
# the same graph thread instead of an API-layer simulation.
graph = build_runtime_decision_graph()