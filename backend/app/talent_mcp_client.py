"""MCP 工具客户端模块 - 动态发现并构建 MCP (Model Context Protocol) 工具的 LangGraph 执行图。

通过 MCP 协议发现外部工具服务，将工具封装为 LangChain Tool，
并构建最小化的 LangGraph 图用于工具调用。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


def build_mcp_tool_graph(tools: Sequence[BaseTool]) -> Any:
    """构建最小的 LangGraph 图，用于执行发现的 MCP 工具。

    图结构：START -> mcp_tools (ToolNode) -> END
    直接将工具列表包装为 ToolNode 并编译为可执行图。

    Args:
        tools: 从 MCP 服务发现的 LangChain Tool 列表。

    Returns:
        编译后的 StateGraph。
    """
    builder = StateGraph(MessagesState)
    builder.add_node("mcp_tools", ToolNode(list(tools)))
    builder.add_edge(START, "mcp_tools")
    builder.add_edge("mcp_tools", END)
    return builder.compile()