"""请求理解模块 - 构建用户需求理解的 LangGraph 图。

核心流程（request graph）：
    1. interpret_input: 解析用户自然语言需求，提取任务类型、候选人过滤条件、特殊要求
    2. clarify_if_needed: 如信息不足则中断请求澄清（interrupt）
    3. filter_candidates: 根据过滤条件从数据库筛选候选人
    4. route_by_status: 根据结果路由到后续评估或返回空结果

支持评估用例录制（capture_call）以便回归测试和 Judge 评分。
"""

from __future__ import annotations

from app.evaluation_case import capture_call, EvaluationModule, safe_record

from collections.abc import Callable
import inspect
from operator import add
from time import perf_counter
from typing import Annotated, Any, Literal

from langchain.messages import AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import TypedDict

from app.database import SessionLocal
from app.config import get_settings
from app.evaluation_case import (
    EvaluationCaseDraft,
    SqlEvaluationCaseStore,
    build_request_understanding_case,
    recent_request_messages,
)
from app.model_provider import get_chat_model
from app.structured_output import invoke_structured
from app.query_plan import (
    Clarification,
    FilterCondition,
    QueryPlan,
    SemanticRequirement,
    TaskType,
    filter_dsl_catalog,
)
from app.talent_decision_graph import DecisionContext
from app.talent_tools import TalentToolContext, TalentToolService


# 输入模式：detailed_requirement=用户给出完整需求描述，job_name=仅给出岗位名称
RequestInputMode = Literal["detailed_requirement", "job_name"]
# 请求处理状态机：从 received 起步，经理解、岗位匹配、计划构建，
# 期间可能进入澄清、修订、取消等分支，最终到达 candidates_ready 或 cancelled
RequestStatus = Literal[
    "received",
    "request_understood",
    "jobs_found",
    "job_confirmed",
    "no_job_match",
    "plan_ready",
    "clarification_required",
    "candidates_empty",
    "candidates_ready",
    "plan_revised",
    "request_revised",
    "cancelled",
]


class JobMatch(TypedDict):
    job_code: str
    name: str
    match_score: float
    match_type: Literal["exact", "contains", "fuzzy"]
    version: int
    content: str


class TargetJob(TypedDict):
    job_code: str
    name: str
    version: int


class TalentRequestDraft(BaseModel):
    """人才请求草稿：任务理解节点解析用户输入后产出的结构化中间表示，是后续 QueryPlan 的原料。"""
    input_mode: RequestInputMode
    job_query: str | None = None
    filters: list[FilterCondition] = Field(default_factory=list)
    semantic_requirements: list[SemanticRequirement] = Field(default_factory=list)
    evaluation_preferences: list[str] = Field(default_factory=list)
    clarifications: list[Clarification] = Field(default_factory=list)


class TalentRequestRevision(BaseModel):
    """一次修订记录：draft 内容 + 修订动作（initial/add/update/remove）与修订说明。"""
    draft: TalentRequestDraft
    revision_action: Literal["initial", "add", "update", "remove"] = "initial"
    revision_summary: str = "建立初始人才要求"


class TalentRequestUnderstanding(TalentRequestRevision):
    """首轮任务理解结果：在 Revision 基础上额外携带会话标题。"""
    conversation_title: str = Field(min_length=1, max_length=80)


class TalentRequestInput(TypedDict, total=False):
    """图入口输入：用户消息（add_messages 自动合并多轮）或直接传入的请求文本。"""
    messages: Annotated[list[AnyMessage], add_messages]
    request_text: str
    talent_request: dict[str, Any]
    conversation_title: str


class TalentRequestOutput(TypedDict):
    """图出口输出：供下游（决策图/评估图）消费的最终状态。"""
    messages: Annotated[list[AnyMessage], add_messages]
    status: RequestStatus
    input_mode: RequestInputMode
    job_matches: list[JobMatch]
    selected_job: JobMatch | None
    talent_request: dict[str, Any]
    query_plan: dict[str, Any]
    candidate_ids: list[str]
    condition_revisions: list[dict[str, Any]]
    plan_version: int
    clarifications: list[dict[str, str]]
    errors: list[str]
    conversation_title: str
    revision_action: str
    revision_summary: str


class TalentRequestState(TypedDict, total=False):
    """请求图内部状态，贯穿所有节点；condition_revisions 用 add 归约以保留完整修订历史。"""
    messages: Annotated[list[AnyMessage], add_messages]
    request_text: str
    input_mode: RequestInputMode
    draft: dict[str, Any]
    job_matches: list[JobMatch]
    selected_job: JobMatch | None
    talent_request: dict[str, Any]
    query_plan: dict[str, Any]
    candidate_ids: list[str]
    condition_revisions: Annotated[list[dict[str, Any]], add]  # 修订历史
    plan_version: int
    clarifications: list[dict[str, str]]
    status: RequestStatus
    errors: list[str]
    conversation_title: str
    revision_action: str
    revision_summary: str


# 可注入的策略函数签名：解释器、岗位查询、候选人筛选、用例录制
RequestInterpreter = Callable[..., TalentRequestUnderstanding | TalentRequestDraft]
JobLookup = Callable[[str, DecisionContext], list[JobMatch]]
CandidateFilter = Callable[[list[FilterCondition], DecisionContext], list[str]]
CaseRecorder = Callable[[EvaluationCaseDraft], Any]


def _receive_request(state: TalentRequestState, runtime: Runtime[DecisionContext]) -> dict[str, Any]:
    """入口节点：校验运行时上下文（租户/权限）与输入文本，初始化图状态。

    优先取 messages 最后一条作为请求文本；若无消息则回退到 request_text 并补一条 HumanMessage。
    """
    if not runtime.context.tenant_id.strip():
        raise ValueError("Runtime Context 中的 tenant_id 不能为空")
    if not runtime.context.permission_scopes:
        raise ValueError("Runtime Context 中的 permission_scopes 不能为空")
    messages = list(state.get("messages", []))
    request_text = str(state.get("request_text", "")).strip()
    if messages:
        request_text = str(messages[-1].content).strip()
    if not request_text:
        raise ValueError("request_text 不能为空")
    update: dict[str, Any] = {
        "request_text": request_text,
        "job_matches": [],
        "selected_job": None,
        "query_plan": {},
        "candidate_ids": [],
        "condition_revisions": [],
        "plan_version": 0,
        "clarifications": [],
        "status": "received",
        "errors": [],
    }
    if not messages:
        update["messages"] = [HumanMessage(content=request_text)]
    return update


def _call_request_interpreter(
    interpreter: RequestInterpreter,
    state: TalentRequestState,
    selected_job: JobMatch | None,
    context: DecisionContext,
    case_recorder: CaseRecorder | None,
) -> TalentRequestUnderstanding:
    """调用任务理解器，兼容新旧两种签名，并在成功/失败时分别录制评估用例（Case）。

    - 旧签名（<=2 参）：interpreter(request_text, selected_job)
    - 新签名（>2 参）：interpreter(messages, previous_request, selected_job, conversation_title)
    """
    started_at = perf_counter()
    try:
        parameters = inspect.signature(interpreter).parameters
        if len(parameters) <= 2:
            # 旧理解器调用
            legacy = interpreter(state["request_text"], selected_job)
            if isinstance(legacy, TalentRequestUnderstanding):
                # 旧理解器直接返回理解结果，无需转换
                understanding = legacy
            else:
                # 旧理解器返回 TalentRequestDraft，需要转换为 TalentRequestUnderstanding
                understanding = TalentRequestUnderstanding(
                    draft=TalentRequestDraft.model_validate(legacy),
                    conversation_title=state.get("conversation_title") or state["request_text"][:30],
                )
        else:
            # 新理解器调用
            result = interpreter(
                recent_request_messages(list(state.get("messages", []))),
                dict(state.get("talent_request", {})),
                selected_job,
                state.get("conversation_title", ""),
            )
            understanding = TalentRequestUnderstanding.model_validate(result)
    except Exception as exc:
        if case_recorder is not None and context.run_id:
            safe_record(case_recorder,
                build_request_understanding_case(
                    source_run_id=context.run_id,
                    conversation_id=context.conversation_id,
                    turn_no=context.turn_no,
                    messages=list(state.get("messages", [])),
                    previous_request=dict(state.get("talent_request", {})),
                    conversation_title=state.get("conversation_title", ""),
                    selected_job=selected_job,
                    output_payload={},
                    replay_context={
                        "tenant_id": context.tenant_id,
                        "permission_scopes": list(context.permission_scopes),
                    },
                    version_snapshot={
                        "prompt": "request-understanding-v1",
                        "model": get_settings().chat_model,
                    },
                    status="failed",
                    error_code=type(exc).__name__,
                    duration_ms=round((perf_counter() - started_at) * 1000),
                )
            )
        raise
    if case_recorder is not None and context.run_id:
        safe_record(case_recorder,
            build_request_understanding_case(
                source_run_id=context.run_id,
                conversation_id=context.conversation_id,
                turn_no=context.turn_no,
                messages=list(state.get("messages", [])),
                previous_request=dict(state.get("talent_request", {})),
                conversation_title=state.get("conversation_title", ""),
                selected_job=selected_job,
                output_payload=understanding.model_dump(mode="json"),
                replay_context={
                    "tenant_id": context.tenant_id,
                    "permission_scopes": list(context.permission_scopes),
                },
                version_snapshot={
                    "prompt": "request-understanding-v1",
                    "model": get_settings().chat_model,
                },
                duration_ms=round((perf_counter() - started_at) * 1000),
            )
        )
    return understanding


def _interpret_input(
    request_interpreter: RequestInterpreter,
    case_recorder: CaseRecorder | None,
):
    """节点工厂：把用户自然语言输入解析为结构化 draft，并确定输入模式与会话标题。"""

    def interpret_input(
        state: TalentRequestState,
        runtime: Runtime[DecisionContext],
    ) -> dict[str, Any]:
        understanding = _call_request_interpreter(
            request_interpreter,
            state,
            None,
            runtime.context,
            case_recorder,
        )
        draft = understanding.draft
        return {
            "draft": draft.model_dump(mode="json"),
            "input_mode": draft.input_mode,
            "status": "request_understood",
            "conversation_title": state.get("conversation_title") or understanding.conversation_title,
            "revision_action": understanding.revision_action,
            "revision_summary": understanding.revision_summary,
        }

    return interpret_input


def _route_input(state: TalentRequestState) -> Literal["lookup_jobs", "build_plan"]:
    """按输入模式路由：仅岗位名称走岗位匹配（lookup_jobs），详细需求直接构建计划（build_plan）。"""
    return "lookup_jobs" if state["input_mode"] == "job_name" else "build_plan"


def _lookup_jobs(job_lookup: JobLookup, case_recorder=None):
    """节点工厂：按岗位名称查询岗位匹配结果（含匹配类型），供后续精确命中/人工确认使用。"""

    def lookup_jobs(state: TalentRequestState, runtime: Runtime[DecisionContext]) -> dict[str, Any]:
        job_query = str(state["draft"].get("job_query") or state["request_text"])

        # 岗位匹配 Case 要在查询完成后、Graph 路由前采集。
        # 输出同时保存完整候选列表和本次路由结果，避免后续只看到最终岗位，无法判断问题来自名称匹配还是岗位确认分支。
        def match_and_route():
            matches = job_lookup(job_query, runtime.context)
            return {"matches": matches, "route": _route_job_matches({"job_matches": matches})}

        result = capture_call(match_and_route, context=runtime.context,
            module_key=EvaluationModule.JOB_MATCHING,
            input_payload={"job_query": job_query}, recorder=case_recorder,
            version_snapshot={"matching_rules": "job-lookup-v1"})
        matches = result["matches"]
        return {"job_matches": matches, "status": "jobs_found"}

    return lookup_jobs


def _route_job_matches(
    state: TalentRequestState,
) -> Literal["select_exact_job", "confirm_job", "no_job_match"]:
    """岗位匹配结果路由：唯一精确命中直接选用，多个候选需人工确认，无结果则进入 no_job_match。"""
    exact_matches = [item for item in state["job_matches"] if item["match_type"] == "exact"]
    if len(exact_matches) == 1:
        return "select_exact_job"
    if state["job_matches"]:
        return "confirm_job"
    return "no_job_match"


def _select_exact_job(state: TalentRequestState) -> dict[str, Any]:
    """唯一精确命中时直接选中该岗位，无需人工确认。"""
    selected = next(item for item in state["job_matches"] if item["match_type"] == "exact")
    return {"selected_job": selected, "status": "job_confirmed"}


def _confirm_job(state: TalentRequestState) -> dict[str, Any]:
    """中断节点：把多个匹配岗位交给用户确认（select 选中 / cancel 取消），并校验恢复数据。"""
    selection = interrupt(
        {
            "type": "job_selection",
            "question": "请选择本次人才评估使用的岗位 JD",
            "request_text": state["request_text"],
            "options": [
                {
                    "job_code": item["job_code"],
                    "name": item["name"],
                    "match_score": item["match_score"],
                    "match_type": item["match_type"],
                    "version": item["version"],
                }
                for item in state["job_matches"]
            ],
            "actions": ["select", "cancel"],
        }
    )
    if not isinstance(selection, dict):
        raise ValueError("恢复数据必须是 JSON 对象")
    if selection.get("action") == "cancel":
        return {"status": "cancelled"}
    if selection.get("action") != "select":
        raise ValueError("action 必须是 select 或 cancel")
    selected_code = selection.get("job_code")
    selected = next(
        (item for item in state["job_matches"] if item["job_code"] == selected_code),
        None,
    )
    if selected is None:
        raise ValueError("恢复数据中的 job_code 不在待确认岗位列表中")
    return {"selected_job": selected, "status": "job_confirmed"}


def _route_job_confirmation(state: TalentRequestState) -> str:
    """确认结果路由：已确认则继续编译岗位，已取消则直接结束。"""
    return "compile_selected_job" if state["status"] == "job_confirmed" else END


def _compile_selected_job(
    request_interpreter: RequestInterpreter,
    case_recorder: CaseRecorder | None,
):
    """节点工厂：携带已确认岗位重新调用理解器，把岗位 JD 与用户要求一起编译进 draft。"""

    def compile_selected_job(
        state: TalentRequestState,
        runtime: Runtime[DecisionContext],
    ) -> dict[str, Any]:
        understanding = _call_request_interpreter(
            request_interpreter,
            state,
            state["selected_job"],
            runtime.context,
            case_recorder,
        )
        draft = understanding.draft
        return {
            "draft": draft.model_dump(mode="json"),
            "input_mode": "job_name",
            "conversation_title": state.get("conversation_title") or understanding.conversation_title,
        }

    return compile_selected_job


def _build_plan(state: TalentRequestState) -> dict[str, Any]:
    """纯计算节点：把 draft 编译为可执行的 QueryPlan 与结构化 talent_request。

    若计划可执行则进入 plan_ready，否则进入 clarification_required（需要澄清）。
    """
    draft = TalentRequestDraft.model_validate(state["draft"])
    selected_job = state.get("selected_job")
    target_job: TargetJob | None = None
    if selected_job is not None:
        target_job = {
            "job_code": selected_job["job_code"],
            "name": selected_job["name"],
            "version": selected_job["version"],
        }

    talent_request = {
        "original_text": state["request_text"],
        "task_type": "evaluate_and_recommend",
        "source": state["input_mode"],
        "target_job": target_job,
        "hard_conditions": [item.model_dump(mode="json") for item in draft.filters],
        "semantic_conditions": [
            item.model_dump(mode="json") for item in draft.semantic_requirements
        ],
        "evaluation_preferences": draft.evaluation_preferences,
    }
    query_plan = QueryPlan(
        task_type=TaskType.FIND_TALENT,
        filters=draft.filters,
        semantic_requirements=draft.semantic_requirements,
        preferences=draft.evaluation_preferences,
        clarifications=draft.clarifications,
    )
    status: RequestStatus = "plan_ready" if query_plan.executable else "clarification_required"
    return {
        "talent_request": talent_request,
        "query_plan": query_plan.model_dump(mode="json"),
        "clarifications": [item.model_dump(mode="json") for item in draft.clarifications],
        "status": status,
    }


def _no_job_match(state: TalentRequestState) -> dict[str, Any]:
    """未匹配到任何岗位 JD 时生成澄清信息并结束（no_job_match）。"""
    clarification = {
        "expression": state["request_text"],
        "reason": "未找到可确认的岗位 JD",
    }
    return {
        "query_plan": {},
        "talent_request": {},
        "clarifications": [clarification],
        "status": "no_job_match",
    }


# ---------------------------------------------------------------------------
# 第 17 课：候选人检索与可恢复人机协同
# ---------------------------------------------------------------------------


def _route_after_plan(state: TalentRequestState) -> str:
    """计划构建后路由：需澄清则进入条件修订，可执行则进入候选人筛选。"""
    if state["status"] == "clarification_required":
        return "resolve_requirements"
    return "filter_candidates"


def _route_candidates(state: TalentRequestState) -> str:
    """候选人筛选后路由：候选人为空则进入需求修订，否则结束。"""
    if state["status"] == "candidates_empty":
        return "resolve_requirements"
    return END


def _route_resolution(state: TalentRequestState) -> str:
    """需求修订后路由：已修订则回到 interpret_input 重新理解，否则结束。"""
    if state["status"] == "request_revised":
        return "interpret_input"
    return END


def _await_decision(payload: dict[str, Any], parse: Callable[[Any], dict[str, Any]]) -> dict[str, Any]:
    """等待恢复数据并完成校验

    校验失败时不抛出异常，而是携带 resume_error 重新 interrupt，
    让 Thread 停留在暂停点等待下一次提交，避免节点失败导致线程卡死
    """
    decision = interrupt(payload)
    while True:
        try:
            return parse(decision)
        except ValueError as exc:
            decision = interrupt({**payload, "resume_error": str(exc)})


def _require_interaction(decision: Any, expected_id: str) -> dict[str, Any]:
    """校验恢复数据为 JSON 对象且 interaction_id 与当前暂停点匹配。"""
    if not isinstance(decision, dict):
        raise ValueError("恢复数据必须是 JSON 对象")
    if decision.get("interaction_id") != expected_id:
        raise ValueError(f"恢复数据与当前暂停点不匹配，期望 interaction_id={expected_id}")
    return decision


def _filter_candidates_node(candidate_filter: CandidateFilter):
    """节点工厂：用结构化过滤条件从数据库筛选候选人 ID。"""

    def filter_candidates(state: TalentRequestState, runtime: Runtime[DecisionContext]) -> dict[str, Any]:
        plan = QueryPlan.model_validate(state["query_plan"])
        candidate_ids = candidate_filter(plan.filters, runtime.context)
        if candidate_ids:
            return {"candidate_ids": candidate_ids, "status": "candidates_ready"}
        return {"candidate_ids": [], "status": "candidates_empty"}

    return filter_candidates


def _diagnose_filters(
    filters: list[FilterCondition],
    candidate_filter: CandidateFilter,
    context: DecisionContext,
) -> list[dict[str, Any]]:
    """leave-one-out 诊断：逐个移除条件，观察候选数变化，定位导致空集的条件"""
    diagnostics = []
    for index, condition in enumerate(filters):
        reduced = filters[:index] + filters[index + 1 :]
        hit_count = len(candidate_filter(reduced, context))
        diagnostics.append(
            {
                "field": str(condition.field),
                "operator": condition.operator,
                "value": condition.value,
                "candidates_without_condition": hit_count,
            }
        )
    return diagnostics


def _resolve_requirements():
    """节点工厂：处理两类中断（计划被澄清阻塞 / 候选人空集），等待用户修订或取消。

    用户修订后重置岗位匹配与候选人集，并将状态置为 request_revised 以便回到 interpret_input 重新理解。
    """

    def resolve_requirements(
        state: TalentRequestState,
        runtime: Runtime[DecisionContext],
    ) -> dict[str, Any]:
        plan_version = state.get("plan_version", 0)
        interaction_id = f"rr-v{plan_version}"
        reason_code = "plan_blocked" if state["status"] == "clarification_required" else "empty_candidates"
        filters = state.get("draft", {}).get("filters", [])
        if reason_code == "plan_blocked":
            blocking_reason = "当前要求存在歧义或冲突，无法生成可执行查询计划"
        else:
            rendered = "、".join(
                f"{item['field']} {item['operator']} {item['value']}" for item in filters
            )
            blocking_reason = f"使用以下结构化条件筛选后未命中候选人：{rendered}"
        selected_job = state.get("selected_job")
        if selected_job is not None:
            editable_request = selected_job["content"]
            request_source = "job_profile"
            selected_job_source: dict[str, Any] | None = {
                "job_code": selected_job["job_code"],
                "name": selected_job["name"],
                "version": selected_job["version"],
            }
        else:
            editable_request = state["request_text"]
            request_source = "user_request"
            selected_job_source = None
        payload = {
            "type": "requirement_revision",
            "interaction_id": interaction_id,
            "reason_code": reason_code,
            "blocking_reason": blocking_reason,
            "question": "请补充或修订人才要求后重新提交，或取消本次任务",
            "request_text": editable_request,
            "request_source": request_source,
            "selected_job": selected_job_source,
            "filters": filters,
            "semantic_requirements": state.get("draft", {}).get("semantic_requirements", []),
            "clarifications": state.get("clarifications", []),
            "actions": ["revise", "cancel"],
        }

        def parse(decision: Any) -> dict[str, Any]:
            """解析用户修订或取消请求。"""
            decision = _require_interaction(decision, interaction_id)
            action = decision.get("action")
            if action == "cancel":
                return {"action": "cancel"}
            if action != "revise":
                raise ValueError("action 必须是 revise 或 cancel")
            revision_text = decision.get("revision_text")
            if not isinstance(revision_text, str) or not revision_text.strip():
                raise ValueError("revise 动作必须携带非空 revision_text")
            return {"action": "revise", "revision_text": revision_text.strip()}

        parsed = _await_decision(payload, parse)
        if parsed["action"] == "cancel":
            return {"status": "cancelled"}

        revision_text = parsed["revision_text"]
        revision = {
            "interaction_id": interaction_id,
            "action": "revise",
            "reason_code": reason_code,
            "revision_text": revision_text,
            "plan_version": plan_version + 1,
        }
        return {
            "request_text": revision_text,
            "messages": [HumanMessage(content=revision_text)],
            "selected_job": None,
            "job_matches": [],
            "candidate_ids": [],
            "condition_revisions": [revision],
            "plan_version": plan_version + 1,
            "status": "request_revised",
        }

    return resolve_requirements


def _apply_relaxations(
    current: list[FilterCondition],
    relaxations: list[dict[str, Any]],
) -> tuple[list[FilterCondition], list[dict[str, Any]]]:
    """对过滤条件应用放宽操作：remove 删除条件、update 替换条件，返回新条件与变更记录。"""
    updated = list(current)
    changes: list[dict[str, Any]] = []
    for item in relaxations:
        field = item.get("field")
        op = item.get("op")
        index = next((i for i, cond in enumerate(updated) if str(cond.field) == field), None)
        if index is None:
            raise ValueError(f"放宽目标条件不存在：{field}")
        if op == "remove":
            removed = updated.pop(index)
            changes.append({"field": field, "op": "remove", "removed": removed.model_dump(mode="json")})
        elif op == "update":
            replacement = FilterCondition.model_validate(
                {"field": field, "operator": item.get("operator"), "value": item.get("value")}
            )
            previous = updated[index]
            updated[index] = replacement
            changes.append(
                {
                    "field": field,
                    "op": "update",
                    "previous": previous.model_dump(mode="json"),
                    "current": replacement.model_dump(mode="json"),
                }
            )
        else:
            raise ValueError(f"不支持的放宽操作：{op}")
    return updated, changes


def _resolve_conditions(state: TalentRequestState) -> dict[str, Any]:
    """中断节点：处理无法直接执行的查询条件（clarifications），等待用户修订 filters 或语义要求。"""
    plan_version = state.get("plan_version", 0)
    interaction_id = f"cc-v{plan_version}"
    draft = dict(state["draft"])
    payload = {
        "type": "condition_clarification",
        "interaction_id": interaction_id,
        "question": "以下条件无法直接执行，请修订后重新提交，或取消本次任务",
        "clarifications": state["clarifications"],
        "current_filters": draft.get("filters", []),
        "current_semantic_requirements": draft.get("semantic_requirements", []),
        "actions": ["revise", "cancel"],
    }

    def parse(decision: Any) -> dict[str, Any]:
        decision = _require_interaction(decision, interaction_id)
        action = decision.get("action")
        if action == "cancel":
            return {"action": "cancel"}
        if action != "revise":
            raise ValueError("action 必须是 revise 或 cancel")
        parsed: dict[str, Any] = {"action": "revise"}
        if "filters" in decision:
            try:
                parsed["filters"] = [
                    FilterCondition.model_validate(item) for item in decision["filters"]
                ]
            except ValidationError as exc:
                raise ValueError(f"filters 校验失败：{exc.errors()[0]['msg']}") from exc
        if "semantic_requirements" in decision:
            items = decision["semantic_requirements"]
            if not isinstance(items, list) or any(not isinstance(item, dict) or not item.get("query") for item in items):
                raise ValueError("semantic_requirements 的每一项都必须包含 query")
            parsed["semantic_requirements"] = [
                SemanticRequirement(
                    requirement_id=f"S{index + 1}",
                    query=str(item["query"]),
                    required=bool(item.get("required", True)),
                )
                for index, item in enumerate(items)
            ]
        if "filters" not in parsed and "semantic_requirements" not in parsed:
            raise ValueError("revise 动作必须携带 filters 或 semantic_requirements")
        return parsed

    parsed = _await_decision(payload, parse)
    if parsed["action"] == "cancel":
        return {"status": "cancelled"}

    changes: list[dict[str, Any]] = []
    invalidates: set[str] = set()
    if "filters" in parsed:
        draft["filters"] = [item.model_dump(mode="json") for item in parsed["filters"]]
        changes.append({"kind": "filters", "count": len(parsed["filters"])})
        invalidates.add("candidate_set")
    if "semantic_requirements" in parsed:
        draft["semantic_requirements"] = [
            item.model_dump(mode="json") for item in parsed["semantic_requirements"]
        ]
        changes.append({"kind": "semantic_requirements", "count": len(parsed["semantic_requirements"])})
        invalidates.update(["evidence_retrieval", "evaluation_dimensions", "evaluation_report"])

    resolved = list(draft.get("clarifications", []))
    draft["clarifications"] = []
    revision = {
        "interaction_id": interaction_id,
        "action": "revise",
        "changes": changes,
        "resolved_clarifications": resolved,
        "invalidates": sorted(invalidates),
        "plan_version": plan_version + 1,
    }
    return {
        "draft": draft,
        "condition_revisions": [revision],
        "plan_version": plan_version + 1,
        "status": "plan_revised",
    }


def _resolve_empty_candidates(candidate_filter: CandidateFilter):
    """节点工厂：候选人为空时用 leave-one-out 诊断定位瓶颈条件，等待用户放宽或取消。"""

    def resolve_empty_candidates(
        state: TalentRequestState,
        runtime: Runtime[DecisionContext],
    ) -> dict[str, Any]:
        plan = QueryPlan.model_validate(state["query_plan"])
        plan_version = state.get("plan_version", 0)
        interaction_id = f"ec-v{plan_version}"
        diagnostics = _diagnose_filters(plan.filters, candidate_filter, runtime.context)
        current = [
            FilterCondition.model_validate(item) for item in state["draft"].get("filters", [])
        ]
        payload = {
            "type": "empty_candidate_set",
            "interaction_id": interaction_id,
            "question": "当前条件未命中任何候选人，可放宽部分条件后重试，或取消本次任务",
            "filters": diagnostics,
            "actions": ["relax", "cancel"],
        }

        def parse(decision: Any) -> dict[str, Any]:
            decision = _require_interaction(decision, interaction_id)
            action = decision.get("action")
            if action == "cancel":
                return {"action": "cancel"}
            if action != "relax":
                raise ValueError("action 必须是 relax 或 cancel")
            relaxations = decision.get("relaxations") or []
            if not relaxations:
                raise ValueError("relax 动作必须携带 relaxations")
            updated, changes = _apply_relaxations(current, relaxations)
            return {"action": "relax", "updated": updated, "changes": changes}

        parsed = _await_decision(payload, parse)
        if parsed["action"] == "cancel":
            return {"status": "cancelled"}

        draft = dict(state["draft"])
        draft["filters"] = [item.model_dump(mode="json") for item in parsed["updated"]]
        revision = {
            "interaction_id": interaction_id,
            "action": "relax",
            "changes": parsed["changes"],
            "resolved_clarifications": [],
            "invalidates": ["candidate_set"],
            "plan_version": plan_version + 1,
        }
        return {
            "draft": draft,
            "condition_revisions": [revision],
            "plan_version": plan_version + 1,
            "status": "plan_revised",
        }

    return resolve_empty_candidates


def build_talent_request_graph(
    *,
    request_interpreter: RequestInterpreter,  # 任务理解器
    job_lookup: JobLookup,  # 岗位匹配器
    candidate_filter: CandidateFilter | None = None,  # 候选人筛选器
    case_recorder: CaseRecorder | None = None,  # 用例记录器
    checkpointer: Any | None = None,
):
    """构建请求理解图。

    主干：receive_request -> interpret_input，随后按输入模式分叉为岗位匹配链或计划构建链。
    当提供 candidate_filter 时，额外挂载候选人筛选与多类可恢复人机协同节点（澄清/放宽/修订）。
    """
    builder = StateGraph(
        TalentRequestState,
        context_schema=DecisionContext,
        input_schema=TalentRequestInput,
        output_schema=TalentRequestOutput,
    )
    builder.add_node("receive_request", _receive_request)
    builder.add_node("interpret_input", _interpret_input(request_interpreter, case_recorder))
    builder.add_node("lookup_jobs", _lookup_jobs(job_lookup, case_recorder))
    builder.add_node("select_exact_job", _select_exact_job)
    builder.add_node("confirm_job", _confirm_job)
    builder.add_node(
        "compile_selected_job",
        _compile_selected_job(request_interpreter, case_recorder),
    )
    builder.add_node("build_plan", _build_plan)
    builder.add_node("no_job_match", _no_job_match)
    builder.add_edge(START, "receive_request")
    builder.add_edge("receive_request", "interpret_input")
    builder.add_conditional_edges("interpret_input", _route_input)
    builder.add_conditional_edges("lookup_jobs", _route_job_matches)
    builder.add_edge("select_exact_job", "compile_selected_job")
    builder.add_conditional_edges(
        "confirm_job",
        _route_job_confirmation,
        {"compile_selected_job": "compile_selected_job", END: END},
    )
    builder.add_edge("compile_selected_job", "build_plan")
    builder.add_edge("no_job_match", END)

    if candidate_filter is None:
        # 第 14 课行为：编译出 QueryPlan 即结束，候选人检索与澄清节点不挂载
        builder.add_edge("build_plan", END)
    else:
        builder.add_node("filter_candidates", _filter_candidates_node(candidate_filter))
        builder.add_node("resolve_requirements", _resolve_requirements())
        builder.add_conditional_edges(
            "build_plan",
            _route_after_plan,
            {"resolve_requirements": "resolve_requirements", "filter_candidates": "filter_candidates"},
        )
        builder.add_conditional_edges(
            "filter_candidates",
            _route_candidates,
            {"resolve_requirements": "resolve_requirements", END: END},
        )
        builder.add_conditional_edges(
            "resolve_requirements",
            _route_resolution,
            {"interpret_input": "interpret_input", END: END},
        )
    return builder.compile(checkpointer=checkpointer) if checkpointer is not None else builder.compile()


# 任务理解器的系统提示词：指导 LLM 如何把自然语言需求编译为结构化 TalentRequestDraft
MODEL_SYSTEM_PROMPT = """你是人才评估任务编译器。按照给定 Schema 输出结构化结果。
结合最近消息窗口和 previous_request，输出本轮修订后的完整 TalentRequestDraft。
首轮 revision_action 为 initial，conversation_title 为空时生成不超过 20 个中文字的标题。
已有 conversation_title 时不生成标题，后续轮根据最新消息在 add、update、remove 中选择，保留未被修改的既有条件。
用 revision_summary 简要说明变化。
如果输入只有岗位名称或岗位简称，input_mode 使用 job_name，只填写 job_query，不根据岗位名称猜测条件。
如果输入包含地区、年限、职级、技能、项目经验或偏好，input_mode 使用 detailed_requirement。
结构化硬条件只能写入 filters，字段定义与允许的操作符如下：
{filter_dsl}
地名只能映射为 region，current_position 只表示当前岗位名称，禁止将地名写入 current_position。
技能、经历和成果写入 semantic_requirements。优先项写入 evaluation_preferences。
只有歧义或矛盾导致无法形成可执行查询计划时才写入 clarifications。资深、经验丰富等可以用于语义检索或排序的描述写入 semantic_requirements 或 evaluation_preferences，不要求补充年龄等结构化阈值。租户和权限不得从用户文本提取。
当输入中包含 confirmed_job 时，按照已确认岗位 JD 编译条件，并保留用户补充要求。"""


def model_request_interpreter(
    messages: list[AnyMessage],
    previous_request: dict[str, Any],
    selected_job: JobMatch | None = None,
    conversation_title: str = "",
) -> TalentRequestUnderstanding:
    """基于 LLM 的任务理解实现：把消息历史、上一轮请求与（可选）确认岗位编译为结构化理解结果。"""
    model = get_chat_model(temperature=0)
    if model is None:
        raise RuntimeError("任务理解模型未配置，请设置 DASHSCOPE_API_KEY")
    messages = recent_request_messages(messages)
    history = "\n".join(
        f"{getattr(message, 'type', 'user')}: {message.content}" for message in messages
    )
    user_content = (
        f"message_history=\n{history}\n"
        f"previous_request={previous_request}\n"
        f"conversation_title={conversation_title}"
    )
    if selected_job is not None:
        user_content += (
            "\n"
            f"confirmed_job={selected_job['name']}\n"
            f"job_description={selected_job['content']}"
        )
    schema = TalentRequestRevision if conversation_title else TalentRequestUnderstanding
    result = invoke_structured(
        model,
        schema,
        [
            ("system", MODEL_SYSTEM_PROMPT.format(filter_dsl=filter_dsl_catalog())),
            ("user", user_content),
        ],
    )
    if conversation_title:
        revision = TalentRequestRevision.model_validate(result)
        return TalentRequestUnderstanding(**revision.model_dump(), conversation_title=conversation_title)
    return TalentRequestUnderstanding.model_validate(result)


def database_job_lookup(query: str, context: DecisionContext) -> list[JobMatch]:
    """基于数据库的岗位匹配实现：调用 TalentToolService 查询岗位 JD 列表。"""
    service = TalentToolService(session_factory=SessionLocal)
    tool_context = TalentToolContext(
        tenant_id=context.tenant_id,
        permission_scopes=context.permission_scopes,
        actor_id="langsmith-studio",
        run_id="lesson-14",
    )
    return service.lookup_job_descriptions(query, context=tool_context)


def database_candidate_filter(filters: list[FilterCondition], context: DecisionContext) -> list[str]:
    """基于数据库的候选人筛选实现：调用 TalentToolService 过滤候选人 ID 列表。"""
    service = TalentToolService(session_factory=SessionLocal)
    tool_context = TalentToolContext(
        tenant_id=context.tenant_id,
        permission_scopes=context.permission_scopes,
        actor_id="langsmith-studio",
        run_id="lesson-17",
    )
    return service.filter_candidates(filters, context=tool_context)


# Agent Server injects its managed checkpointer when this graph is loaded from langgraph.json.
graph = build_talent_request_graph(
    request_interpreter=model_request_interpreter,
    job_lookup=database_job_lookup,
    candidate_filter=database_candidate_filter,
    case_recorder=SqlEvaluationCaseStore(SessionLocal).record,
)