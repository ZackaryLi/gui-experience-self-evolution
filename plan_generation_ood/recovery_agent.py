import asyncio
import json
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

from .config import OODRecoveryAgentConfig
from .hooks import (
    PostToolUseEvent,
    PreToolUseEvent,
    ToolHook,
    ToolHookError,
    ToolHookRunner,
)
from .llm import extract_json
from .recovery_prompts import build_initial_messages
from .recovery_schemas import (
    AgentAction,
    AgentActionType,
    AgentRunResult,
    AgentRunStatus,
    AgentStep,
    PermissionDecision,
    PermissionOutcome,
    RecoveryContext,
    RecoveryDecision,
    ToolCall,
    ToolObservation,
)
from .safety import DefaultPermissionPolicy, PermissionPolicy
from .schemas import EvidenceRef
from .tools import ToolRegistry, build_default_tool_registry

if TYPE_CHECKING:
    from openai import AsyncOpenAI


class OODRecoveryAgent:
    def __init__(
        self,
        context: RecoveryContext,
        llm_agent: "AsyncOpenAI",
        model: str,
        config: OODRecoveryAgentConfig | None = None,
        close_llm: bool = True,
        tool_registry: Optional[ToolRegistry] = None,
        permission_policy: Optional[PermissionPolicy] = None,
        tool_hooks: Optional[Iterable[ToolHook]] = None,
    ):
        if not context.intent_id.strip():
            raise ValueError("RecoveryContext.intent_id 不能为空")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model 必须是非空字符串")
        self.context = context
        self.llm = llm_agent
        self.model = model.strip()
        self.config = config or OODRecoveryAgentConfig()
        self.close_llm = close_llm
        self.tool_registry = tool_registry or build_default_tool_registry(context)
        self.permission_policy = permission_policy or DefaultPermissionPolicy()
        self.tool_hook_runner = ToolHookRunner(
            tool_hooks,
            timeout_seconds=self.config.hook_timeout_seconds,
        )
        self._allowed_evidence = self._index_evidence(context.evidence_refs)
        self._run_id = ""
        self._event_sequence = 0

    def run(self) -> AgentRunResult:
        return asyncio.run(self.arun())

    async def arun(self) -> AgentRunResult:
        self._run_id = uuid.uuid4().hex
        self._event_sequence = 0
        try:
            result = await self._run()
            result.run_id = self._run_id
            return result
        finally:
            if self.close_llm:
                await self.llm.close()

    async def _run(self) -> AgentRunResult:
        messages = build_initial_messages(self.context)
        steps: List[AgentStep] = []
        llm_calls = 0
        tool_rounds = 0
        invalid_responses = 0

        while True:
            raw_response = ""
            call_error = ""
            response_message: Optional[Dict[str, Any]] = None
            try:
                completion = await self.llm.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tool_registry.to_openai_tools(),
                    tool_choice="auto",
                    temperature=0.1,
                )
                message = completion.choices[0].message
                if hasattr(message, "model_dump"):
                    response = message.model_dump(exclude_none=True)
                elif isinstance(message, dict):
                    response = message
                else:
                    raise ValueError("LLM 响应不是 OpenAI assistant message 对象")
                if response.get("role", "assistant") != "assistant":
                    raise ValueError("LLM 响应的 role 必须是 assistant")
                response_message = response
                raw_response = json.dumps(
                    response,
                    ensure_ascii=False,
                    default=str,
                )
            except Exception as error:
                call_error = f"LLM_CALL_FAILED: {error}"

            llm_calls += 1
            step_id = f"step-{len(steps) + 1}"
            if call_error:
                invalid_responses += 1
                steps.append(AgentStep(
                    step_id=step_id,
                    raw_response=raw_response,
                    validation_error=call_error,
                ))
                messages.append({
                    "role": "user",
                    "content": f"上一轮调用失败，请修正：{call_error}",
                })
                if invalid_responses >= self.config.max_invalid_responses:
                    return AgentRunResult(
                        status=AgentRunStatus.INVALID_MODEL_OUTPUT,
                        reason=call_error,
                        decision=None,
                        steps=steps,
                        llm_calls=llm_calls,
                        tool_rounds=tool_rounds,
                        invalid_responses=invalid_responses,
                    )
                continue

            tool_calls = response_message.get("tool_calls") or []
            if tool_calls:
                call_id = None
                assistant_appended = False
                try:
                    call_id, tool_call = self._parse_native_tool_call(tool_calls)
                    messages.append(self._assistant_message(response_message))
                    assistant_appended = True
                    action = self._action_from_tool_call(tool_call)
                except Exception as error:
                    validation_error = str(error)
                    invalid_responses += 1
                    steps.append(AgentStep(
                        step_id=step_id,
                        raw_response=raw_response,
                        validation_error=validation_error,
                    ))
                    if assistant_appended and call_id:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": json.dumps({
                                "success": False,
                                "error": validation_error,
                            }, ensure_ascii=False),
                        })
                    else:
                        messages.append({
                            "role": "user",
                            "content": f"上一轮工具调用格式无效，请修正：{validation_error}",
                        })
                    if invalid_responses >= self.config.max_invalid_responses:
                        return AgentRunResult(
                            status=AgentRunStatus.INVALID_MODEL_OUTPUT,
                            reason=validation_error,
                            decision=None,
                            steps=steps,
                            llm_calls=llm_calls,
                            tool_rounds=tool_rounds,
                            invalid_responses=invalid_responses,
                        )
                    continue
            else:
                messages.append(self._assistant_message(response_message))
                try:
                    action = self._parse_finish(response_message.get("content"))
                except Exception as error:
                    validation_error = str(error)
                    invalid_responses += 1
                    steps.append(AgentStep(
                        step_id=step_id,
                        raw_response=raw_response,
                        validation_error=validation_error,
                    ))
                    messages.append({
                        "role": "user",
                        "content": f"上一轮最终决策格式无效，请修正：{validation_error}",
                    })
                    if invalid_responses >= self.config.max_invalid_responses:
                        return AgentRunResult(
                            status=AgentRunStatus.INVALID_MODEL_OUTPUT,
                            reason=validation_error,
                            decision=None,
                            steps=steps,
                            llm_calls=llm_calls,
                            tool_rounds=tool_rounds,
                            invalid_responses=invalid_responses,
                        )
                    continue

            if action.action is AgentActionType.FINISH:
                steps.append(AgentStep(
                    step_id=step_id,
                    raw_response=raw_response,
                    action=action,
                ))
                return AgentRunResult(
                    status=AgentRunStatus.FINISHED,
                    reason=action.reason,
                    decision=action.decision,
                    steps=steps,
                    llm_calls=llm_calls,
                    tool_rounds=tool_rounds,
                    invalid_responses=invalid_responses,
                )

            if tool_rounds >= self.config.max_tool_rounds:
                limit_error = (
                    f"工具调用轮数超过上限 {self.config.max_tool_rounds}，"
                    "本动作未执行"
                )
                steps.append(AgentStep(
                    step_id=step_id,
                    raw_response=raw_response,
                    action=action,
                    validation_error=limit_error,
                ))
                return AgentRunResult(
                    status=AgentRunStatus.MAX_TOOL_ROUNDS_EXCEEDED,
                    reason=limit_error,
                    decision=None,
                    steps=steps,
                    llm_calls=llm_calls,
                    tool_rounds=tool_rounds,
                    invalid_responses=invalid_responses,
                )

            permission = self._evaluate_permission(action.tool_call)
            if permission.outcome is not PermissionOutcome.ALLOW:
                observation = ToolObservation(
                    tool=action.tool_call.tool,
                    success=False,
                    data={"permission": permission.to_dict()},
                    error=f"{permission.reason_code}: {permission.reason}",
                )
                steps.append(AgentStep(
                    step_id=step_id,
                    raw_response=raw_response,
                    action=action,
                    observation=observation,
                    permission=permission,
                ))
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(
                        observation.to_dict(),
                        ensure_ascii=False,
                        default=str,
                    ),
                })
                run_status = (
                    AgentRunStatus.PERMISSION_DENIED
                    if permission.outcome is PermissionOutcome.DENY
                    else AgentRunStatus.MANUAL_REVIEW_REQUIRED
                )
                return AgentRunResult(
                    status=run_status,
                    reason=permission.reason,
                    decision=None,
                    steps=steps,
                    llm_calls=llm_calls,
                    tool_rounds=tool_rounds,
                    invalid_responses=invalid_responses,
                )

            definition = self.tool_registry.get(action.tool_call.tool)
            self._event_sequence += 1
            pre_event = PreToolUseEvent(
                run_id=self._run_id,
                step_id=step_id,
                sequence=self._event_sequence,
                tool_call_id=call_id,
                tool=action.tool_call.tool,
                intent_id=self.context.intent_id,
                operation=definition.operation,
                permission=permission,
            )
            try:
                await self.tool_hook_runner.run_before(pre_event)
            except ToolHookError as error:
                observation = ToolObservation(
                    tool=action.tool_call.tool,
                    success=False,
                    data={"hook": {
                        "phase": error.phase,
                        "hook": error.hook,
                        "error_type": error.error_type,
                        "handler_called": False,
                    }},
                    error=str(error),
                )
                steps.append(AgentStep(
                    step_id=step_id,
                    raw_response=raw_response,
                    action=action,
                    observation=observation,
                    permission=permission,
                    hook_error=str(error),
                ))
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(
                        observation.to_dict(),
                        ensure_ascii=False,
                        default=str,
                    ),
                })
                return AgentRunResult(
                    status=AgentRunStatus.MANUAL_REVIEW_REQUIRED,
                    reason=str(error),
                    decision=None,
                    steps=steps,
                    llm_calls=llm_calls,
                    tool_rounds=tool_rounds,
                    invalid_responses=invalid_responses,
                )

            started_at = time.perf_counter()
            observation = await self.tool_registry.dispatch(
                action.tool_call.tool,
                action.tool_call.input,
            )
            self._absorb_observation_evidence(observation.data)
            duration_ms = (time.perf_counter() - started_at) * 1000
            tool_rounds += 1
            self._event_sequence += 1
            post_event = PostToolUseEvent(
                run_id=self._run_id,
                step_id=step_id,
                sequence=self._event_sequence,
                tool_call_id=call_id,
                tool=action.tool_call.tool,
                intent_id=self.context.intent_id,
                operation=definition.operation,
                permission=permission,
                observation=observation,
                duration_ms=duration_ms,
                handler_called=True,
            )
            try:
                await self.tool_hook_runner.run_after(post_event)
            except ToolHookError as error:
                steps.append(AgentStep(
                    step_id=step_id,
                    raw_response=raw_response,
                    action=action,
                    observation=observation,
                    permission=permission,
                    hook_error=str(error),
                ))
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(
                        observation.to_dict(),
                        ensure_ascii=False,
                        default=str,
                    ),
                })
                return AgentRunResult(
                    status=AgentRunStatus.MANUAL_REVIEW_REQUIRED,
                    reason=str(error),
                    decision=None,
                    steps=steps,
                    llm_calls=llm_calls,
                    tool_rounds=tool_rounds,
                    invalid_responses=invalid_responses,
                )

            steps.append(AgentStep(
                step_id=step_id,
                raw_response=raw_response,
                action=action,
                observation=observation,
                permission=permission,
            ))
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(
                    observation.to_dict(),
                    ensure_ascii=False,
                    default=str,
                ),
            })

    def _evaluate_permission(self, tool_call: ToolCall) -> PermissionDecision:
        definition = self.tool_registry.get(tool_call.tool)
        try:
            return self.permission_policy.evaluate(
                self.context,
                definition,
                tool_call,
            )
        except Exception as error:
            return PermissionDecision(
                outcome=PermissionOutcome.MANUAL_REVIEW,
                reason_code="PERMISSION_POLICY_ERROR",
                reason="权限策略执行失败，已按失败关闭原则停止工具调用。",
                tool=tool_call.tool,
                intent_id=self.context.intent_id,
                details={"error_type": type(error).__name__},
            )

    @staticmethod
    def _assistant_message(response_message: Dict[str, Any]) -> Dict[str, Any]:
        message = dict(response_message)
        message["role"] = "assistant"
        return message

    @staticmethod
    def _parse_native_tool_call(tool_calls: Any) -> Tuple[str, ToolCall]:
        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            raise ValueError("每轮必须且只能包含一个原生 tool_call")
        call = tool_calls[0]
        if not isinstance(call, dict) or call.get("type") != "function":
            raise ValueError("tool_call 必须是 function 类型")
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("tool_call.id 必须是非空字符串")
        function = call.get("function")
        if not isinstance(function, dict):
            raise ValueError("tool_call.function 必须是对象")
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("tool_call.function.name 必须是非空字符串")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise ValueError("tool_call.function.arguments 必须是 JSON 字符串")
        try:
            tool_input = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise ValueError(f"工具参数不是合法 JSON: {error}") from error
        if not isinstance(tool_input, dict):
            raise ValueError("工具参数 JSON 必须是对象")
        return call_id, ToolCall(tool=name.strip(), input=tool_input)

    def _action_from_tool_call(self, tool_call: ToolCall) -> AgentAction:
        self.tool_registry.validate(tool_call.tool, tool_call.input)
        return AgentAction(
            action=AgentActionType.CALL_TOOL,
            reason=f"调用原生工具 {tool_call.tool}",
            evidence_refs=[],
            tool_call=tool_call,
        )

    def _parse_finish(self, content: Any) -> AgentAction:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("最终决策必须是 assistant content 中的非空 JSON")
        payload = extract_json(content, default={})
        if not isinstance(payload, dict):
            raise ValueError("最终决策必须是 JSON 对象")
        allowed_fields = {
            "decision",
            "reason",
            "evidence_refs",
            "experiment_id",
            "conclusion",
            "hypothesis",
        }
        unknown_fields = set(payload) - allowed_fields
        if unknown_fields:
            raise ValueError(f"最终决策包含未知字段: {sorted(unknown_fields)}")
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason 必须是非空字符串")
        decision = RecoveryDecision(str(payload.get("decision", "")))
        optional = {}
        for name in ("experiment_id", "conclusion", "hypothesis"):
            value = payload.get(name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} 必须是非空字符串或 null")
            optional[name] = value.strip() if isinstance(value, str) else None
        return AgentAction(
            action=AgentActionType.FINISH,
            reason=reason.strip(),
            evidence_refs=self._parse_evidence_refs(payload.get("evidence_refs")),
            decision=decision,
            **optional,
        )

    def _parse_evidence_refs(self, raw_refs: Any) -> List[EvidenceRef]:
        if not isinstance(raw_refs, list):
            raise ValueError("evidence_refs 必须是数组")
        refs = []
        for raw_ref in raw_refs:
            if not isinstance(raw_ref, dict) or set(raw_ref) - {"sessionid", "step_ids"}:
                raise ValueError("证据引用只能包含 sessionid 和 step_ids")
            sessionid = raw_ref.get("sessionid")
            step_ids = raw_ref.get("step_ids", [])
            if not isinstance(sessionid, str) or sessionid not in self._allowed_evidence:
                raise ValueError(f"引用了不存在的证据 sessionid: {sessionid}")
            if not isinstance(step_ids, list) or any(
                not isinstance(step_id, str) for step_id in step_ids
            ):
                raise ValueError("step_ids 必须是字符串数组")
            unknown_steps = set(step_ids) - self._allowed_evidence[sessionid]
            if unknown_steps:
                raise ValueError(f"引用了不存在的证据步骤: {sorted(unknown_steps)}")
            refs.append(EvidenceRef(sessionid=sessionid, step_ids=list(step_ids)))
        return refs

    def _absorb_observation_evidence(self, value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                self._absorb_observation_evidence(item)
            return
        if not isinstance(value, dict):
            return
        sessionid = value.get("sessionid")
        if isinstance(sessionid, str) and sessionid.strip():
            for field_name in ("step_ids", "key_evidence_steps"):
                steps = value.get(field_name)
                if isinstance(steps, list) and all(
                    isinstance(step, str) for step in steps
                ):
                    self._allowed_evidence.setdefault(sessionid, set()).update(steps)
        trajectory = value.get("trajectory")
        if isinstance(trajectory, dict):
            trajectory_session = trajectory.get("sessionid")
            if isinstance(trajectory_session, str) and trajectory_session.strip():
                steps = [
                    key for key in trajectory
                    if isinstance(key, str)
                    and key.startswith("step")
                    and key[4:].isdigit()
                ]
                self._allowed_evidence.setdefault(trajectory_session, set()).update(steps)
        for item in value.values():
            self._absorb_observation_evidence(item)

    @staticmethod
    def _index_evidence(evidence_refs: List[EvidenceRef]) -> Dict[str, set]:
        allowed: Dict[str, set] = {}
        for ref in evidence_refs:
            allowed.setdefault(ref.sessionid, set()).update(ref.step_ids)
        return allowed
