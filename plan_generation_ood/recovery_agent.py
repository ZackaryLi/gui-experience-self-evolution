import asyncio
from typing import Any, Dict, List, Optional

from .config import OODRecoveryAgentConfig
from .llm import LLMAgent, extract_json
from .recovery_prompts import build_recovery_prompt
from .recovery_schemas import (
    AgentAction,
    AgentActionType,
    AgentRunResult,
    AgentRunStatus,
    AgentStep,
    RecoveryContext,
    RecoveryDecision,
    ToolCall,
)
from .schemas import EvidenceRef
from .tools import ToolRegistry, build_default_tool_registry


class OODRecoveryAgent:
    def __init__(
        self,
        context: RecoveryContext,
        llm_agent: LLMAgent,
        config: OODRecoveryAgentConfig | None = None,
        close_llm: bool = True,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        if not context.intent_id.strip():
            raise ValueError("RecoveryContext.intent_id 不能为空")
        self.context = context
        self.llm = llm_agent
        self.config = config or OODRecoveryAgentConfig()
        self.close_llm = close_llm
        self.tool_registry = tool_registry or build_default_tool_registry(context)
        self._allowed_evidence = self._index_evidence(context.evidence_refs)

    def run(self) -> AgentRunResult:
        return asyncio.run(self.arun())

    async def arun(self) -> AgentRunResult:
        try:
            return await self._run()
        finally:
            if self.close_llm:
                await self.llm.close()

    async def _run(self) -> AgentRunResult:
        transcript: List[Dict[str, Any]] = []
        steps: List[AgentStep] = []
        llm_calls = 0
        tool_rounds = 0
        invalid_responses = 0

        while True:
            prompt = build_recovery_prompt(
                self.context,
                transcript,
                self.tool_registry,
            )
            raw_response = ""
            call_error = ""
            try:
                response = await self.llm.ainvoke(prompt, temperature=0.1)
                if isinstance(response, tuple):
                    response = response[0]
                if not isinstance(response, str):
                    raise ValueError("LLM 响应不是字符串")
                raw_response = response
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
                transcript.extend([
                    {"type": "assistant", "content": raw_response},
                    {"type": "validation_error", "content": call_error},
                ])
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

            try:
                action = self._parse_action(extract_json(raw_response, default={}))
            except Exception as error:
                validation_error = str(error)
                invalid_responses += 1
                steps.append(AgentStep(
                    step_id=step_id,
                    raw_response=raw_response,
                    validation_error=validation_error,
                ))
                transcript.extend([
                    {"type": "assistant", "content": raw_response},
                    {"type": "validation_error", "content": validation_error},
                ])
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

            observation = await self.tool_registry.dispatch(
                action.tool_call.tool,
                action.tool_call.input,
            )
            tool_rounds += 1
            steps.append(AgentStep(
                step_id=step_id,
                raw_response=raw_response,
                action=action,
                observation=observation,
            ))
            transcript.extend([
                {"type": "assistant", "content": raw_response},
                {"type": "tool_result", "content": observation.to_dict()},
            ])

    def _parse_action(self, payload: Any) -> AgentAction:
        if not isinstance(payload, dict):
            raise ValueError("Agent 输出必须是 JSON 对象")
        allowed_fields = {"action", "tool", "input", "decision", "reason", "evidence_refs"}
        unknown_fields = set(payload) - allowed_fields
        if unknown_fields:
            raise ValueError(f"Agent 输出包含未知字段: {sorted(unknown_fields)}")

        action_type = AgentActionType(str(payload.get("action", "")))
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason 必须是非空字符串")
        evidence_refs = self._parse_evidence_refs(payload.get("evidence_refs"))

        if action_type is AgentActionType.CALL_TOOL:
            if "decision" in payload:
                raise ValueError("call_tool 不能包含 decision")
            tool_name = payload.get("tool")
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError("tool 必须是非空字符串")
            tool_name = tool_name.strip()
            tool_input = payload.get("input")
            if not isinstance(tool_input, dict):
                raise ValueError("input 必须是对象")
            self.tool_registry.validate(tool_name, tool_input)
            return AgentAction(
                action=action_type,
                reason=reason.strip(),
                evidence_refs=evidence_refs,
                tool_call=ToolCall(tool=tool_name, input=tool_input),
            )

        if "tool" in payload or "input" in payload:
            raise ValueError("finish 不能包含 tool 或 input")
        decision = RecoveryDecision(str(payload.get("decision", "")))
        return AgentAction(
            action=action_type,
            reason=reason.strip(),
            evidence_refs=evidence_refs,
            decision=decision,
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

    @staticmethod
    def _index_evidence(evidence_refs: List[EvidenceRef]) -> Dict[str, set]:
        allowed: Dict[str, set] = {}
        for ref in evidence_refs:
            allowed.setdefault(ref.sessionid, set()).update(ref.step_ids)
        return allowed
