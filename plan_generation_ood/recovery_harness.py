from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .config import OODExperimentConfig, OODRecoveryAgentConfig
from .experiment_plan import ExperimentPlanStatus
from .experiments import ExperimentExecutor
from .hooks import ToolHook
from .memory import RecoveryMemory, experience_version_id
from .patch_generator import PatchGenerationResult, PatchGenerator
from .recovery_agent import OODRecoveryAgent
from .recovery_pipeline import RecoveryWorkflow
from .recovery_schemas import (
    AgentRunResult,
    AgentRunStatus,
    RecoveryContext,
    RecoveryDecision,
)
from .runner import DeviceRunner
from .safety import ExecutionPermissionPolicy
from .schemas import (
    AttemptStatus,
    BaselinePlan,
    DiagnosisReport,
    DictSerializable,
    RecoveryState,
    RecoveryStateSnapshot,
    RecoveryValueAssessment,
)
from .skills import SkillLoader
from .tools import ToolDefinition, ToolOperation, ToolRegistry, build_default_tool_registry
from .verdict import AttemptVerdictAnalyzer


class RecoveryHarnessError(ValueError):
    pass


@dataclass
class RecoveryIterationResult(DictSerializable):
    agent_run: AgentRunResult
    snapshot: RecoveryStateSnapshot
    patch: Optional[PatchGenerationResult] = None
    decision_applied: bool = False
    application_reason: str = ""


def build_experiment_tool_definitions(
    executor: ExperimentExecutor,
) -> List[ToolDefinition]:
    intent_property = {"type": "string", "const": executor.memory.intent_id}
    submit_schema = {
        "type": "object",
        "properties": {
            "intent_id": intent_property,
            "plan_item_id": {"type": "string", "minLength": 1},
            "instruction_ids": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
            "attempts_per_instruction": {
                "type": "integer",
                "minimum": 1,
                "maximum": executor.config.max_attempts_per_experiment,
            },
        },
        "required": ["intent_id", "plan_item_id"],
        "additionalProperties": False,
    }
    experiment_schema = {
        "type": "object",
        "properties": {
            "intent_id": intent_property,
            "experiment_id": {"type": "string", "minLength": 1},
        },
        "required": ["intent_id", "experiment_id"],
        "additionalProperties": False,
    }
    cancel_schema = {
        "type": "object",
        "properties": {
            "intent_id": intent_property,
            "idempotency_key": {"type": "string", "minLength": 1},
        },
        "required": ["intent_id", "idempotency_key"],
        "additionalProperties": False,
    }

    async def submit(tool_input: Dict[str, Any]) -> Dict[str, Any]:
        result = await executor.aexecute(
            plan_item_id=tool_input["plan_item_id"],
            instruction_ids=tool_input.get("instruction_ids"),
            attempts_per_instruction=tool_input.get("attempts_per_instruction", 1),
        )
        return result.to_dict()

    def get_result(tool_input: Dict[str, Any]) -> Dict[str, Any]:
        return executor.get_experiment(tool_input["experiment_id"])

    async def cancel(tool_input: Dict[str, Any]) -> Dict[str, Any]:
        result = await executor.cancel(tool_input["idempotency_key"])
        return result.to_dict()

    return [
        ToolDefinition(
            name="submit_experiment",
            description=(
                "对当前 PENDING/ACTIVE 假设和当前经验版本执行受控实验；"
                "执行器检查预算、指令范围、经验版本和幂等，不允许提供任意设备或经验正文。"
            ),
            input_schema=submit_schema,
            handler=submit,
            timeout_seconds=executor.config.tool_timeout_seconds,
            operation=ToolOperation.EXPERIMENT,
            requires_intent_scope=True,
        ),
        ToolDefinition(
            name="get_experiment_result",
            description="精确读取一个实验的原始 attempt、独立 Verdict 和派生反馈，不触发新执行。",
            input_schema=experiment_schema,
            handler=get_result,
            operation=ToolOperation.READ_ONLY,
            requires_intent_scope=True,
        ),
        ToolDefinition(
            name="cancel_experiment",
            description="按 attempt 幂等键请求 Runner 尽力取消；返回取消回执，不伪造最终轨迹。",
            input_schema=cancel_schema,
            handler=cancel,
            timeout_seconds=30.0,
            operation=ToolOperation.EXPERIMENT,
            requires_intent_scope=True,
        ),
    ]


class RecoveryHarness:
    def __init__(
        self,
        recovery_dir: str,
        baseline_plan: BaselinePlan,
        diagnosis: DiagnosisReport,
        recovery_value: RecoveryValueAssessment,
        llm_agent,
        model: str,
        runner: Optional[DeviceRunner] = None,
        execution_authorized: bool = False,
        experiment_config: Optional[OODExperimentConfig] = None,
        agent_config: Optional[OODRecoveryAgentConfig] = None,
        patch_generator: Optional[PatchGenerator] = None,
        verdict_analyzer: Optional[AttemptVerdictAnalyzer] = None,
        environment: Optional[Dict[str, Any]] = None,
        safety_context: Optional[Dict[str, Any]] = None,
        tool_hooks: Optional[Iterable[ToolHook]] = None,
    ):
        self.memory = RecoveryMemory(recovery_dir, baseline_plan.intent_id)
        self.baseline_plan = baseline_plan
        self.diagnosis = diagnosis
        self.recovery_value = recovery_value
        self.llm = llm_agent
        self.model = model
        self.agent_config = agent_config or OODRecoveryAgentConfig()
        self.tool_hooks = tuple(tool_hooks or ())
        self.patch_generator = patch_generator or PatchGenerator(
            llm_agent,
            model,
            skill_loader=SkillLoader(),
        )
        self.verdict_analyzer = verdict_analyzer or AttemptVerdictAnalyzer(
            llm_agent,
            model,
        )
        self.executor = ExperimentExecutor(
            self.memory,
            baseline_plan,
            runner=runner,
            verdict_analyzer=self.verdict_analyzer,
            config=experiment_config,
            environment=environment,
            safety_context=safety_context,
        )
        self.workflow = RecoveryWorkflow(recovery_dir)
        self.permission_policy = ExecutionPermissionPolicy(execution_authorized)

    async def arun_iteration(self) -> RecoveryIterationResult:
        self.workflow.resume_pending(self.memory)
        context = self._context()
        registry = build_default_tool_registry(
            context,
            context_provider=self._context,
        )
        if self.executor.runner is not None:
            for definition in build_experiment_tool_definitions(self.executor):
                registry.register(definition)
        agent = OODRecoveryAgent(
            context=context,
            llm_agent=self.llm,
            model=self.model,
            config=self.agent_config,
            close_llm=False,
            tool_registry=registry,
            permission_policy=self.permission_policy,
            tool_hooks=self.tool_hooks,
        )
        agent_run = await agent.arun()
        result = RecoveryIterationResult(
            agent_run=agent_run,
            snapshot=self.memory.recover(),
        )
        if agent_run.status is not AgentRunStatus.FINISHED:
            result.application_reason = "Agent 未返回可应用的结构化最终决策"
            return result
        return await self._apply_decision(result)

    def _context(self) -> RecoveryContext:
        return self.memory.build_context(self.diagnosis, self.recovery_value)

    async def _apply_decision(self, result: RecoveryIterationResult) -> RecoveryIterationResult:
        action = next((
            step.action for step in reversed(result.agent_run.steps)
            if step.action is not None and step.action.decision is not None
        ), None)
        if action is None or action.decision is None:
            raise RecoveryHarnessError("Agent 运行结果缺少最终动作")
        decision = action.decision
        if decision in {RecoveryDecision.REVISE_EXPERIENCE, RecoveryDecision.PROPOSE_EXPERIENCE}:
            return await self._apply_patch(result, action)
        snapshot = self.memory.recover()
        if decision is RecoveryDecision.ACCEPT_EXPERIENCE:
            plan = self.memory.load_plan()
            active = plan.active_item
            if active is None or active.experiment_id is None:
                raise RecoveryHarnessError("没有可请求验证的活动实验")
            feedback, evidence = self.executor.feedback_for_experiment(active.experiment_id)
            if not any(
                item.status is AttemptStatus.COMPLETED
                and item.request.experience is not None
                and experience_version_id(item.request.experience) == snapshot.current_experience_id
                for item in feedback
            ):
                raise RecoveryHarnessError("当前经验没有有效 completed 反馈，不能请求验证")
            plan.conclude(
                active.item_id,
                ExperimentPlanStatus.SUPPORTED,
                "当前假设获得有效完成反馈；仅请求后续独立验证，不代表目标达成",
                evidence,
            )
            self.memory.save_plan(plan)
            result.snapshot = self.workflow.commit_exploration_state(
                self.memory,
                RecoveryState.VALIDATING,
                stage="VALIDATION_REQUESTED",
                reason=action.reason,
                snapshot=snapshot,
            )
            result.decision_applied = True
            result.application_reason = "已进入 VALIDATING，仍需 s17 独立目标闸门"
            return result
        if decision in {
            RecoveryDecision.STOP_BLOCKED,
            RecoveryDecision.STOP_RETRYABLE,
            RecoveryDecision.ESCALATE,
            RecoveryDecision.STOP_UNSOLVABLE,
            RecoveryDecision.REQUEST_MORE_BASELINE,
        }:
            if decision is RecoveryDecision.STOP_UNSOLVABLE:
                target = RecoveryState.MANUAL_REVIEW
                reason = "s15 不接受 Agent 直接宣布不可解，转人工复核：" + action.reason
            elif decision is RecoveryDecision.ESCALATE:
                target = RecoveryState.MANUAL_REVIEW
                reason = action.reason
            elif decision is RecoveryDecision.REQUEST_MORE_BASELINE:
                target = RecoveryState.STOP_RETRYABLE
                reason = "需要补充基线证据：" + action.reason
            else:
                target = RecoveryState(decision.value)
                reason = action.reason
            result.snapshot = self.workflow.commit_exploration_state(
                self.memory,
                target,
                stage="AGENT_EXIT_DECISION",
                reason=reason,
                snapshot=snapshot,
            )
            result.decision_applied = True
            result.application_reason = reason
            return result
        result.snapshot = snapshot
        result.application_reason = "该建议不直接修改状态；需要通过实验工具获取新证据"
        return result

    async def _apply_patch(self, result: RecoveryIterationResult, action) -> RecoveryIterationResult:
        snapshot = self.memory.recover()
        plan = self.memory.load_plan()
        active = plan.active_item
        if active is None or active.experiment_id is None:
            raise RecoveryHarnessError("修订前必须存在 ACTIVE 实验及其反馈")
        if active.experience_version_id != snapshot.current_experience_id:
            raise RecoveryHarnessError("活动实验与当前经验版本不一致")
        if action.experiment_id is not None and action.experiment_id != active.experiment_id:
            raise RecoveryHarnessError("Agent 指定的实验与活动实验不一致")
        mode = action.decision
        conclusion = action.conclusion or ExperimentPlanStatus.INCONCLUSIVE.value
        try:
            outcome = ExperimentPlanStatus(conclusion)
        except ValueError as error:
            raise RecoveryHarnessError("实验结论状态非法") from error
        if outcome not in {
            ExperimentPlanStatus.SUPPORTED,
            ExperimentPlanStatus.REFUTED,
            ExperimentPlanStatus.INCONCLUSIVE,
        }:
            raise RecoveryHarnessError("只能提交 SUPPORTED、REFUTED 或 INCONCLUSIVE 结论")
        if mode is RecoveryDecision.PROPOSE_EXPERIENCE:
            if outcome is not ExperimentPlanStatus.REFUTED:
                raise RecoveryHarnessError("新方向必须基于 REFUTED 实验结论")
            if action.hypothesis is None:
                raise RecoveryHarnessError("新方向必须显式提供 hypothesis")
        elif outcome is ExperimentPlanStatus.REFUTED:
            raise RecoveryHarnessError("原假设已否定时应使用 PROPOSE_EXPERIENCE 提供新方向")

        feedback, evidence = self.executor.feedback_for_experiment(active.experiment_id)
        current = self.memory.experiences.get(snapshot.current_experience_id).experience
        patch_result = await self.patch_generator.agenerate(
            plan=plan,
            diagnosis=self.diagnosis,
            current_experience=current,
            experiment_feedback=feedback,
            mode=mode,
            new_hypothesis=action.hypothesis,
        )
        plan.conclude(
            active.item_id,
            outcome,
            action.reason,
            evidence,
        )
        plan.add_item(
            hypothesis=patch_result.experience.hypothesis,
            research_question=(
                "携带新的候选经验版本后，能否获得支持当前假设的新业务证据？"
            ),
            expected_evidence=list(active.expected_evidence),
            falsification_conditions=list(active.falsification_conditions),
            failure_pattern_refs=list(active.failure_pattern_refs),
            evidence_refs=list(patch_result.experience.evidence_refs),
            reason=patch_result.change_summary,
            parent_item_id=active.item_id,
        )
        self.memory.experiences.register(
            patch_result.experience,
            source="PATCH_GENERATOR",
            source_failure_pattern_id=(
                active.failure_pattern_refs[0] if active.failure_pattern_refs else ""
            ),
            source_details={
                "mode": mode.value,
                "plan_item_id": active.item_id,
                "experiment_id": active.experiment_id,
                "change_summary": patch_result.change_summary,
                "skill_name": patch_result.skill_name,
                "skill_version": patch_result.skill_version,
                "skill_content_hash": patch_result.skill_content_hash,
            },
        )
        self.memory.save_plan(plan)
        result.snapshot = self.workflow.commit_exploration_state(
            self.memory,
            RecoveryState.EXPERIENCE_EXPLORING,
            stage="EXPERIENCE_REVISION_REGISTERED",
            reason=patch_result.change_summary,
            experience_version_id=experience_version_id(patch_result.experience),
            hypothesis=patch_result.experience.hypothesis,
            round_index=snapshot.round_index + 1,
            snapshot=snapshot,
        )
        result.patch = patch_result
        result.decision_applied = True
        result.application_reason = patch_result.change_summary
        return result
