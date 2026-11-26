import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .runner import (
    ExperienceMode,
    ExecutionStatus,
    InfrastructureStatus,
    RUNNER_SCHEMA_VERSION,
    DeviceRunner,
    RunRequest,
    RunResult,
)
from .schemas import (
    AttemptRequest,
    AttemptResult,
    AttemptStatus,
    ExperienceDraft,
    canonical_json,
)


class RunnerAdapterError(ValueError):
    pass


class RunnerResultNotReady(RunnerAdapterError):
    pass


class RunnerResultNotFound(RunnerAdapterError):
    pass


def experience_version_id(experience: ExperienceDraft) -> str:
    return f"{experience.experience_id}:version-{experience.version}"


def experience_to_plan_text(experience: Optional[ExperienceDraft]) -> str:
    if experience is None:
        return ""
    payload = {
        "experience_id": experience.experience_id,
        "intent_id": experience.intent_id,
        "version": experience.version,
        "hypothesis": experience.hypothesis,
        "applicable_scope": experience.applicable_scope,
        "trigger_conditions": list(experience.trigger_conditions),
        "strategy": experience.strategy,
        "expected_stage": experience.expected_stage,
        "forbidden_behaviors": list(experience.forbidden_behaviors),
        "exit_conditions": list(experience.exit_conditions),
    }
    return canonical_json(payload)


@dataclass(frozen=True)
class RunAttemptContext:
    app: str
    run_id: str
    experiment_id: str
    category: str = ""
    intent_name: str = ""
    experience_mode: Optional[ExperienceMode] = None
    accepted_patch_ids: List[str] = field(default_factory=list)
    initial_state: str = "clean_app_start"
    timeout_sec: int = 600
    device_constraints: Dict[str, Any] = field(default_factory=dict)
    safety_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrajectoryAdapter:
    @staticmethod
    def normalize(trajectory: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if trajectory is None:
            return {}
        if not isinstance(trajectory, dict):
            raise RunnerAdapterError("Runner trajectory 必须是字典或 null")
        normalized = json.loads(json.dumps(trajectory, ensure_ascii=False, default=str))
        for key, step in normalized.items():
            if not key.startswith("step") or not key[4:].isdigit():
                continue
            if not isinstance(step, dict):
                raise RunnerAdapterError(f"轨迹步骤必须是对象: {key}")
            observation = step.get("observation")
            if observation is None:
                observation = {}
                step["observation"] = observation
            if not isinstance(observation, dict):
                raise RunnerAdapterError(f"轨迹 observation 必须是对象: {key}")
            for source_name in ("uiSummary", "ui_summary"):
                value = step.get(source_name)
                if value not in (None, "", {}) and "text" not in observation:
                    observation["text"] = value
            for name in ("page_id", "page_name", "activity", "current_page"):
                if name in step and name not in observation:
                    observation[name] = step[name]
        return normalized


class RunnerAdapter:
    def to_run_request(
        self,
        request: AttemptRequest,
        context: RunAttemptContext,
    ) -> RunRequest:
        if not isinstance(request, AttemptRequest):
            raise RunnerAdapterError("request 必须是 AttemptRequest")
        if not context.app.strip() or not context.run_id.strip() or not context.experiment_id.strip():
            raise RunnerAdapterError("Runner 上下文必须提供 app、run_id 和 experiment_id")
        if context.timeout_sec <= 0:
            raise RunnerAdapterError("Runner timeout_sec 必须是正数")
        experience = request.experience
        mode = context.experience_mode or (
            ExperienceMode.CANDIDATE if experience is not None else ExperienceMode.BASELINE
        )
        if experience is None and mode is not ExperienceMode.BASELINE:
            raise RunnerAdapterError("无经验 request 只能使用 baseline 模式")
        if experience is not None and mode is ExperienceMode.BASELINE:
            raise RunnerAdapterError("携带经验的 request 不能使用 baseline 模式")
        plan_text = experience_to_plan_text(experience)
        metadata = dict(context.metadata)
        metadata.update({
            "plan_id": request.plan_id,
            "instruction_id": request.instruction_id,
            "experience_version_id": (
                experience_version_id(experience) if experience is not None else None
            ),
        })
        return RunRequest(
            schema_version=RUNNER_SCHEMA_VERSION,
            idempotency_key=request.idempotency_key,
            run_id=context.run_id,
            intent_id=request.intent_id,
            experiment_id=context.experiment_id,
            instruction=request.instruction,
            app=context.app,
            attempt_index=request.attempt_index,
            experience_mode=mode,
            plan_text=plan_text,
            initial_state=context.initial_state,
            timeout_sec=context.timeout_sec,
            accepted_patch_ids=list(context.accepted_patch_ids),
            candidate_patch_id=(
                experience_version_id(experience) if experience is not None else None
            ),
            device_constraints=dict(context.device_constraints),
            safety_context=dict(context.safety_context),
            trajectory_context={
                "category": {"category": context.category},
                "app": context.app,
                "intent_cluster_name": context.intent_name,
            },
            metadata=metadata,
        )

    def to_attempt_result(
        self,
        run_request: RunRequest,
        source_request: AttemptRequest,
        result: RunResult,
    ) -> AttemptResult:
        if not isinstance(result, RunResult):
            raise RunnerAdapterError("result 必须是 RunResult")
        if result.schema_version != RUNNER_SCHEMA_VERSION:
            raise RunnerAdapterError("不支持的 Runner 结果 Schema 版本")
        if (
            result.idempotency_key != run_request.idempotency_key
            or result.instruction != run_request.instruction
            or result.app != run_request.app
            or result.attempt_index != run_request.attempt_index
        ):
            raise RunnerAdapterError("Runner 结果与提交请求身份不一致")
        if result.injected_plan_text != run_request.plan_text:
            raise RunnerAdapterError("Runner 实际注入文本与请求不一致")
        if result.execution_status is ExecutionStatus.RUNNING:
            raise RunnerResultNotReady("Runner attempt 仍在执行")
        status = self._status(result)
        task_completed = None
        error = None
        if result.error is not None:
            error_type = str(result.error.get("type", "unknown"))
            error_message = str(result.error.get("message", ""))
            error = f"{error_type}: {error_message}".rstrip()
        environment = {
            "device_meta": dict(result.device_meta),
            "run_context": dict(result.run_context),
            "runner_version": result.runner_version,
            "duration_sec": result.duration_sec,
            "raw_ref": result.raw_ref,
            "artifacts": dict(result.artifacts),
            "injected_plan_text": result.injected_plan_text,
            "runner_execution_status": result.execution_status.value,
            "runner_infrastructure_status": result.infrastructure_status.value,
            "runner_error": dict(result.error) if result.error is not None else None,
        }
        return AttemptResult(
            request=source_request,
            status=status,
            task_completed=task_completed,
            trajectory=TrajectoryAdapter.normalize(result.trajectory),
            environment=environment,
            error=error,
            started_at=result.started_at or None,
            finished_at=result.finished_at or None,
        )

    async def arun_attempt(
        self,
        runner: DeviceRunner,
        request: AttemptRequest,
        context: RunAttemptContext,
    ) -> AttemptResult:
        run_request = self.to_run_request(request, context)
        result = await runner.run_attempt(run_request)
        return self.to_attempt_result(run_request, request, result)

    async def aget_result(
        self,
        runner: DeviceRunner,
        run_request: RunRequest,
        source_request: AttemptRequest,
    ) -> AttemptResult:
        result = await runner.get_result(run_request.idempotency_key)
        if result is None:
            raise RunnerResultNotFound("Runner 未找到该幂等键的结果")
        return self.to_attempt_result(run_request, source_request, result)

    @staticmethod
    def _status(result: RunResult) -> AttemptStatus:
        if result.execution_status is ExecutionStatus.CANCELLED:
            return AttemptStatus.CANCELLED
        if result.infrastructure_status is not InfrastructureStatus.OK:
            return {
                InfrastructureStatus.TIMEOUT: AttemptStatus.TIMEOUT_UNKNOWN,
                InfrastructureStatus.DEVICE_ERROR: AttemptStatus.DEVICE_ERROR,
                InfrastructureStatus.RUNNER_ERROR: AttemptStatus.RUNNER_ERROR,
                InfrastructureStatus.UNKNOWN: AttemptStatus.UNCERTAIN,
            }[result.infrastructure_status]
        return {
            ExecutionStatus.FINISHED: AttemptStatus.FINISHED,
            ExecutionStatus.TIMEOUT: AttemptStatus.TIMEOUT_UNKNOWN,
            ExecutionStatus.DEVICE_ERROR: AttemptStatus.DEVICE_ERROR,
            ExecutionStatus.RUNNER_ERROR: AttemptStatus.RUNNER_ERROR,
            ExecutionStatus.CANCELLED: AttemptStatus.CANCELLED,
            ExecutionStatus.UNKNOWN: AttemptStatus.UNCERTAIN,
            ExecutionStatus.RUNNING: AttemptStatus.UNCERTAIN,
        }[result.execution_status]
