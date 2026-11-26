import asyncio
import copy
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ._analysis_llm import StructuredAnalysisError
from .config import OODExperimentConfig
from .experiment_plan import ExperimentPlanStatus
from .memory import RecoveryMemory, experience_version_id
from .recovery_pipeline import RecoveryWorkflow, _attempt_result
from .runner import (
    CancelResult,
    DeviceRunner,
    ExecutionStatus,
    RunnerHealth,
    RunResult,
)
from .runner_adapter import RunAttemptContext, RunnerAdapter, RunnerResultNotReady
from .schemas import (
    AttemptRequest,
    AttemptResult,
    AttemptStatus,
    BaselinePlan,
    DictSerializable,
    EvidenceRef,
    ExperimentCard,
    RecoveryState,
    content_hash,
)
from .storage import atomic_write_json
from .verdict import AttemptVerdictAnalyzer


class ExperimentExecutionError(ValueError):
    pass


class RunnerNotConfiguredError(ExperimentExecutionError):
    pass


class ExperimentBudgetExceeded(ExperimentExecutionError):
    pass


@dataclass
class ExperimentExecutionResult(DictSerializable):
    experiment_id: str
    status: str
    attempt_results: List[AttemptResult] = field(default_factory=list)
    feedback: List[AttemptResult] = field(default_factory=list)
    evidence_refs: List[EvidenceRef] = field(default_factory=list)
    pending_keys: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


class ExperimentExecutor:
    def __init__(
        self,
        memory: RecoveryMemory,
        baseline_plan: BaselinePlan,
        runner: Optional[DeviceRunner] = None,
        verdict_analyzer: Optional[AttemptVerdictAnalyzer] = None,
        config: Optional[OODExperimentConfig] = None,
        environment: Optional[Dict[str, Any]] = None,
        safety_context: Optional[Dict[str, Any]] = None,
    ):
        if baseline_plan.intent_id != memory.intent_id:
            raise ExperimentExecutionError("冻结计划与 RecoveryMemory intent 不一致")
        self.memory = memory
        self.baseline_plan = baseline_plan
        self.runner = runner
        self.verdict_analyzer = verdict_analyzer
        self.config = config or OODExperimentConfig()
        self.environment = dict(environment or {})
        self.safety_context = dict(safety_context or {})
        self.adapter = RunnerAdapter()
        self.workflow = RecoveryWorkflow(memory.root)

    async def aexecute(
        self,
        plan_item_id: str,
        instruction_ids: Optional[List[str]] = None,
        attempts_per_instruction: int = 1,
    ) -> ExperimentExecutionResult:
        snapshot = self.workflow.resume_pending(self.memory)
        if snapshot.state is not RecoveryState.EXPERIENCE_EXPLORING:
            raise ExperimentExecutionError("只有 EXPERIENCE_EXPLORING 可以执行候选经验实验")
        if self.runner is None:
            raise RunnerNotConfiguredError("未配置 DeviceRunner，实验保持 NOT_DISPATCHED")
        if snapshot.current_experience_id is None:
            raise ExperimentExecutionError("当前没有可执行的经验版本")
        if (
            isinstance(attempts_per_instruction, bool)
            or not isinstance(attempts_per_instruction, int)
            or attempts_per_instruction <= 0
        ):
            raise ExperimentExecutionError("attempts_per_instruction 必须是正整数")

        experiment_plan = self.memory.load_plan()
        item = experiment_plan.get_item(plan_item_id)
        if item.status not in {ExperimentPlanStatus.PENDING, ExperimentPlanStatus.ACTIVE}:
            raise ExperimentExecutionError("只有 PENDING 或 ACTIVE 计划项可以执行")
        current = self.memory.experiences.get(snapshot.current_experience_id).experience
        if item.status is ExperimentPlanStatus.ACTIVE and (
            item.experience_version_id != snapshot.current_experience_id
        ):
            raise ExperimentExecutionError("活动计划项与当前经验版本不一致")

        samples = {sample.instruction_id: sample for sample in self.baseline_plan.samples}
        selected_ids = list(instruction_ids or samples)
        if not selected_ids or len(set(selected_ids)) != len(selected_ids):
            raise ExperimentExecutionError("instruction_ids 必须非空且不能重复")
        unknown = set(selected_ids) - set(samples)
        if unknown:
            raise ExperimentExecutionError(f"实验引用了冻结计划外指令: {sorted(unknown)}")
        total_attempts = len(selected_ids) * attempts_per_instruction
        if total_attempts > self.config.max_attempts_per_experiment:
            raise ExperimentBudgetExceeded("单次实验 attempt 数超过配置上限")

        experiment_id = self._experiment_id(
            item.item_id,
            snapshot.current_experience_id,
            selected_ids,
            attempts_per_instruction,
        )
        if item.status is ExperimentPlanStatus.ACTIVE and item.experiment_id != experiment_id:
            raise ExperimentExecutionError("活动实验的指令或次数不能在执行中变更")
        card = ExperimentCard(
            experiment_id=experiment_id,
            intent_id=self.memory.intent_id,
            hypothesis=item.hypothesis,
            baseline_plan_id=self.baseline_plan.plan_id,
            experience=current,
            instruction_ids=selected_ids,
            attempts_per_instruction=attempts_per_instruction,
            expected_stage=current.expected_stage,
            falsification_criteria=list(item.falsification_conditions),
            environment=dict(self.environment),
        )
        self.memory.experiments.record_plan(card)
        if item.status is ExperimentPlanStatus.PENDING:
            experiment_plan.activate(
                item.item_id,
                reason="通过受控 ExperimentExecutor 预占并登记候选经验实验",
                experience_version_id=snapshot.current_experience_id,
                experiment_id=experiment_id,
            )
            self.memory.save_plan(experiment_plan)
        snapshot = self._reserve_budget(snapshot, experiment_id, total_attempts)

        requests = [
            AttemptRequest(
                intent_id=self.memory.intent_id,
                plan_id=self.baseline_plan.plan_id,
                instruction_id=instruction_id,
                instruction=samples[instruction_id].instruction,
                attempt_index=attempt_index,
                idempotency_key=content_hash({
                    "experiment_id": experiment_id,
                    "instruction_id": instruction_id,
                    "attempt_index": attempt_index,
                    "experience_version_id": snapshot.current_experience_id,
                }),
                experience=current,
                environment=dict(self.environment),
            )
            for instruction_id in selected_ids
            for attempt_index in range(1, attempts_per_instruction + 1)
        ]
        return await self._execute_requests(card, requests)

    async def _execute_requests(
        self,
        card: ExperimentCard,
        requests: List[AttemptRequest],
    ) -> ExperimentExecutionResult:
        raw_results = []
        pending_keys = []
        issues = []
        existing = self._attempt_results(card.experiment_id)
        for request in requests:
            if request.idempotency_key in existing:
                attempt = existing[request.idempotency_key]
                raw_results.append(attempt)
                if not await self._ensure_verdict(card.experiment_id, attempt):
                    issues.append(f"VERDICT_UNAVAILABLE:{request.idempotency_key}")
                continue
            run_request = self.adapter.to_run_request(
                request,
                RunAttemptContext(
                    app=self.baseline_plan.intent_key.app,
                    category=self.baseline_plan.intent_key.category,
                    intent_name=self.baseline_plan.intent_key.intent_name,
                    run_id=f"run-{card.experiment_id}",
                    experiment_id=card.experiment_id,
                    timeout_sec=self.config.runner_timeout_sec,
                    device_constraints=dict(self.environment.get("device_constraints", {})),
                    safety_context=dict(self.safety_context),
                    initial_state=str(self.environment.get("initial_state", "clean_app_start")),
                    metadata={"environment": dict(self.environment)},
                ),
            )
            try:
                runner_result = await self.runner.run_attempt(run_request)
                if runner_result.execution_status is ExecutionStatus.RUNNING:
                    pending_keys.append(request.idempotency_key)
                    break
                self._save_runner_result(runner_result)
                attempt = self.adapter.to_attempt_result(run_request, request, runner_result)
            except asyncio.CancelledError:
                raise
            except RunnerResultNotReady:
                pending_keys.append(request.idempotency_key)
                break
            except Exception as error:
                attempt = AttemptResult(
                    request=request,
                    status=AttemptStatus.RUNNER_ERROR,
                    task_completed=None,
                    trajectory={},
                    environment={"adapter_error_type": type(error).__name__},
                    error=f"{type(error).__name__}: {error}",
                )
                issues.append(f"RUNNER_OR_ADAPTER_ERROR:{request.idempotency_key}")
            self.memory.experiments.record_attempt(card.experiment_id, attempt)
            raw_results.append(attempt)
            existing[request.idempotency_key] = attempt
            if not await self._ensure_verdict(card.experiment_id, attempt):
                issues.append(f"VERDICT_UNAVAILABLE:{request.idempotency_key}")

        feedback, evidence = self.feedback_for_experiment(card.experiment_id)
        return ExperimentExecutionResult(
            experiment_id=card.experiment_id,
            status="RUNNING" if pending_keys else "COLLECTED",
            attempt_results=raw_results,
            feedback=feedback,
            evidence_refs=evidence,
            pending_keys=pending_keys,
            issues=issues,
        )

    async def _ensure_verdict(self, experiment_id: str, attempt: AttemptResult) -> bool:
        if attempt.status is not AttemptStatus.FINISHED or self.verdict_analyzer is None:
            return True
        if self._verdict_payload(experiment_id, attempt.request.idempotency_key) is not None:
            return True
        try:
            verdict = await self.verdict_analyzer.aanalyze(attempt)
        except StructuredAnalysisError:
            return False
        self.memory.experiments.record_verdict(experiment_id, verdict)
        return True

    def feedback_for_experiment(
        self,
        experiment_id: str,
    ) -> Tuple[List[AttemptResult], List[EvidenceRef]]:
        raw_by_key = self._attempt_results(experiment_id)
        feedback = []
        evidence = []
        for attempt_key, raw in raw_by_key.items():
            verdict = self._verdict_payload(experiment_id, attempt_key)
            if verdict is None:
                feedback.append(raw)
                continue
            feedback.append(replace(
                raw,
                status=AttemptStatus(verdict["completion_status"]),
                task_completed=verdict["task_completed"],
            ))
            evidence.append(EvidenceRef(
                sessionid=verdict["sessionid"],
                step_ids=list(verdict["key_evidence_steps"]),
            ))
        return feedback, evidence

    def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        events = self.memory.experiments.query(experiment_id=experiment_id, limit=50)
        if not events:
            return {"status": "NOT_FOUND", "experiment_id": experiment_id, "events": []}
        feedback, evidence = self.feedback_for_experiment(experiment_id)
        return {
            "status": "OK",
            "experiment_id": experiment_id,
            "events": [event.to_dict() for event in events],
            "feedback": [item.to_dict() for item in feedback],
            "evidence_refs": [item.to_dict() for item in evidence],
        }

    async def cancel(self, idempotency_key: str) -> CancelResult:
        if self.runner is None:
            raise RunnerNotConfiguredError("未配置 DeviceRunner，不能取消实验")
        return await self.runner.cancel(idempotency_key)

    async def health_check(self) -> RunnerHealth:
        if self.runner is None:
            raise RunnerNotConfiguredError("未配置 DeviceRunner，不能检查执行环境")
        return await self.runner.health_check()

    def _reserve_budget(
        self,
        snapshot,
        experiment_id: str,
        attempt_count: int,
    ):
        budget = copy.deepcopy(snapshot.budget)
        reserved = budget.get("reserved_experiment_ids", [])
        if experiment_id in reserved:
            return snapshot
        max_experiments = min(
            self.config.max_experiments,
            int(budget.get("max_experiments", self.config.max_experiments)),
        )
        used_experiments = int(budget.get("experiments_used", len(reserved)))
        if used_experiments + 1 > max_experiments:
            raise ExperimentBudgetExceeded("实验次数预算已耗尽")
        max_attempts = int(budget.get("max_total_attempts", self.config.max_total_attempts))
        used_attempts = int(budget.get("attempts_used", 0))
        if used_attempts + attempt_count > max_attempts:
            raise ExperimentBudgetExceeded("真机 attempt 总预算不足")
        reserved_counts = dict(budget.get("reserved_attempt_counts", {}))
        reserved.append(experiment_id)
        reserved_counts[experiment_id] = attempt_count
        budget.update({
            "experiments_used": used_experiments + 1,
            "attempts_used": used_attempts + attempt_count,
            "reserved_experiment_ids": reserved,
            "reserved_attempt_counts": reserved_counts,
        })
        return self.workflow.commit_exploration_state(
            self.memory,
            RecoveryState.EXPERIENCE_EXPLORING,
            stage="EXPERIMENT_BUDGET_RESERVED",
            reason=f"预占实验 {experiment_id} 及 {attempt_count} 个 attempt",
            budget=budget,
            snapshot=snapshot,
        )

    def _attempt_results(self, experiment_id: str) -> Dict[str, AttemptResult]:
        results = {}
        for event in self.memory.experiments.query(
            experiment_id=experiment_id,
            event_type="ATTEMPT_RESULT_RECORDED",
            limit=50,
        ):
            attempt = _attempt_result(event.payload["result"])
            results[attempt.request.idempotency_key] = attempt
        return results

    def _verdict_payload(self, experiment_id: str, attempt_key: str) -> Optional[Dict[str, Any]]:
        for event in self.memory.experiments.query(
            experiment_id=experiment_id,
            event_type="VERDICT_RECORDED",
            limit=50,
        ):
            if event.payload["verdict"]["attempt_key"] == attempt_key:
                return event.payload["verdict"]
        return None

    def _save_runner_result(self, result: RunResult) -> None:
        path = self.memory.root / "runner_results" / f"{result.idempotency_key}.json"
        payload = result.to_dict()
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ExperimentExecutionError("既有 Runner 原始结果损坏") from error
            if existing != payload:
                raise ExperimentExecutionError("同一幂等键的 Runner 原始结果不一致")
            return
        atomic_write_json(path, payload)

    @staticmethod
    def _experiment_id(
        plan_item_id: str,
        version_id: str,
        instruction_ids: List[str],
        attempts_per_instruction: int,
    ) -> str:
        return "experiment-" + content_hash({
            "plan_item_id": plan_item_id,
            "experience_version_id": version_id,
            "instruction_ids": list(instruction_ids),
            "attempts_per_instruction": attempts_per_instruction,
        })[:20]
