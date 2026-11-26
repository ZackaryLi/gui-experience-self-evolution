"""顶层意图编排器：把一个 OOD 意图从分诊产物驱动到终态。

串联三段已有组件（均为确定性编排，不新增决策逻辑）：

1. **第一层短路**：读取 ``batch_value.json``——非 ``RUN_BASELINE`` 决策直接出报告，
   不进入恢复流程（批量价值判断已判定的意图不重复消耗预算）。
2. **基线下发（Phase A）**：把冻结 ``baseline_plan.json`` 的无经验 attempt 经
   ``RunnerAdapter`` 逐条下发给 ``DeviceRunner``，结果增量落盘
   ``baseline_results.jsonl``；崩溃恢复时只补发缺失的幂等键。
3. **固定前置链（Phase B）**：``RecoveryWorkflow.arun_fixed_preparation``
   完成基线分析 → 第二层价值判断 → version=1 经验注册（含产物复用与 journal 恢复）。
4. **Agent 探索循环（Phase C）**：驱动 ``RecoveryHarness.arun_iteration``
   直到 ``RecoveryState`` 进入终态 / VALIDATING（等待 s17）/ 预算耗尽 /
   连续空转 / 达到轮数上限。

编排器自身不解释 Agent 决策、不判业务成败、不绕过任何状态机；
所有状态变化仍由 Workflow 的 journal + 原子提交完成，崩溃后重新调用
``arun_intent()`` 即可从断点续跑。
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .baseline import BaselinePlanner
from .batch_value import BatchRunDecision
from .config import OODExperimentConfig
from .experiments import RunnerNotConfiguredError
from .hooks import ToolHook
from .memory import RecoveryMemory
from .recovery_harness import RecoveryHarness
from .recovery_pipeline import RecoveryWorkflow, _attempt_result
from .runner import DeviceRunner
from .runner_adapter import RunnerAdapter, RunAttemptContext
from .schemas import (
    AttemptStatus,
    BaselinePlan,
    DictSerializable,
    IntentKey,
    LoadedTrajectory,
    RecoveryState,
    RecoveryStateSnapshot,
    content_hash,
)
from .storage import append_jsonl, atomic_write_json
from .verdict import AttemptVerdictAnalyzer


ORCHESTRATOR_SCHEMA_VERSION = "1.0"

_BASELINE_RESULTS_FILE = "baseline_results.jsonl"
_REPORT_FILE = "intent_run_report.json"

# 编排器观察到的终态：这些状态不需要再驱动迭代。
_TERMINAL_STATES = {
    RecoveryState.STOP_BLOCKED,
    RecoveryState.STOP_UNSOLVABLE,
    RecoveryState.STOP_RETRYABLE,
    RecoveryState.MANUAL_REVIEW,
    RecoveryState.SUCCEEDED,
}
# 编排器可以继续驱动的中间状态。
_ACTIVE_STATES = {
    RecoveryState.BASELINE_PENDING,
    RecoveryState.BASELINE_ANALYZED,
    RecoveryState.RECOVERY_VALUE_CHECKED,
    RecoveryState.EXPERIENCE_EXPLORING,
}


class IntentOrchestratorError(ValueError):
    """编排输入或状态不合法。"""


@dataclass
class IterationRecord(DictSerializable):
    """一轮探索迭代的摘要记录。"""

    round_index: int
    agent_status: str
    decision_applied: bool
    reason: str = ""


@dataclass
class IntentRunReport(DictSerializable):
    """一次意图编排的最终报告，落盘 ``intent_run_report.json``。"""

    schema_version: str
    intent_id: str
    intent_key: Dict[str, str]
    final_state: str
    final_reason: str
    batch_value_decision: str
    runner_status: str
    baseline_attempt_count: int
    rounds_executed: int
    iterations: List[IterationRecord] = field(default_factory=list)
    experience_versions: List[str] = field(default_factory=list)
    experiment_ids: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    recovery_dir: str = ""
    created_at: str = ""


class OODIntentOrchestrator:
    """单意图顶层编排：分诊产物 → 基线下发 → 固定前置链 → Agent 探索循环。"""

    def __init__(
        self,
        intent_dir: str | Path,
        recovery_dir: str | Path,
        llm_agent,
        model: str,
        runner: Optional[DeviceRunner] = None,
        execution_authorized: bool = False,
        experiment_config: Optional[OODExperimentConfig] = None,
        agent_config=None,
        baseline_analyzer=None,
        recovery_value_assessor=None,
        initial_experience_generator=None,
        patch_generator=None,
        verdict_analyzer: Optional[AttemptVerdictAnalyzer] = None,
        environment: Optional[Dict[str, Any]] = None,
        safety_context: Optional[Dict[str, Any]] = None,
        tool_hooks: Optional[List[ToolHook]] = None,
        max_agent_rounds: int = 12,
        max_stalled_rounds: int = 2,
    ):
        if max_agent_rounds < 1:
            raise IntentOrchestratorError("max_agent_rounds 必须是正数")
        if max_stalled_rounds < 1:
            raise IntentOrchestratorError("max_stalled_rounds 必须是正数")
        self.intent_dir = Path(intent_dir)
        self.recovery_dir = Path(recovery_dir)
        self.recovery_dir.mkdir(parents=True, exist_ok=True)
        self.llm = llm_agent
        self.model = model
        self.runner = runner
        self.execution_authorized = execution_authorized
        self.experiment_config = experiment_config or OODExperimentConfig()
        self.agent_config = agent_config
        self.baseline_analyzer = baseline_analyzer
        self.recovery_value_assessor = recovery_value_assessor
        self.initial_experience_generator = initial_experience_generator
        self.patch_generator = patch_generator
        self.verdict_analyzer = verdict_analyzer
        self.environment = environment
        self.safety_context = safety_context
        self.tool_hooks = tuple(tool_hooks or ())
        self.max_agent_rounds = max_agent_rounds
        self.max_stalled_rounds = max_stalled_rounds
        self._notes: List[str] = []
        self._iterations: List[IterationRecord] = []

    # ------------------------------------------------------------------
    # 公开入口
    # ------------------------------------------------------------------

    def run_intent(self) -> IntentRunReport:
        return asyncio.run(self.arun_intent())

    async def arun_intent(self) -> IntentRunReport:
        self._notes = []
        self._iterations = []
        key, intent_id = self._load_intent_identity()
        self._memory = RecoveryMemory(self.recovery_dir, intent_id)
        snapshot = self._recover_or_none()
        batch_decision = self._load_batch_value_decision(intent_id)

        if batch_decision != BatchRunDecision.RUN_BASELINE.value:
            return self._report(
                key=key,
                snapshot=snapshot,
                batch_decision=batch_decision,
                final_state=(snapshot.state.value if snapshot else "TRIAGED"),
                final_reason="第一层批量价值判断未给出 RUN_BASELINE，意图不进入恢复流程",
                runner_status="NOT_DISPATCHED",
                baseline_attempt_count=0,
            )

        plan, _ = self._load_frozen_plan(key)

        # -------- Phase A：基线无经验 attempt 下发（可跳过/可续跑） --------
        baseline_attempt_count = 0
        runner_status = "NOT_DISPATCHED"
        results = self._load_baseline_results(intent_id, plan)
        needs_baseline = (
            snapshot is None
            or (
                snapshot.state is RecoveryState.BASELINE_PENDING
                and not (self.recovery_dir / "diagnosis.json").is_file()
            )
        )
        if needs_baseline:
            missing = [
                request
                for request in plan.to_attempt_requests()
                if request.idempotency_key not in {item.request.idempotency_key for item in results}
            ]
            if missing and self.runner is None:
                self._note(
                    f"基线尚有 {len(missing)} 个 attempt 未执行，且未配置 DeviceRunner，"
                    "保持 NOT_DISPATCHED"
                )
            elif missing:
                runner_status = "DISPATCHED"
                dispatched = await self._dispatch_baseline(plan, key, missing)
                results = results + dispatched
            elif self.runner is not None:
                runner_status = "REUSED"
            results = self._reorder_results(plan, results)
            baseline_attempt_count = len(results)
            if not results:
                return self._report(
                    key=key,
                    snapshot=snapshot,
                    batch_decision=batch_decision,
                    final_state=(
                        snapshot.state.value if snapshot else RecoveryState.BASELINE_PENDING.value
                    ),
                    final_reason="基线没有任何可用的执行结果，无法进入固定分析",
                    runner_status=runner_status,
                    baseline_attempt_count=0,
                )
        else:
            baseline_attempt_count = len(results)
            if snapshot is not None and snapshot.revision > 0:
                runner_status = "REUSED"
            self._note("基线阶段产物已存在，跳过基线下发")

        # -------- Phase B：固定前置链（分析 → 价值判断 → v1 经验） --------
        workflow = RecoveryWorkflow(
            self.recovery_dir,
            baseline_analyzer=self.baseline_analyzer,
            recovery_value_assessor=self.recovery_value_assessor,
            initial_experience_generator=self.initial_experience_generator,
        )
        budget = {
            "max_experiments": self.experiment_config.max_experiments,
            "max_total_attempts": self.experiment_config.max_total_attempts,
        }
        preparation = await workflow.arun_fixed_preparation(
            plan=plan,
            results=results if needs_baseline else None,
            # 已有状态时不能再传预算：持久化预算含 experiments_used 等运行数据，
            # 重复传入会被 StateStore 视为“用新预算覆盖”而冲突。
            budget=budget if snapshot is None else None,
        )
        snapshot = preparation.snapshot
        if snapshot.state not in _ACTIVE_STATES:
            # 固定链已把意图带到终态（NO_REPAIR_NEEDED / STOP_* / MANUAL_REVIEW）。
            return self._report(
                key=key,
                snapshot=snapshot,
                batch_decision=batch_decision,
                final_state=snapshot.state.value,
                final_reason=self._final_reason(snapshot, preparation),
                runner_status=runner_status,
                baseline_attempt_count=baseline_attempt_count,
            )
        if snapshot.state is not RecoveryState.EXPERIENCE_EXPLORING:
            raise IntentOrchestratorError(
                f"固定前置链停在意外中间状态: {snapshot.state.value}"
            )

        # -------- Phase C：Agent 探索循环 --------
        report = await self._run_exploration_loop(
            key=key,
            plan=plan,
            preparation=preparation,
            batch_decision=batch_decision,
            runner_status=runner_status,
            baseline_attempt_count=baseline_attempt_count,
        )
        return report

    # ------------------------------------------------------------------
    # Phase A：基线下发
    # ------------------------------------------------------------------

    async def _dispatch_baseline(
        self,
        plan: BaselinePlan,
        key: IntentKey,
        requests: List,
    ) -> List:
        adapter = RunnerAdapter()
        context = RunAttemptContext(
            app=key.app,
            run_id=f"baseline-{plan.plan_id}",
            experiment_id=plan.plan_id,
            category=key.category,
            intent_name=key.intent_name,
            timeout_sec=self.experiment_config.runner_timeout_sec,
            metadata={"phase": "baseline", "baseline_plan_id": plan.plan_id},
        )
        dispatched = []
        for request in requests:
            result = await adapter.arun_attempt(self.runner, request, context)
            append_jsonl(
                self.recovery_dir / _BASELINE_RESULTS_FILE,
                result.to_dict(),
            )
            dispatched.append(result)
            if result.status not in {
                AttemptStatus.FINISHED,
                AttemptStatus.COMPLETED,
                AttemptStatus.FAILED,
                AttemptStatus.UNCERTAIN,
            }:
                self._note(
                    f"基线 attempt {request.instruction_id}#{request.attempt_index} "
                    f"返回基础设施状态 {result.status.value}，将按异常样本参与分析"
                )
        return dispatched

    def _load_baseline_results(
        self,
        intent_id: str,
        plan: BaselinePlan,
    ) -> List:
        """读取已落盘的基线结果；崩溃恢复时只补发缺失幂等键。"""
        path = self.recovery_dir / _BASELINE_RESULTS_FILE
        if not path.is_file():
            return []
        results = []
        for payload in _read_jsonl(path):
            result = _attempt_result(payload)
            if result.request.intent_id != intent_id:
                raise IntentOrchestratorError(
                    "基线结果文件包含其他意图的 attempt"
                )
            if result.request.plan_id != plan.plan_id:
                raise IntentOrchestratorError(
                    "基线结果与冻结计划身份不一致，禁止混用"
                )
            results.append(result)
        return results

    @staticmethod
    def _reorder_results(plan: BaselinePlan, results: List) -> List:
        """按冻结计划顺序排列结果，缺失位置剔除（分析层按身份校验）。"""
        by_key = {item.request.idempotency_key: item for item in results}
        ordered = []
        for request in plan.to_attempt_requests():
            item = by_key.get(request.idempotency_key)
            if item is not None:
                ordered.append(item)
        return ordered

    # ------------------------------------------------------------------
    # Phase C：探索循环
    # ------------------------------------------------------------------

    async def _run_exploration_loop(
        self,
        key: IntentKey,
        plan: BaselinePlan,
        preparation,
        batch_decision: str,
        runner_status: str,
        baseline_attempt_count: int,
    ) -> IntentRunReport:
        harness = RecoveryHarness(
            recovery_dir=str(self.recovery_dir),
            baseline_plan=plan,
            diagnosis=preparation.diagnosis,
            recovery_value=preparation.recovery_value,
            llm_agent=self.llm,
            model=self.model,
            runner=self.runner,
            execution_authorized=self.execution_authorized,
            experiment_config=self.experiment_config,
            agent_config=self.agent_config,
            patch_generator=self.patch_generator,
            verdict_analyzer=self.verdict_analyzer,
            environment=self.environment,
            safety_context=self.safety_context,
            tool_hooks=self.tool_hooks,
        )
        stalled = 0
        rounds = 0
        snapshot = preparation.snapshot
        final_reason = ""
        for _ in range(self.max_agent_rounds):
            snapshot = self._memory.recover()
            if snapshot.state in _TERMINAL_STATES:
                final_reason = snapshot.blocker_reason or "Agent 循环已达终态"
                break
            if snapshot.state is RecoveryState.VALIDATING:
                final_reason = "经验已请求验证，等待 s17 独立目标闸门"
                self._note("VALIDATING 需要 s17 目标验证，编排器在此停止驱动")
                break
            budget_blocker = self._budget_blocker(snapshot)
            if budget_blocker is not None:
                final_reason = budget_blocker
                self._note(budget_blocker)
                break

            event_count = len(self._memory.experiments.load())
            revision = snapshot.revision
            iteration = await harness.arun_iteration()
            rounds += 1
            result_snapshot = iteration.snapshot
            applied = iteration.decision_applied
            self._iterations.append(IterationRecord(
                round_index=rounds,
                agent_status=iteration.agent_run.status.value,
                decision_applied=applied,
                reason=iteration.application_reason,
            ))
            if result_snapshot.state in _TERMINAL_STATES:
                snapshot = result_snapshot
                final_reason = result_snapshot.blocker_reason or iteration.application_reason
                break
            if (
                not applied
                and result_snapshot.revision == revision
                and len(self._memory.experiments.load()) == event_count
            ):
                stalled += 1
                if stalled >= self.max_stalled_rounds:
                    final_reason = (
                        f"Agent 连续 {stalled} 轮既未应用决策也未产生新实验证据，停止驱动"
                    )
                    self._note(final_reason)
                    break
            else:
                stalled = 0
            snapshot = result_snapshot
        else:
            final_reason = f"达到编排器轮数上限 {self.max_agent_rounds}，转人工检查"
            self._note(final_reason)

        if not final_reason:
            final_reason = snapshot.blocker_reason or "探索循环结束"
        return self._report(
            key=key,
            snapshot=snapshot,
            batch_decision=batch_decision,
            final_state=snapshot.state.value,
            final_reason=final_reason,
            runner_status=runner_status,
            baseline_attempt_count=baseline_attempt_count,
        )

    def _budget_blocker(self, snapshot: RecoveryStateSnapshot) -> Optional[str]:
        budget = snapshot.budget or {}
        max_experiments = int(budget.get(
            "max_experiments", self.experiment_config.max_experiments
        ))
        max_attempts = int(budget.get(
            "max_total_attempts", self.experiment_config.max_total_attempts
        ))
        used_experiments = int(budget.get("experiments_used", 0))
        used_attempts = int(budget.get("attempts_used", 0))
        if used_experiments >= max_experiments:
            return (
                f"实验预算耗尽（{used_experiments}/{max_experiments} 个实验），"
                "保持当前状态转人工"
            )
        if used_attempts >= max_attempts:
            return (
                f"attempt 预算耗尽（{used_attempts}/{max_attempts}），保持当前状态转人工"
            )
        return None

    # ------------------------------------------------------------------
    # 输入加载
    # ------------------------------------------------------------------

    def _load_intent_identity(self) -> tuple:
        triage_path = self.intent_dir / "triage.json"
        if not triage_path.is_file():
            raise IntentOrchestratorError(f"意图目录缺少 triage.json: {self.intent_dir}")
        triage = _read_json(triage_path)
        key = IntentKey(**triage["intent_key"])
        if triage.get("intent_id") != key.intent_id:
            raise IntentOrchestratorError("triage.json 的 intent_id 与 intent_key 不一致")
        return key, key.intent_id

    def _load_batch_value_decision(self, intent_id: str) -> str:
        path = self.intent_dir / "batch_value.json"
        if not path.is_file():
            raise IntentOrchestratorError(
                f"意图缺少第一层批量价值判断产物 batch_value.json: {self.intent_dir}"
            )
        payload = _read_json(path)
        if payload.get("intent_id") != intent_id:
            raise IntentOrchestratorError("batch_value.json 与意图身份不一致")
        decision = payload.get("decision")
        if decision not in {item.value for item in BatchRunDecision}:
            raise IntentOrchestratorError(f"未知的第一层决策: {decision}")
        return decision

    def _load_frozen_plan(self, key: IntentKey) -> tuple:
        plan_path = self.intent_dir / "baseline_plan.json"
        if not plan_path.is_file():
            raise IntentOrchestratorError(
                f"RUN_BASELINE 意图缺少冻结基线计划: {plan_path}"
            )
        candidates_path = self.intent_dir / "exploration_candidates.jsonl"
        if not candidates_path.is_file():
            raise IntentOrchestratorError(
                f"意图缺少候选轨迹文件，无法校验冻结计划: {candidates_path}"
            )
        # 与 batch_pipeline._load_candidates 相同的加载语义：原始记录 + content_hash。
        records = [
            LoadedTrajectory(
                record=payload,
                content_hash=content_hash(payload),
                line_number=index,
            )
            for index, payload in enumerate(_read_jsonl(candidates_path), start=1)
        ]
        plan, reused = BaselinePlanner(
            config=None,
        ).create_or_load(plan_path, key, records)
        if plan.status != "FROZEN":
            raise IntentOrchestratorError("基线计划不是 FROZEN 状态")
        if not reused:
            self._note("冻结计划被重新生成——与复用语义不符，请检查意图目录")
        return plan, reused

    def _recover_or_none(self) -> Optional[RecoveryStateSnapshot]:
        state_path = self.recovery_dir / "state.json"
        if not state_path.is_file():
            return None
        return self._memory.recover()

    # ------------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------------

    def _report(
        self,
        key: IntentKey,
        snapshot: Optional[RecoveryStateSnapshot],
        batch_decision: str,
        final_state: str,
        final_reason: str,
        runner_status: str,
        baseline_attempt_count: int,
    ) -> IntentRunReport:
        experience_versions = []
        experiment_ids = []
        if snapshot is not None:
            experience_versions = [
                record.experience_version_id
                for record in self._memory.experiences.load()
            ]
            seen = set()
            for event in self._memory.experiments.load():
                if event.experiment_id and event.experiment_id not in seen:
                    seen.add(event.experiment_id)
                    experiment_ids.append(event.experiment_id)
        report = IntentRunReport(
            schema_version=ORCHESTRATOR_SCHEMA_VERSION,
            intent_id=key.intent_id,
            intent_key=key.to_dict(),
            final_state=final_state,
            final_reason=final_reason,
            batch_value_decision=batch_decision,
            runner_status=runner_status,
            baseline_attempt_count=baseline_attempt_count,
            rounds_executed=len(self._iterations),
            iterations=list(self._iterations),
            experience_versions=experience_versions,
            experiment_ids=experiment_ids,
            notes=list(self._notes),
            recovery_dir=str(self.recovery_dir),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        atomic_write_json(self.recovery_dir / _REPORT_FILE, report.to_dict())
        return report

    def _note(self, message: str) -> None:
        self._notes.append(message)

    @staticmethod
    def _final_reason(snapshot: RecoveryStateSnapshot, preparation) -> str:
        if snapshot.state is RecoveryState.BASELINE_PENDING:
            return "基线尚未产出可分析的结果，保持 BASELINE_PENDING"
        if preparation.recovery_value is not None:
            return preparation.recovery_value.reason or snapshot.blocker_reason
        return snapshot.blocker_reason


def _read_json(path: Path) -> Dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise IntentOrchestratorError(f"产物必须是 JSON 对象: {path}")
    return payload


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    import json

    payloads = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise IntentOrchestratorError(f"JSONL 行必须是对象: {path}")
        payloads.append(payload)
    return payloads
