import asyncio
import json
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .experiment_plan import ExperimentPlan
from .initial_experience import InitialExperienceError, InitialExperienceResult
from .memory import (
    ArtifactConflictError,
    JournalEntry,
    RecoveryMemory,
    experience_version_id,
    file_sha256,
)
from .recovery_schemas import RecoveryContext
from .schemas import (
    AttemptAnalysis,
    AttemptRequest,
    AttemptResult,
    AttemptStatus,
    AttemptVerdict,
    BaselineAnalysisIssue,
    BaselinePlan,
    BaselineStatistics,
    BusinessStage,
    BusinessStageRubric,
    DiagnosisReport,
    DictSerializable,
    EvidenceRef,
    ExperienceDraft,
    FailurePattern,
    RecoveryState,
    RecoveryStateSnapshot,
    RecoveryValueAssessment,
    RecoveryValueDecision,
    StageProgress,
)
from .storage import atomic_write_json


class RecoveryWorkflowError(ValueError):
    pass


@dataclass
class FixedPreparationResult(DictSerializable):
    snapshot: RecoveryStateSnapshot
    diagnosis: Optional[DiagnosisReport] = None
    recovery_value: Optional[RecoveryValueAssessment] = None
    initial_experience: Optional[InitialExperienceResult] = None
    reused_artifacts: List[str] = None
    runner_status: str = "NOT_DISPATCHED"

    def __post_init__(self):
        if self.reused_artifacts is None:
            self.reused_artifacts = []


_ALLOWED_TRANSITIONS = {
    RecoveryState.BASELINE_PENDING: {RecoveryState.BASELINE_ANALYZED},
    RecoveryState.BASELINE_ANALYZED: {RecoveryState.RECOVERY_VALUE_CHECKED},
    RecoveryState.RECOVERY_VALUE_CHECKED: {
        RecoveryState.EXPERIENCE_EXPLORING,
        RecoveryState.STOP_BLOCKED,
        RecoveryState.STOP_UNSOLVABLE,
        RecoveryState.STOP_RETRYABLE,
        RecoveryState.MANUAL_REVIEW,
    },
    RecoveryState.EXPERIENCE_EXPLORING: {
        RecoveryState.EXPERIENCE_EXPLORING,
        RecoveryState.VALIDATING,
        RecoveryState.STOP_BLOCKED,
        RecoveryState.STOP_RETRYABLE,
        RecoveryState.MANUAL_REVIEW,
    },
}
_TERMINAL_DECISIONS = {
    RecoveryValueDecision.STOP_BLOCKED: RecoveryState.STOP_BLOCKED,
    RecoveryValueDecision.STOP_UNSOLVABLE: RecoveryState.STOP_UNSOLVABLE,
    RecoveryValueDecision.STOP_RETRYABLE: RecoveryState.STOP_RETRYABLE,
    RecoveryValueDecision.MANUAL_REVIEW: RecoveryState.MANUAL_REVIEW,
}


class RecoveryWorkflow:
    def __init__(
        self,
        recovery_dir: str | Path,
        baseline_analyzer=None,
        recovery_value_assessor=None,
        initial_experience_generator=None,
    ):
        self.recovery_dir = Path(recovery_dir)
        self.baseline_analyzer = baseline_analyzer
        self.recovery_value_assessor = recovery_value_assessor
        self.initial_experience_generator = initial_experience_generator
        self._reused_artifacts: List[str] = []

    def run_fixed_preparation(
        self,
        plan: BaselinePlan,
        results: Optional[List[AttemptResult]] = None,
        stage_rubric: Optional[BusinessStageRubric] = None,
        budget: Optional[Dict[str, Any]] = None,
    ) -> FixedPreparationResult:
        return asyncio.run(self.arun_fixed_preparation(
            plan=plan,
            results=results,
            stage_rubric=stage_rubric,
            budget=budget,
        ))

    async def arun_fixed_preparation(
        self,
        plan: BaselinePlan,
        results: Optional[List[AttemptResult]] = None,
        stage_rubric: Optional[BusinessStageRubric] = None,
        budget: Optional[Dict[str, Any]] = None,
    ) -> FixedPreparationResult:
        if not isinstance(plan, BaselinePlan) or plan.status != "FROZEN":
            raise RecoveryWorkflowError("固定前置流程需要 FROZEN BaselinePlan")
        if results is not None and (
            not isinstance(results, list)
            or any(not isinstance(item, AttemptResult) for item in results)
        ):
            raise RecoveryWorkflowError("results 必须是 AttemptResult 数组或 None")
        self._reused_artifacts = []
        memory = RecoveryMemory(self.recovery_dir, plan.intent_id)
        memory.initialize(plan, budget)

        for _ in range(5):
            snapshot = memory.recover()
            pending = memory.pending_transition()
            if pending is not None:
                self._commit_journal_entry(memory, snapshot, pending)
                continue
            if snapshot.state is RecoveryState.BASELINE_PENDING:
                diagnosis = self._load_diagnosis_if_present(plan)
                if diagnosis is None:
                    if not results:
                        return self._result(memory, snapshot)
                    if self.baseline_analyzer is None:
                        raise RecoveryWorkflowError("缺少 BaselineAnalyzer，保持 BASELINE_PENDING")
                    diagnosis = await self.baseline_analyzer.aanalyze(
                        plan=plan,
                        results=results,
                        stage_rubric=stage_rubric,
                        output_path=self.recovery_dir / "diagnosis.json",
                    )
                    self._validate_diagnosis(diagnosis, plan)
                snapshot = self._transition(
                    memory,
                    snapshot,
                    RecoveryState.BASELINE_ANALYZED,
                    stage="BASELINE_ANALYSIS",
                    reason="固定基线诊断产物已写入并通过身份校验",
                    artifacts={"diagnosis.json"},
                )
                continue

            if snapshot.state is RecoveryState.BASELINE_ANALYZED:
                diagnosis = self._require_diagnosis(plan)
                assessment = self._load_assessment_if_present(plan)
                if assessment is None:
                    if self.recovery_value_assessor is None:
                        raise RecoveryWorkflowError(
                            "缺少 RecoveryValueAssessor，保持 BASELINE_ANALYZED"
                        )
                    assessment = await self.recovery_value_assessor.aassess(
                        diagnosis=diagnosis,
                        output_path=self.recovery_dir / "recovery_value.json",
                    )
                    self._validate_assessment(assessment, plan)
                snapshot = self._transition(
                    memory,
                    snapshot,
                    RecoveryState.RECOVERY_VALUE_CHECKED,
                    stage="RECOVERY_VALUE_ASSESSMENT",
                    reason=f"第二层修复价值判断已提交：{assessment.decision.value}",
                    artifacts={"diagnosis.json", "recovery_value.json"},
                    hypothesis=assessment.primary_hypothesis,
                )
                continue

            if snapshot.state is RecoveryState.RECOVERY_VALUE_CHECKED:
                assessment = self._require_assessment(plan)
                if assessment.decision is RecoveryValueDecision.NO_REPAIR_NEEDED:
                    return self._result(memory, snapshot)
                terminal = _TERMINAL_DECISIONS.get(assessment.decision)
                if terminal is not None:
                    snapshot = self._transition(
                        memory,
                        snapshot,
                        terminal,
                        stage="RECOVERY_VALUE_EXIT",
                        reason=assessment.reason,
                        artifacts={"diagnosis.json", "recovery_value.json"},
                    )
                    continue
                if assessment.decision is not RecoveryValueDecision.START_RECOVERY:
                    raise RecoveryWorkflowError("第二层包含未知修复价值决策")

                diagnosis = self._require_diagnosis(plan)
                initial = self._load_initial_if_present(plan, assessment)
                if initial is None:
                    if self.initial_experience_generator is None:
                        raise RecoveryWorkflowError(
                            "缺少 InitialExperienceGenerator，保持 RECOVERY_VALUE_CHECKED"
                        )
                    try:
                        initial = await self.initial_experience_generator.agenerate(
                            diagnosis=diagnosis,
                            assessment=assessment,
                            output_path=self.recovery_dir / "initial_experience.json",
                        )
                    except InitialExperienceError as error:
                        error_path = self.recovery_dir / "initial_experience_error.json"
                        atomic_write_json(error_path, {
                            "intent_id": plan.intent_id,
                            "baseline_plan_id": plan.plan_id,
                            "error_type": type(error).__name__,
                            "message": str(error),
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        })
                        snapshot = self._transition(
                            memory,
                            snapshot,
                            RecoveryState.MANUAL_REVIEW,
                            stage="INITIAL_EXPERIENCE_FAILED",
                            reason="固定第一条经验生成持续失败，禁止转交 Agent 代替生成",
                            artifacts={
                                "diagnosis.json",
                                "recovery_value.json",
                                "initial_experience_error.json",
                            },
                        )
                        continue
                    self._validate_initial(initial, plan, assessment)
                self._register_initial(memory, initial)
                self._ensure_initial_plan(memory, diagnosis, assessment, initial)
                version_id = experience_version_id(initial.experience)
                snapshot = self._transition(
                    memory,
                    snapshot,
                    RecoveryState.EXPERIENCE_EXPLORING,
                    stage="INITIAL_EXPERIENCE_REGISTERED",
                    reason="version=1 候选经验和 PENDING 假设计划已提交，尚未下发实验",
                    artifacts={
                        "diagnosis.json",
                        "recovery_value.json",
                        "initial_experience.json",
                        "plan.json",
                    },
                    experience_version_id=version_id,
                    hypothesis=assessment.primary_hypothesis,
                )
                continue

            return self._result(memory, snapshot)

        raise RecoveryWorkflowError("固定前置流程迁移次数异常，已停止")

    def resume_pending(self, memory: RecoveryMemory) -> RecoveryStateSnapshot:
        snapshot = memory.recover()
        pending = memory.pending_transition()
        if pending is None:
            return snapshot
        return self._commit_journal_entry(memory, snapshot, pending)

    def commit_exploration_state(
        self,
        memory: RecoveryMemory,
        target: RecoveryState,
        stage: str,
        reason: str,
        experience_version_id: Optional[str] = None,
        hypothesis: str = "",
        budget: Optional[Dict[str, Any]] = None,
        round_index: Optional[int] = None,
        snapshot: Optional[RecoveryStateSnapshot] = None,
    ) -> RecoveryStateSnapshot:
        snapshot = snapshot or self.resume_pending(memory)
        current = memory.states.load()
        if current.revision != snapshot.revision or (
            current.last_transition_id != snapshot.last_transition_id
        ):
            raise RecoveryWorkflowError("探索阶段提交使用了过期状态快照")
        if snapshot.state is not RecoveryState.EXPERIENCE_EXPLORING:
            raise RecoveryWorkflowError("只有 EXPERIENCE_EXPLORING 可以提交探索阶段变化")
        return self._transition(
            memory,
            snapshot,
            target,
            stage=stage,
            reason=reason,
            artifacts=set(snapshot.artifact_hashes),
            experience_version_id=experience_version_id,
            hypothesis=hypothesis,
            budget=budget,
            round_index=round_index,
        )

    def _transition(
        self,
        memory: RecoveryMemory,
        snapshot: RecoveryStateSnapshot,
        target: RecoveryState,
        stage: str,
        reason: str,
        artifacts: Dict[str, Any],
        experience_version_id: Optional[str] = None,
        hypothesis: str = "",
        budget: Optional[Dict[str, Any]] = None,
        round_index: Optional[int] = None,
    ) -> RecoveryStateSnapshot:
        if target not in _ALLOWED_TRANSITIONS.get(snapshot.state, set()):
            raise RecoveryWorkflowError(
                f"非法 Workflow 状态迁移：{snapshot.state.value} → {target.value}"
            )
        artifact_hashes = {
            name: file_sha256(self.recovery_dir / name)
            for name in artifacts
        }
        entry = JournalEntry.create(
            sequence=len(memory.journal.load()) + 1,
            snapshot=snapshot,
            to_state=target,
            stage=stage,
            reason=reason,
            artifact_hashes=artifact_hashes,
            experience_version_id=experience_version_id,
            hypothesis=hypothesis,
            budget=budget,
            round_index=round_index,
        )
        memory.journal.append(entry)
        return self._commit_journal_entry(memory, snapshot, entry)

    @staticmethod
    def _commit_journal_entry(
        memory: RecoveryMemory,
        snapshot: RecoveryStateSnapshot,
        entry: JournalEntry,
    ) -> RecoveryStateSnapshot:
        if (
            entry.intent_id != snapshot.intent_id
            or entry.from_state is not snapshot.state
            or entry.from_revision != snapshot.revision
            or entry.to_state not in _ALLOWED_TRANSITIONS.get(snapshot.state, set())
        ):
            raise RecoveryWorkflowError("待提交 journal 与当前状态或合法迁移不一致")
        target = entry.to_state
        if target is RecoveryState.RECOVERY_VALUE_CHECKED:
            allowed_actions = ["GENERATE_INITIAL_EXPERIENCE"] if entry.hypothesis else []
        else:
            allowed_actions = {
                RecoveryState.BASELINE_ANALYZED: ["ASSESS_RECOVERY_VALUE"],
                RecoveryState.EXPERIENCE_EXPLORING: ["AGENT_DECISION_REQUIRED"],
                RecoveryState.VALIDATING: ["GOAL_EVALUATION_REQUIRED"],
            }.get(target, [])
        updated = replace(
            snapshot,
            state=target,
            revision=entry.to_revision,
            current_experience_id=(
                entry.experience_version_id or snapshot.current_experience_id
            ),
            current_hypothesis=entry.hypothesis or snapshot.current_hypothesis,
            budget=dict(entry.budget) if entry.budget is not None else dict(snapshot.budget),
            round_index=(
                entry.round_index if entry.round_index is not None else snapshot.round_index
            ),
            blocker_reason=entry.reason if target in {
                RecoveryState.STOP_BLOCKED,
                RecoveryState.STOP_UNSOLVABLE,
                RecoveryState.STOP_RETRYABLE,
                RecoveryState.MANUAL_REVIEW,
            } else "",
            allowed_actions=allowed_actions,
            artifact_hashes=dict(entry.artifact_hashes),
            last_transition_id=entry.transition_id,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        committed, _ = memory.states.commit(
            updated,
            expected_revision=snapshot.revision,
            transition_id=entry.transition_id,
        )
        return committed

    def _result(
        self,
        memory: RecoveryMemory,
        snapshot: RecoveryStateSnapshot,
    ) -> FixedPreparationResult:
        diagnosis = None
        assessment = None
        initial = None
        if "diagnosis.json" in snapshot.artifact_hashes:
            diagnosis = _diagnosis_from_dict(_read_artifact(
                self.recovery_dir / "diagnosis.json"
            ))
        if "recovery_value.json" in snapshot.artifact_hashes:
            assessment = _assessment_from_dict(_read_artifact(
                self.recovery_dir / "recovery_value.json"
            ))
        if "initial_experience.json" in snapshot.artifact_hashes:
            initial = _initial_from_dict(_read_artifact(
                self.recovery_dir / "initial_experience.json"
            ))
        return FixedPreparationResult(
            snapshot=snapshot,
            diagnosis=diagnosis,
            recovery_value=assessment,
            initial_experience=initial,
            reused_artifacts=list(self._reused_artifacts),
        )

    def _load_diagnosis_if_present(self, plan: BaselinePlan) -> Optional[DiagnosisReport]:
        path = self.recovery_dir / "diagnosis.json"
        if not path.exists():
            return None
        self._reused_artifacts.append(path.name)
        diagnosis = _diagnosis_from_dict(_read_artifact(path))
        self._validate_diagnosis(diagnosis, plan)
        return diagnosis

    def _require_diagnosis(self, plan: BaselinePlan) -> DiagnosisReport:
        diagnosis = self._load_diagnosis_if_present(plan)
        if diagnosis is None:
            raise RecoveryWorkflowError("状态要求 diagnosis.json，但产物不存在")
        return diagnosis

    def _load_assessment_if_present(
        self,
        plan: BaselinePlan,
    ) -> Optional[RecoveryValueAssessment]:
        path = self.recovery_dir / "recovery_value.json"
        if not path.exists():
            return None
        self._reused_artifacts.append(path.name)
        assessment = _assessment_from_dict(_read_artifact(path))
        self._validate_assessment(assessment, plan)
        return assessment

    def _require_assessment(self, plan: BaselinePlan) -> RecoveryValueAssessment:
        assessment = self._load_assessment_if_present(plan)
        if assessment is None:
            raise RecoveryWorkflowError("状态要求 recovery_value.json，但产物不存在")
        return assessment

    def _load_initial_if_present(
        self,
        plan: BaselinePlan,
        assessment: RecoveryValueAssessment,
    ) -> Optional[InitialExperienceResult]:
        path = self.recovery_dir / "initial_experience.json"
        if not path.exists():
            return None
        self._reused_artifacts.append(path.name)
        initial = _initial_from_dict(_read_artifact(path))
        self._validate_initial(initial, plan, assessment)
        return initial

    @staticmethod
    def _validate_diagnosis(diagnosis: DiagnosisReport, plan: BaselinePlan) -> None:
        if not isinstance(diagnosis, DiagnosisReport):
            raise RecoveryWorkflowError("分析器没有返回 DiagnosisReport")
        if diagnosis.intent_id != plan.intent_id or diagnosis.baseline_plan_id != plan.plan_id:
            raise ArtifactConflictError("诊断产物与当前冻结计划身份不一致")

    @staticmethod
    def _validate_assessment(
        assessment: RecoveryValueAssessment,
        plan: BaselinePlan,
    ) -> None:
        if not isinstance(assessment, RecoveryValueAssessment):
            raise RecoveryWorkflowError("价值判断器没有返回 RecoveryValueAssessment")
        if assessment.intent_id != plan.intent_id or assessment.baseline_plan_id != plan.plan_id:
            raise ArtifactConflictError("第二层产物与当前冻结计划身份不一致")
        if not isinstance(assessment.decision, RecoveryValueDecision):
            raise RecoveryWorkflowError("第二层决策类型非法")

    @staticmethod
    def _validate_initial(
        initial: InitialExperienceResult,
        plan: BaselinePlan,
        assessment: RecoveryValueAssessment,
    ) -> None:
        if not isinstance(initial, InitialExperienceResult):
            raise RecoveryWorkflowError("生成器没有返回 InitialExperienceResult")
        experience = initial.experience
        if experience.intent_id != plan.intent_id or experience.version != 1:
            raise ArtifactConflictError("第一条经验身份或版本非法")
        if experience.parent_experience_id is not None:
            raise ArtifactConflictError("第一条经验不能包含父版本")
        if (
            experience.hypothesis != assessment.primary_hypothesis
            or experience.expected_stage != assessment.expected_stage
            or initial.source_failure_pattern_id != assessment.primary_failure_pattern_id
        ):
            raise ArtifactConflictError("第一条经验与第二层选定方向不一致")

    def _register_initial(
        self,
        memory: RecoveryMemory,
        initial: InitialExperienceResult,
    ) -> None:
        initial_path = self.recovery_dir / "initial_experience.json"
        memory.experiences.register(
            initial.experience,
            source="FIXED_INITIAL_GENERATOR",
            source_failure_pattern_id=initial.source_failure_pattern_id,
            source_details={
                "prompt_version": initial.prompt_version,
                "skill_name": initial.skill_name,
                "skill_version": initial.skill_version,
                "skill_content_hash": initial.skill_content_hash,
                "initial_artifact_sha256": file_sha256(initial_path),
            },
        )

    def _ensure_initial_plan(
        self,
        memory: RecoveryMemory,
        diagnosis: DiagnosisReport,
        assessment: RecoveryValueAssessment,
        initial: InitialExperienceResult,
    ) -> None:
        path = self.recovery_dir / "plan.json"
        if path.exists():
            plan = memory.load_plan()
            if len(plan.items) != 1 or plan.active_item is not None:
                raise ArtifactConflictError("初始 Workflow 只能复用单个 PENDING 假设计划")
            item = plan.items[0]
            if (
                item.hypothesis != assessment.primary_hypothesis
                or item.failure_pattern_refs != [assessment.primary_failure_pattern_id]
                or item.experience_version_id is not None
                or item.experiment_id is not None
            ):
                raise ArtifactConflictError("既有实验计划与当前第一条经验方向不一致")
            return
        context = RecoveryContext(
            intent_id=diagnosis.intent_id,
            state=RecoveryState.EXPERIENCE_EXPLORING,
            diagnosis=diagnosis,
            recovery_value=assessment,
            current_experience=initial.experience,
        )
        plan = ExperimentPlan.create(context)
        plan.add_item(
            hypothesis=assessment.primary_hypothesis,
            research_question=(
                "GUI Agent 携带 version=1 候选经验后，能否到达第二层指定的业务阶段？"
            ),
            expected_evidence=[f"到达共享业务阶段：{assessment.expected_stage}"],
            falsification_conditions=list(assessment.falsification_conditions),
            failure_pattern_refs=[assessment.primary_failure_pattern_id],
            evidence_refs=list(assessment.evidence_refs),
            reason=assessment.intervention_rationale or assessment.reason,
            item_id="experiment-plan-1",
        )
        memory.save_plan(plan)


def _read_artifact(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RecoveryWorkflowError(f"阶段产物不存在: {path}") from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RecoveryWorkflowError(f"阶段产物损坏: {path}") from error
    if not isinstance(payload, dict):
        raise RecoveryWorkflowError(f"阶段产物必须是 JSON 对象: {path}")
    return payload


def _data(payload: Any, model, label: str) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise RecoveryWorkflowError(f"{label} 必须是对象")
    expected = {item.name for item in fields(model)}
    if set(payload) != expected:
        raise RecoveryWorkflowError(
            f"{label} 字段不匹配，缺少 {sorted(expected - set(payload))}，"
            f"未知 {sorted(set(payload) - expected)}"
        )
    return dict(payload)


def _evidence(payload: Any) -> EvidenceRef:
    return EvidenceRef(**_data(payload, EvidenceRef, "证据引用"))


def _evidence_list(payload: Any) -> List[EvidenceRef]:
    if not isinstance(payload, list):
        raise RecoveryWorkflowError("证据引用必须是数组")
    return [_evidence(item) for item in payload]


def _experience(payload: Any) -> ExperienceDraft:
    data = _data(payload, ExperienceDraft, "经验")
    data["evidence_refs"] = _evidence_list(data["evidence_refs"])
    return ExperienceDraft(**data)


def _request(payload: Any) -> AttemptRequest:
    data = _data(payload, AttemptRequest, "AttemptRequest")
    if data["experience"] is not None:
        data["experience"] = _experience(data["experience"])
    return AttemptRequest(**data)


def _attempt_result(payload: Any) -> AttemptResult:
    data = _data(payload, AttemptResult, "AttemptResult")
    data["request"] = _request(data["request"])
    try:
        data["status"] = AttemptStatus(data["status"])
    except (TypeError, ValueError) as error:
        raise RecoveryWorkflowError("AttemptResult 包含未知状态") from error
    return AttemptResult(**data)


def _verdict(payload: Any) -> AttemptVerdict:
    data = _data(payload, AttemptVerdict, "AttemptVerdict")
    try:
        data["completion_status"] = AttemptStatus(data["completion_status"])
    except (TypeError, ValueError) as error:
        raise RecoveryWorkflowError("AttemptVerdict 包含未知状态") from error
    return AttemptVerdict(**data)


def _analysis(payload: Any) -> AttemptAnalysis:
    data = _data(payload, AttemptAnalysis, "AttemptAnalysis")
    if data["verdict"] is not None:
        data["verdict"] = _verdict(data["verdict"])
    return AttemptAnalysis(**data)


def _statistics(payload: Any) -> BaselineStatistics:
    if isinstance(payload, dict) and "finished_count" not in payload:
        payload = {**payload, "finished_count": 0}
    return BaselineStatistics(**_data(payload, BaselineStatistics, "基线统计"))


def _rubric(payload: Any) -> BusinessStageRubric:
    data = _data(payload, BusinessStageRubric, "业务阶段 Rubric")
    if not isinstance(data["stages"], list):
        raise RecoveryWorkflowError("业务阶段 stages 必须是数组")
    data["stages"] = [
        BusinessStage(**_data(item, BusinessStage, "业务阶段"))
        for item in data["stages"]
    ]
    return BusinessStageRubric(**data)


def _diagnosis_from_dict(payload: Any) -> DiagnosisReport:
    data = _data(payload, DiagnosisReport, "诊断报告")
    if not isinstance(data["attempt_results"], list):
        raise RecoveryWorkflowError("诊断 attempt_results 必须是数组")
    data["attempt_results"] = [_attempt_result(item) for item in data["attempt_results"]]
    if not isinstance(data["progress_by_attempt"], dict):
        raise RecoveryWorkflowError("诊断 progress_by_attempt 必须是对象")
    data["progress_by_attempt"] = {
        key: StageProgress(**_data(item, StageProgress, "业务进度"))
        for key, item in data["progress_by_attempt"].items()
    }
    if not isinstance(data["failure_patterns"], list):
        raise RecoveryWorkflowError("诊断 failure_patterns 必须是数组")
    patterns = []
    for item in data["failure_patterns"]:
        pattern = _data(item, FailurePattern, "失败模式")
        pattern["evidence_refs"] = _evidence_list(pattern["evidence_refs"])
        patterns.append(FailurePattern(**pattern))
    data["failure_patterns"] = patterns
    if not isinstance(data["attempt_analyses"], list):
        raise RecoveryWorkflowError("诊断 attempt_analyses 必须是数组")
    data["attempt_analyses"] = [_analysis(item) for item in data["attempt_analyses"]]
    if data["overall_statistics"] is not None:
        data["overall_statistics"] = _statistics(data["overall_statistics"])
    if not isinstance(data["statistics_by_instruction"], dict):
        raise RecoveryWorkflowError("逐指令统计必须是对象")
    data["statistics_by_instruction"] = {
        key: _statistics(item) for key, item in data["statistics_by_instruction"].items()
    }
    if data["stage_rubric"] is not None:
        data["stage_rubric"] = _rubric(data["stage_rubric"])
    if not isinstance(data["issues"], list):
        raise RecoveryWorkflowError("诊断 issues 必须是数组")
    data["issues"] = [
        BaselineAnalysisIssue(**_data(item, BaselineAnalysisIssue, "诊断问题"))
        for item in data["issues"]
    ]
    return DiagnosisReport(**data)


def _assessment_from_dict(payload: Any) -> RecoveryValueAssessment:
    data = _data(payload, RecoveryValueAssessment, "第二层修复价值判断")
    try:
        data["decision"] = RecoveryValueDecision(data["decision"])
        data["llm_decision"] = (
            None if data["llm_decision"] is None
            else RecoveryValueDecision(data["llm_decision"])
        )
    except (TypeError, ValueError) as error:
        raise RecoveryWorkflowError("第二层产物包含未知决策") from error
    data["evidence_refs"] = _evidence_list(data["evidence_refs"])
    if data["initial_experience"] is not None:
        data["initial_experience"] = _experience(data["initial_experience"])
    return RecoveryValueAssessment(**data)


def _initial_from_dict(payload: Any) -> InitialExperienceResult:
    data = _data(payload, InitialExperienceResult, "第一条经验结果")
    data["experience"] = _experience(data["experience"])
    return InitialExperienceResult(**data)
