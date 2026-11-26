import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "1.0"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class TriageStatus(str, Enum):
    EXPLORABLE = "EXPLORABLE"
    UNSOLVABLE = "UNSOLVABLE"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    REWRITING = "REWRITING"
    MANUAL_REVIEW = "MANUAL_REVIEW"


@dataclass(frozen=True)
class IntentKey:
    category: str
    app: str
    intent_name: str

    @property
    def intent_id(self) -> str:
        raw_key = canonical_json({
            "category": self.category,
            "app": self.app,
            "intent_name": self.intent_name,
        })
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, str]:
        return {
            "category": self.category,
            "app": self.app,
            "intent_name": self.intent_name,
        }


@dataclass
class LoadedTrajectory:
    record: Dict[str, Any]
    content_hash: str
    line_number: int

    @property
    def sessionid(self) -> str:
        return str(self.record.get("sessionid", ""))

    @property
    def primary_parent(self) -> str:
        verdict = self.record.get("verdict", {})
        value = verdict.get("primary_failure_parent_category", "") if isinstance(verdict, dict) else ""
        value = str(value).strip().upper()
        return value if value in {"A", "B", "C", "D", "E"} else "UNKNOWN"

    @property
    def primary_subcategory(self) -> str:
        verdict = self.record.get("verdict", {})
        value = verdict.get("primary_failure_subcategory", "") if isinstance(verdict, dict) else ""
        return str(value).strip().upper()


@dataclass
class ValidationIssue:
    code: str
    message: str
    sessionid: str = ""
    line_number: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "sessionid": self.sessionid,
            "line_number": self.line_number,
        }


@dataclass
class InconsistentRecord:
    record: Any
    issues: List[ValidationIssue]
    intent_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "record": self.record,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass
class IntentWorkItem:
    key: IntentKey
    records: List[LoadedTrajectory] = field(default_factory=list)
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def intent_id(self) -> str:
        return self.key.intent_id


@dataclass
class LoadOutcome:
    items: List[IntentWorkItem]
    inconsistent_records: List[InconsistentRecord]
    total_input_records: int = 0
    skipped_cluster_mismatch: int = 0
    duplicate_records: int = 0


@dataclass
class TriageResult:
    key: IntentKey
    status: TriageStatus
    rule_id: str
    reason: str
    total_records: int
    root_distribution: Dict[str, Any]
    thresholds: Dict[str, Any]
    issues: List[ValidationIssue] = field(default_factory=list)
    candidate_records: List[LoadedTrajectory] = field(default_factory=list)

    @property
    def intent_id(self) -> str:
        return self.key.intent_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "intent_id": self.intent_id,
            "intent_key": self.key.to_dict(),
            "status": self.status.value,
            "rule_id": self.rule_id,
            "reason": self.reason,
            "total_records": self.total_records,
            "root_distribution": self.root_distribution,
            "thresholds": self.thresholds,
            "issues": [issue.to_dict() for issue in self.issues],
            "exploration_candidate_count": len(self.candidate_records),
            "exploration_candidate_session_ids": [
                record.sessionid for record in self.candidate_records
            ],
        }


class DictSerializable:
    def to_dict(self) -> Dict[str, Any]:
        return _serialize_dataclass(self)


def _serialize_dataclass(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _serialize_dataclass(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    if isinstance(value, dict):
        return {
            str(key): _serialize_dataclass(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_serialize_dataclass(item) for item in value]
    return value


class BatchRunDecision(str, Enum):
    RUN_BASELINE = "RUN_BASELINE"
    SKIP_NO_BATCH_VALUE = "SKIP_NO_BATCH_VALUE"
    STOP_BLOCKED = "STOP_BLOCKED"
    STOP_UNSOLVABLE = "STOP_UNSOLVABLE"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class RecoveryValueDecision(str, Enum):
    START_RECOVERY = "START_RECOVERY"
    NO_REPAIR_NEEDED = "NO_REPAIR_NEEDED"
    STOP_BLOCKED = "STOP_BLOCKED"
    STOP_UNSOLVABLE = "STOP_UNSOLVABLE"
    STOP_RETRYABLE = "STOP_RETRYABLE"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class AttemptStatus(str, Enum):
    FINISHED = "finished"
    COMPLETED = "completed"
    FAILED = "failed"
    UNCERTAIN = "uncertain"
    DEVICE_ERROR = "device_error"
    RUNNER_ERROR = "runner_error"
    TIMEOUT_UNKNOWN = "timeout_unknown"
    NOT_DISPATCHED = "not_dispatched"
    CANCELLED = "cancelled"


class RecoveryState(str, Enum):
    TRIAGED = "TRIAGED"
    BATCH_VALUE_CHECKED = "BATCH_VALUE_CHECKED"
    BASELINE_PENDING = "BASELINE_PENDING"
    BASELINE_ANALYZED = "BASELINE_ANALYZED"
    RECOVERY_VALUE_CHECKED = "RECOVERY_VALUE_CHECKED"
    EXPERIENCE_EXPLORING = "EXPERIENCE_EXPLORING"
    VALIDATING = "VALIDATING"
    SUCCEEDED = "SUCCEEDED"
    STOP_BLOCKED = "STOP_BLOCKED"
    STOP_UNSOLVABLE = "STOP_UNSOLVABLE"
    STOP_RETRYABLE = "STOP_RETRYABLE"
    MANUAL_REVIEW = "MANUAL_REVIEW"


@dataclass(frozen=True)
class EvidenceRef(DictSerializable):
    sessionid: str
    step_ids: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class BatchRunConstraints(DictSerializable):
    execution_authorized: bool = True
    safety_cleared: bool = True
    environment_ready: bool = True
    has_reusable_baseline: bool = False
    max_attempts: Optional[int] = None
    notes: str = ""


@dataclass
class BatchRunValueAssessment(DictSerializable):
    intent_id: str
    intent_key: IntentKey
    decision: BatchRunDecision
    reason: str
    information_value: str
    repair_potential: str
    confidence: float
    evidence_refs: List[EvidenceRef]
    candidate_count: int
    unique_instruction_count: int
    planned_attempt_count: int
    source_fingerprint: str
    blocker_type: str = ""
    llm_decision: Optional[BatchRunDecision] = None
    assessment_mode: str = "LLM_WITH_RULE_GUARDS"
    rule_findings: List[str] = field(default_factory=list)
    prompt_version: str = "batch_value_v1"
    created_at: str = ""


@dataclass(frozen=True)
class BaselineSample(DictSerializable):
    instruction_id: str
    instruction: str
    source_session_ids: List[str]
    source_content_hashes: Dict[str, str]
    attempt_indices: List[int]


@dataclass(frozen=True)
class BaselinePlan(DictSerializable):
    plan_id: str
    intent_id: str
    intent_key: IntentKey
    status: str
    instruction_sample_size: int
    attempts_per_instruction: int
    random_seed: int
    source_candidate_count: int
    source_fingerprint: str
    samples: List[BaselineSample]
    created_at: str

    @property
    def total_attempts(self) -> int:
        return sum(len(sample.attempt_indices) for sample in self.samples)

    def to_attempt_requests(self) -> List["AttemptRequest"]:
        requests = []
        for sample in self.samples:
            for attempt_index in sample.attempt_indices:
                idempotency_key = content_hash({
                    "plan_id": self.plan_id,
                    "instruction_id": sample.instruction_id,
                    "attempt_index": attempt_index,
                })
                requests.append(AttemptRequest(
                    intent_id=self.intent_id,
                    plan_id=self.plan_id,
                    instruction_id=sample.instruction_id,
                    instruction=sample.instruction,
                    attempt_index=attempt_index,
                    idempotency_key=idempotency_key,
                ))
        return requests


@dataclass(frozen=True)
class ExperienceDraft(DictSerializable):
    experience_id: str
    intent_id: str
    version: int
    hypothesis: str
    applicable_scope: str
    trigger_conditions: List[str]
    strategy: str
    expected_stage: str
    evidence_refs: List[EvidenceRef]
    parent_experience_id: Optional[str] = None
    forbidden_behaviors: List[str] = field(default_factory=list)
    exit_conditions: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class AttemptRequest(DictSerializable):
    intent_id: str
    plan_id: str
    instruction_id: str
    instruction: str
    attempt_index: int
    idempotency_key: str
    experience: Optional[ExperienceDraft] = None
    environment: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttemptResult(DictSerializable):
    request: AttemptRequest
    status: AttemptStatus
    task_completed: Optional[bool]
    trajectory: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


@dataclass(frozen=True)
class StageProgress(DictSerializable):
    stage_name: str
    stage_order: int
    evidence_steps: List[str]
    furthest_anchor: Optional[str] = None
    first_deviation_step: Optional[str] = None
    repeated_action_steps: List[str] = field(default_factory=list)
    page_oscillation_steps: List[str] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.stage_order, bool) or not isinstance(self.stage_order, int):
            raise ValueError("stage_order 必须是整数")
        if self.stage_order < 0:
            raise ValueError("stage_order 不能小于 0")


@dataclass(frozen=True)
class BusinessStage(DictSerializable):
    stage_name: str
    stage_order: int
    description: str
    completion_evidence: List[str]


@dataclass(frozen=True)
class BusinessStageRubric(DictSerializable):
    intent_id: str
    stages: List[BusinessStage]
    rubric_id: str
    prompt_version: str = "business_stage_rubric_v1"


@dataclass(frozen=True)
class AttemptVerdict(DictSerializable):
    attempt_key: str
    sessionid: str
    completion_status: AttemptStatus
    task_completed: Optional[bool]
    completion_analysis: str
    failure_analysis: str
    primary_failure_parent_category: str
    primary_failure_subcategory: str
    key_evidence_steps: List[str]
    confidence: float
    blocker_type: str = ""


@dataclass
class AttemptAnalysis(DictSerializable):
    attempt_key: str
    instruction_id: str
    attempt_index: int
    sessionid: str
    execution_status: AttemptStatus
    execution_class: str
    included_in_strategy: bool = False
    conflict: bool = False
    verdict: Optional[AttemptVerdict] = None
    issues: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class BaselineAnalysisIssue(DictSerializable):
    code: str
    message: str
    attempt_key: str = ""
    instruction_id: str = ""


@dataclass
class BaselineStatistics(DictSerializable):
    planned_count: int = 0
    received_count: int = 0
    finished_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    uncertain_count: int = 0
    device_error_count: int = 0
    runner_error_count: int = 0
    timeout_unknown_count: int = 0
    not_dispatched_count: int = 0
    cancelled_count: int = 0
    valid_completed_count: int = 0
    valid_failed_count: int = 0
    conflict_count: int = 0
    valid_attempt_keys: List[str] = field(default_factory=list)
    excluded_attempt_keys: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class FailurePattern(DictSerializable):
    pattern_id: str
    name: str
    description: str
    evidence_refs: List[EvidenceRef]
    blocker_type: str = ""


@dataclass
class DiagnosisReport(DictSerializable):
    intent_id: str
    baseline_plan_id: str
    attempt_results: List[AttemptResult]
    progress_by_attempt: Dict[str, StageProgress]
    failure_patterns: List[FailurePattern]
    summary: str
    evidence_quality: str = "UNKNOWN"
    attempt_analyses: List[AttemptAnalysis] = field(default_factory=list)
    overall_statistics: Optional[BaselineStatistics] = None
    statistics_by_instruction: Dict[str, BaselineStatistics] = field(default_factory=dict)
    stage_rubric: Optional[BusinessStageRubric] = None
    issues: List[BaselineAnalysisIssue] = field(default_factory=list)
    analysis_version: str = "baseline_analysis_v1"
    created_at: str = ""
    summary_skill_name: Optional[str] = None
    summary_skill_version: Optional[str] = None
    summary_skill_content_hash: Optional[str] = None


@dataclass
class RecoveryValueAssessment(DictSerializable):
    intent_id: str
    baseline_plan_id: str
    decision: RecoveryValueDecision
    reason: str
    evidence_refs: List[EvidenceRef]
    initial_experience: Optional[ExperienceDraft] = None
    confidence: float = 0.0
    repair_potential: str = "UNKNOWN"
    intervention_rationale: str = ""
    primary_failure_pattern_id: str = ""
    primary_hypothesis: str = ""
    expected_stage: str = ""
    falsification_conditions: List[str] = field(default_factory=list)
    blocker_type: str = ""
    llm_decision: Optional[RecoveryValueDecision] = None
    assessment_mode: str = "DETERMINISTIC_PRECHECK"
    rule_findings: List[str] = field(default_factory=list)
    prompt_version: str = "recovery_value_v1"
    created_at: str = ""


@dataclass(frozen=True)
class ExperimentCard(DictSerializable):
    experiment_id: str
    intent_id: str
    hypothesis: str
    baseline_plan_id: str
    experience: ExperienceDraft
    instruction_ids: List[str]
    attempts_per_instruction: int
    expected_stage: str
    falsification_criteria: List[str]
    control_description: str = "无经验基线"
    environment: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryStateSnapshot(DictSerializable):
    intent_id: str
    state: RecoveryState
    current_experience_id: Optional[str] = None
    current_hypothesis: str = ""
    round_index: int = 0
    budget: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    schema_version: str = "1.0"
    revision: int = 0
    baseline_plan_id: Optional[str] = None
    blocker_reason: str = ""
    allowed_actions: List[str] = field(default_factory=list)
    artifact_hashes: Dict[str, str] = field(default_factory=dict)
    last_transition_id: str = ""
    created_at: str = ""
    updated_at: str = ""
