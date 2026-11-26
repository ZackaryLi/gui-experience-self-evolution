from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .schemas import (
    DiagnosisReport,
    DictSerializable,
    EvidenceRef,
    ExperienceDraft,
    RecoveryState,
    RecoveryValueAssessment,
)


class AgentActionType(str, Enum):
    CALL_TOOL = "call_tool"
    FINISH = "finish"


class RecoveryDecision(str, Enum):
    PROPOSE_EXPERIENCE = "PROPOSE_EXPERIENCE"
    REVISE_EXPERIENCE = "REVISE_EXPERIENCE"
    RETRY_EXPERIENCE = "RETRY_EXPERIENCE"
    REQUEST_MORE_BASELINE = "REQUEST_MORE_BASELINE"
    ACCEPT_EXPERIENCE = "ACCEPT_EXPERIENCE"
    STOP_BLOCKED = "STOP_BLOCKED"
    STOP_UNSOLVABLE = "STOP_UNSOLVABLE"
    STOP_RETRYABLE = "STOP_RETRYABLE"
    ESCALATE = "ESCALATE"


class AgentRunStatus(str, Enum):
    FINISHED = "FINISHED"
    MAX_TOOL_ROUNDS_EXCEEDED = "MAX_TOOL_ROUNDS_EXCEEDED"
    INVALID_MODEL_OUTPUT = "INVALID_MODEL_OUTPUT"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"


class PermissionOutcome(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    MANUAL_REVIEW = "MANUAL_REVIEW"


@dataclass(frozen=True)
class RecoveryContext(DictSerializable):
    intent_id: str
    state: RecoveryState
    diagnosis: Optional[DiagnosisReport] = None
    recovery_value: Optional[RecoveryValueAssessment] = None
    current_experience: Optional[ExperienceDraft] = None
    evidence_refs: List[EvidenceRef] = field(default_factory=list)
    budget: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall(DictSerializable):
    tool: str
    input: Dict[str, Any]


@dataclass(frozen=True)
class PermissionDecision(DictSerializable):
    outcome: PermissionOutcome
    reason_code: str
    reason: str
    tool: str
    intent_id: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentAction(DictSerializable):
    action: AgentActionType
    reason: str
    evidence_refs: List[EvidenceRef]
    tool_call: Optional[ToolCall] = None
    decision: Optional[RecoveryDecision] = None
    experiment_id: Optional[str] = None
    conclusion: Optional[str] = None
    hypothesis: Optional[str] = None


@dataclass(frozen=True)
class ToolObservation(DictSerializable):
    tool: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass(frozen=True)
class AgentStep(DictSerializable):
    step_id: str
    raw_response: str
    action: Optional[AgentAction] = None
    observation: Optional[ToolObservation] = None
    validation_error: Optional[str] = None
    permission: Optional[PermissionDecision] = None
    hook_error: Optional[str] = None


@dataclass
class AgentRunResult(DictSerializable):
    status: AgentRunStatus
    reason: str
    decision: Optional[RecoveryDecision]
    steps: List[AgentStep]
    llm_calls: int
    tool_rounds: int
    invalid_responses: int
    run_id: str = ""
