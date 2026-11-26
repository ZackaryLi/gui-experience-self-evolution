from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .schemas import DictSerializable


RUNNER_SCHEMA_VERSION = "1.0"


class ExperienceMode(str, Enum):
    BASELINE = "baseline"
    ACCEPTED = "accepted"
    CANDIDATE = "candidate"
    VALIDATION = "validation"
    EXPANSION = "expansion"


class ExecutionStatus(str, Enum):
    RUNNING = "running"
    FINISHED = "finished"
    TIMEOUT = "timeout"
    DEVICE_ERROR = "device_error"
    RUNNER_ERROR = "runner_error"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class InfrastructureStatus(str, Enum):
    OK = "ok"
    TIMEOUT = "timeout"
    DEVICE_ERROR = "device_error"
    RUNNER_ERROR = "runner_error"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RunRequest(DictSerializable):
    schema_version: str
    idempotency_key: str
    run_id: str
    intent_id: str
    experiment_id: str
    instruction: str
    app: str
    attempt_index: int
    experience_mode: ExperienceMode
    plan_text: str = ""
    initial_state: str = "clean_app_start"
    timeout_sec: int = 600
    accepted_patch_ids: List[str] = field(default_factory=list)
    candidate_patch_id: Optional[str] = None
    device_constraints: Dict[str, Any] = field(default_factory=dict)
    safety_context: Dict[str, Any] = field(default_factory=dict)
    trajectory_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult(DictSerializable):
    schema_version: str
    idempotency_key: str
    execution_status: ExecutionStatus
    infrastructure_status: InfrastructureStatus
    instruction: str
    app: str
    attempt_index: int
    injected_plan_text: str
    trajectory: Optional[Dict[str, Any]] = None
    raw_ref: Optional[str] = None
    device_meta: Dict[str, Any] = field(default_factory=dict)
    run_context: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    started_at: str = ""
    finished_at: str = ""
    duration_sec: float = 0.0
    runner_version: str = ""
    artifacts: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CancelResult(DictSerializable):
    idempotency_key: str
    cancelled: bool
    reason: str = ""


@dataclass(frozen=True)
class RunnerHealth(DictSerializable):
    ok: bool
    devices: List[str] = field(default_factory=list)
    agent_endpoint_ok: Optional[bool] = None
    message: str = ""


@runtime_checkable
class DeviceRunner(Protocol):
    async def run_attempt(self, request: RunRequest) -> RunResult:
        ...

    async def get_result(self, idempotency_key: str) -> Optional[RunResult]:
        ...

    async def cancel(self, idempotency_key: str) -> CancelResult:
        ...

    async def health_check(self) -> RunnerHealth:
        ...

    async def close(self) -> None:
        ...
