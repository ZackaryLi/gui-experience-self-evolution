import asyncio
import inspect
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Dict, Iterable, Optional, Protocol

from .recovery_schemas import PermissionDecision, ToolObservation
from .schemas import DictSerializable
from .storage import append_jsonl
from .tools import ToolOperation


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class PreToolUseEvent(DictSerializable):
    run_id: str
    step_id: str
    sequence: int
    tool_call_id: str
    tool: str
    intent_id: str
    operation: ToolOperation
    permission: PermissionDecision
    occurred_at: str = field(default_factory=_utc_now)
    event_type: str = "tool.started"


@dataclass(frozen=True)
class PostToolUseEvent(DictSerializable):
    run_id: str
    step_id: str
    sequence: int
    tool_call_id: str
    tool: str
    intent_id: str
    operation: ToolOperation
    permission: PermissionDecision
    observation: ToolObservation
    duration_ms: float
    handler_called: bool
    occurred_at: str = field(default_factory=_utc_now)
    event_type: str = "tool.finished"


class ToolHook(Protocol):
    def before_tool_use(
        self,
        event: PreToolUseEvent,
    ) -> Optional[Awaitable[None]]:
        ...

    def after_tool_use(
        self,
        event: PostToolUseEvent,
    ) -> Optional[Awaitable[None]]:
        ...


class ToolHookError(RuntimeError):
    def __init__(self, phase: str, hook: str, error_type: str):
        self.phase = phase
        self.hook = hook
        self.error_type = error_type
        self.reason_code = (
            "PRE_TOOL_HOOK_FAILED"
            if phase == "PRE_TOOL_USE"
            else "POST_TOOL_HOOK_FAILED"
        )
        super().__init__(
            f"{self.reason_code}: hook={hook}, error_type={error_type}"
        )


class ToolHookRunner:
    def __init__(
        self,
        hooks: Optional[Iterable[ToolHook]] = None,
        timeout_seconds: float = 10.0,
    ):
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ValueError("Hook 超时时间必须是正数")
        self.hooks = list(hooks or [])
        self.timeout_seconds = float(timeout_seconds)

    async def run_before(self, event: PreToolUseEvent) -> None:
        for hook in self.hooks:
            await self._run_one(hook, "before_tool_use", "PRE_TOOL_USE", event)

    async def run_after(self, event: PostToolUseEvent) -> None:
        for hook in self.hooks:
            await self._run_one(hook, "after_tool_use", "POST_TOOL_USE", event)

    async def _run_one(
        self,
        hook: ToolHook,
        method_name: str,
        phase: str,
        event: Any,
    ) -> None:
        method = getattr(hook, method_name, None)
        if method is None:
            return
        try:
            await asyncio.wait_for(
                self._invoke(method, event),
                timeout=self.timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise ToolHookError(
                phase=phase,
                hook=type(hook).__name__,
                error_type=type(error).__name__,
            ) from error

    @staticmethod
    async def _invoke(method, event) -> None:
        result = method(event)
        if inspect.isawaitable(result):
            await result


class JsonlAuditHook:
    def __init__(self, path: str | Path):
        if not str(path).strip():
            raise ValueError("审计事件路径不能为空")
        self.path = Path(path)

    def before_tool_use(self, event: PreToolUseEvent) -> None:
        append_jsonl(self.path, self._record(event))

    def after_tool_use(self, event: PostToolUseEvent) -> None:
        append_jsonl(self.path, self._record(event))

    @staticmethod
    def _record(event) -> Dict[str, Any]:
        record = {
            "event_type": event.event_type,
            "run_id": event.run_id,
            "step_id": event.step_id,
            "sequence": event.sequence,
            "tool_call_id": event.tool_call_id,
            "tool": event.tool,
            "intent_id": event.intent_id,
            "operation": event.operation.value,
            "permission_outcome": event.permission.outcome.value,
            "permission_reason_code": event.permission.reason_code,
            "occurred_at": event.occurred_at,
        }
        if isinstance(event, PostToolUseEvent):
            record.update({
                "handler_called": event.handler_called,
                "success": event.observation.success,
                "duration_ms": event.duration_ms,
                "error_code": _observation_error_code(event.observation),
                "result_keys": sorted(event.observation.data),
            })
        return record


def _observation_error_code(observation: ToolObservation) -> Optional[str]:
    if observation.success:
        return None
    if not observation.error:
        return "UNKNOWN_TOOL_ERROR"
    return observation.error.split(":", 1)[0]
