from typing import Any, Dict, Protocol, runtime_checkable

from .schemas import AttemptRequest, AttemptResult


@runtime_checkable
class DeviceRunner(Protocol):
    async def run_batch(self, request: AttemptRequest) -> Dict[str, Any]:
        ...

    async def get_result(self, idempotency_key: str) -> AttemptResult:
        ...

    async def cancel(self, idempotency_key: str) -> Dict[str, Any]:
        ...

    async def health_check(self) -> Dict[str, Any]:
        ...
