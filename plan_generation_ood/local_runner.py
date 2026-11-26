"""L2 Runner 适配层：LocalDeviceRunner 真机执行实现（Phase 2+4）。

按 ``OOD_RUNNER_INTERFACE.md`` / ``OOD_ARCHITECTURE_LAYERS.md`` 约定实现：

- 一个 request = 一条指令的一次 attempt；
- 幂等：同 ``idempotency_key`` 不重复下发真机，进程重启后已落盘结果仍可命中；
- 超时：自身计时 + ``task.stop()`` + 有限宽限期，宽限期后设备标记不可用；
- 取消：``task.stop()`` 后等待线程退出，结果标记 ``cancelled``；
- 产物落盘：``result_root/<key>/{request,result,trajectory}.json`` 与 ``raw/``；
- 轨迹适配：``step_data.json`` → ``step1..stepN`` 统一轨迹（非连续 step 重编号）；
- 不返回业务 verdict：``finished`` 只表示执行流程正常结束。

L3/L4 的 ``scripts.*``（``CeliaMobileTask``、``device_api`` 等）位于另一个仓库，
通过 :class:`DeviceTaskApi` 工厂注入隔离：默认实现 :class:`ScriptDeviceTaskApi`
在首次使用时才延迟 import，本仓库可在无 ``scripts.*`` 环境下单测。
"""

import asyncio
import importlib
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .runner import (
    RUNNER_SCHEMA_VERSION,
    CancelResult,
    ExecutionStatus,
    ExperienceMode,
    InfrastructureStatus,
    RunRequest,
    RunResult,
    RunnerHealth,
)
from .runner_adapter import TrajectoryAdapter
from .storage import atomic_write_json


LOCAL_RUNNER_VERSION = f"local-runner-{RUNNER_SCHEMA_VERSION}"

_IN_PROGRESS_FILE = "in_progress.json"
_REQUEST_FILE = "request.json"
_RESULT_FILE = "result.json"
_TRAJECTORY_FILE = "trajectory.json"
_RAW_DIR = "raw"

_POLL_INTERVAL_SEC = 0.02


class LocalRunnerConfigError(ValueError):
    """Runner 配置或请求不合法。"""


class RunnerClosedError(RuntimeError):
    """Runner 已关闭。"""


@runtime_checkable
class DeviceTaskApi(Protocol):
    """L3/L4 执行接口的抽象，供注入与单测替换。"""

    def list_devices(self) -> List[str]:
        """返回当前可用设备 ID 列表。"""

    def create_task(self, instruction: str, device_id: str, plan_text: str) -> Any:
        """创建一个任务对象，需提供阻塞的 ``execute()`` 与 ``stop()``。"""

    def classify_error(self, error: BaseException) -> str:
        """把底层异常分类为 ``device_error`` 或 ``runner_error``。"""

    def agent_endpoint_configured(self) -> bool:
        """Agent 接口配置是否就绪。"""

    def ensure_config(self) -> None:
        """加载进程级配置；实现方只应在首次调用时执行一次。"""


class ScriptDeviceTaskApi:
    """默认工厂：延迟 import 另一仓库的 ``scripts.*``。

    本仓库没有 ``scripts`` 包，import 失败会抛出带说明的 :class:`ImportError`；
    构造实例本身不触发 import。
    """

    _MODULE_TASK = "scripts.appagent_test.task.mobile_task"
    _MODULE_DEVICE = "scripts.appagent_test.api.device_api"
    _MODULE_CONFIG = "scripts.config.global_config"

    def __init__(self, root_dir: Optional[str] = None):
        self._root_dir = root_dir
        self._modules: Dict[str, Any] = {}
        self._config_loaded = False
        self._device_error_types: tuple = ()

    def _import_module(self, name: str) -> Any:
        if name not in self._modules:
            try:
                self._modules[name] = importlib.import_module(name)
            except ImportError as exc:
                raise ImportError(
                    f"无法导入设备执行层模块 {name}；请在包含 scripts/ 的部署环境中运行，"
                    f"或通过 DeviceTaskApi 注入自定义工厂: {exc}"
                ) from exc
        return self._modules[name]

    def ensure_config(self) -> None:
        if self._config_loaded:
            return
        # load_config 写进程级 os.environ，进程内只允许调用一次。
        config_module = self._import_module(self._MODULE_CONFIG)
        root_dir = self._root_dir or os.path.join(os.path.dirname(__file__), "..", "..")
        config_module.load_config(root_dir)
        self._config_loaded = True

    def list_devices(self) -> List[str]:
        device_api = self._import_module(self._MODULE_DEVICE)
        return list(device_api.get_connected_devices())

    def create_task(self, instruction: str, device_id: str, plan_text: str) -> Any:
        task_module = self._import_module(self._MODULE_TASK)
        return task_module.CeliaMobileTask(instruction, device_id, plan_text)

    def classify_error(self, error: BaseException) -> str:
        if not self._device_error_types:
            device_api = self._import_module(self._MODULE_DEVICE)
            self._device_error_types = (device_api.HDCEError,)
        if isinstance(error, self._device_error_types):
            return "device_error"
        return "runner_error"

    def agent_endpoint_configured(self) -> bool:
        return bool(os.environ.get("AGENT_MODEL_URL"))


class StepDataTrajectoryAdapter:
    """``step_data.json`` → 分流/IND 契约轨迹（``step1..stepN``）的适配器。"""

    @staticmethod
    def step_data_to_trajectory(
        step_data: Dict[str, Any],
        request: RunRequest,
    ) -> Dict[str, Any]:
        if not isinstance(step_data, dict):
            raise LocalRunnerConfigError("step_data 必须是字典")
        context = dict(request.trajectory_context or {})
        category = context.get("category")
        if not isinstance(category, dict):
            category = {"category": str(category or "")}

        # stepN 可能不从 1 连续，重编号为 1..N，字段缺失时补空串。
        steps: List[tuple] = []
        for key, value in step_data.items():
            if not key.startswith("step") or not key[4:].isdigit():
                continue
            steps.append((int(key[4:]), value))
        steps.sort(key=lambda item: item[0])

        trajectory: Dict[str, Any] = {
            "sessionid": step_data.get("sessionId", ""),
            "instruction": step_data.get("instruction", "") or request.instruction,
            # step_data 的 app 只有 open 动作才会写入，为空时回填请求的 app。
            "app": step_data.get("app", "") or request.app,
            "plan": step_data.get("plan", "") or request.plan_text,
            "category": category,
            "intent_cluster_name": context.get("intent_cluster_name", ""),
        }

        for index, (_, raw_step) in enumerate(steps, start=1):
            if isinstance(raw_step, dict):
                action = raw_step.get("action") or json.dumps(
                    raw_step.get("actionSeq", []), ensure_ascii=False
                )
                step = {
                    "manager_action": raw_step.get("manager_action", ""),
                    "thought": raw_step.get("thought", ""),
                    "action": action,
                    "role": raw_step.get("role", ""),
                }
                for source_name in ("uiSummary", "ui_summary"):
                    summary = raw_step.get(source_name)
                    if summary not in (None, ""):
                        step["uiSummary"] = summary
                        break
            else:
                step = {
                    "manager_action": "",
                    "thought": "",
                    "action": "",
                    "role": "",
                }
            trajectory[f"step{index}"] = step

        # 统一补充 observation 结构，与 RunnerAdapter 的校验保持一致。
        return TrajectoryAdapter.normalize(trajectory)


class RunnerStore:
    """幂等存储：``result_root/<key>/`` 下的原子 JSON 与进行中标记。"""

    def __init__(self, result_root: str | Path):
        self._root = Path(result_root)

    def key_dir(self, idempotency_key: str) -> Path:
        return self._root / idempotency_key

    def save_request(self, request: RunRequest) -> Path:
        path = self.key_dir(request.idempotency_key) / _REQUEST_FILE
        atomic_write_json(path, request.to_dict())
        return path

    def load_request(self, idempotency_key: str) -> Optional[RunRequest]:
        path = self.key_dir(idempotency_key) / _REQUEST_FILE
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            return request_from_dict(data)
        except (OSError, ValueError, KeyError):
            return None

    def mark_in_progress(self, idempotency_key: str, request: RunRequest) -> None:
        atomic_write_json(
            self.key_dir(idempotency_key) / _IN_PROGRESS_FILE,
            {
                "idempotency_key": idempotency_key,
                "marked_at": datetime.now().isoformat(),
                "run_id": request.run_id,
            },
        )

    def clear_in_progress(self, idempotency_key: str) -> None:
        marker = self.key_dir(idempotency_key) / _IN_PROGRESS_FILE
        if marker.exists():
            marker.unlink()

    def has_in_progress(self, idempotency_key: str) -> bool:
        return (self.key_dir(idempotency_key) / _IN_PROGRESS_FILE).exists()

    def save_trajectory(self, idempotency_key: str, trajectory: Dict[str, Any]) -> Path:
        path = self.key_dir(idempotency_key) / _TRAJECTORY_FILE
        atomic_write_json(path, trajectory)
        return path

    def save_result(self, result: RunResult) -> Path:
        path = self.key_dir(result.idempotency_key) / _RESULT_FILE
        atomic_write_json(path, result.to_dict())
        return path

    def load_result(self, idempotency_key: str) -> Optional[RunResult]:
        path = self.key_dir(idempotency_key) / _RESULT_FILE
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            return result_from_dict(data)
        except (OSError, ValueError):
            # 结果文件损坏按未见过处理，调用方决定是否人工排查。
            return None

    def raw_dir(self, idempotency_key: str) -> Path:
        return self.key_dir(idempotency_key) / _RAW_DIR


def result_from_dict(data: Dict[str, Any]) -> RunResult:
    """从持久化 JSON 恢复 :class:`RunResult`（枚举按值还原）。"""
    if data.get("schema_version") != RUNNER_SCHEMA_VERSION:
        raise ValueError(f"不支持的结果 Schema 版本: {data.get('schema_version')}")
    return RunResult(
        schema_version=data["schema_version"],
        idempotency_key=data["idempotency_key"],
        execution_status=ExecutionStatus(data["execution_status"]),
        infrastructure_status=InfrastructureStatus(data["infrastructure_status"]),
        instruction=data["instruction"],
        app=data["app"],
        attempt_index=data["attempt_index"],
        injected_plan_text=data["injected_plan_text"],
        trajectory=data.get("trajectory"),
        raw_ref=data.get("raw_ref"),
        device_meta=dict(data.get("device_meta") or {}),
        run_context=dict(data.get("run_context") or {}),
        error=data.get("error"),
        started_at=data.get("started_at", ""),
        finished_at=data.get("finished_at", ""),
        duration_sec=float(data.get("duration_sec", 0.0) or 0.0),
        runner_version=data.get("runner_version", ""),
        artifacts=dict(data.get("artifacts") or {}),
    )


def request_from_dict(data: Dict[str, Any]) -> RunRequest:
    """从持久化 JSON 恢复 :class:`RunRequest`（枚举按值还原）。"""
    return RunRequest(
        schema_version=data["schema_version"],
        idempotency_key=data["idempotency_key"],
        run_id=data["run_id"],
        intent_id=data["intent_id"],
        experiment_id=data["experiment_id"],
        instruction=data["instruction"],
        app=data["app"],
        attempt_index=data["attempt_index"],
        experience_mode=ExperienceMode(data["experience_mode"]),
        plan_text=data.get("plan_text", ""),
        initial_state=data.get("initial_state", "clean_app_start"),
        timeout_sec=data.get("timeout_sec", 600),
        accepted_patch_ids=list(data.get("accepted_patch_ids") or []),
        candidate_patch_id=data.get("candidate_patch_id"),
        device_constraints=dict(data.get("device_constraints") or {}),
        safety_context=dict(data.get("safety_context") or {}),
        trajectory_context=dict(data.get("trajectory_context") or {}),
        metadata=dict(data.get("metadata") or {}),
    )


@dataclass
class _ActiveAttempt:
    """一次进行中 attempt 的内部句柄。"""

    idempotency_key: str
    device_id: Optional[str]
    task: Any = None
    cancel_requested: bool = False
    exit_event: threading.Event = field(default_factory=threading.Event)
    outcome: Dict[str, Any] = field(default_factory=dict)

    def finish_with(self, status: str, **payload: Any) -> None:
        self.outcome.clear()
        self.outcome["status"] = status
        self.outcome.update(payload)
        self.exit_event.set()


class LocalDeviceRunner:
    """真机 Runner 实现，对接 task 级 ``CeliaMobileTask``（不经过 AppExploration）。"""

    def __init__(
        self,
        result_root: str | Path,
        task_api: Optional[DeviceTaskApi] = None,
        max_concurrency: int = 1,
        default_timeout_sec: int = 600,
        device_cache_ttl_sec: float = 5.0,
        stop_grace_sec: float = 30.0,
    ):
        if max_concurrency < 1:
            raise LocalRunnerConfigError("max_concurrency 必须不小于 1")
        if default_timeout_sec <= 0:
            raise LocalRunnerConfigError("default_timeout_sec 必须是正数")
        if stop_grace_sec <= 0:
            raise LocalRunnerConfigError("stop_grace_sec 必须是正数")
        self._store = RunnerStore(result_root)
        self._task_api: DeviceTaskApi = task_api or ScriptDeviceTaskApi()
        self._max_concurrency = max_concurrency
        self._default_timeout_sec = default_timeout_sec
        self._device_cache_ttl_sec = device_cache_ttl_sec
        self._stop_grace_sec = stop_grace_sec
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._futures: Dict[str, asyncio.Future] = {}
        self._active: Dict[str, _ActiveAttempt] = {}
        self._busy_devices: set = set()
        self._unavailable_devices: set = set()
        self._device_cache: Optional[tuple] = None  # (monotonic 秒, devices)
        self._lock = asyncio.Lock()
        self._closed = False
        self._config_ensured = False

    # ------------------------------------------------------------------
    # DeviceRunner 协议实现
    # ------------------------------------------------------------------

    async def run_attempt(self, request: RunRequest) -> RunResult:
        self._ensure_open()
        self._validate_request(request)
        async with self._lock:
            existing = self._futures.get(request.idempotency_key)
            if existing is not None:
                return await asyncio.shield(existing)
        persisted = self._store.load_result(request.idempotency_key)
        if persisted is not None:
            return persisted
        if self._store.has_in_progress(request.idempotency_key):
            # 进程重启后遗留的进行中标记：线程已死，不重复跑真机。
            return self._finalize_stale_attempt(request=request)

        future: asyncio.Future = asyncio.get_running_loop().create_future()
        async with self._lock:
            existing = self._futures.get(request.idempotency_key)
            if existing is not None:
                return await asyncio.shield(existing)
            self._futures[request.idempotency_key] = future
        try:
            result = await self._execute(request)
            if not future.done():
                future.set_result(result)
            return result
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
            raise
        finally:
            async with self._lock:
                self._futures.pop(request.idempotency_key, None)

    async def get_result(self, idempotency_key: str) -> Optional[RunResult]:
        self._ensure_open()
        async with self._lock:
            if idempotency_key in self._futures:
                return self._running_result(idempotency_key)
        persisted = self._store.load_result(idempotency_key)
        if persisted is not None:
            return persisted
        if self._store.has_in_progress(idempotency_key):
            return self._finalize_stale_attempt(idempotency_key=idempotency_key)
        return None

    async def cancel(self, idempotency_key: str) -> CancelResult:
        self._ensure_open()
        async with self._lock:
            attempt = self._active.get(idempotency_key)
            if attempt is None:
                persisted = self._store.load_result(idempotency_key)
                reason = "already_finished" if persisted is not None else "unknown_key"
                return CancelResult(
                    idempotency_key=idempotency_key,
                    cancelled=False,
                    reason=reason,
                )
            attempt.cancel_requested = True
            if attempt.task is not None:
                try:
                    attempt.task.stop()
                except Exception:
                    pass
        # 尽力等待线程退出；未退出也视为已请求取消，最终状态由主流程落盘。
        exited = await asyncio.to_thread(attempt.exit_event.wait, self._stop_grace_sec)
        return CancelResult(
            idempotency_key=idempotency_key,
            cancelled=True,
            reason="" if exited else "still_running",
        )

    async def health_check(self) -> RunnerHealth:
        try:
            devices = await asyncio.to_thread(self._refresh_devices, True)
        except Exception as exc:
            return RunnerHealth(
                ok=False,
                devices=[],
                agent_endpoint_ok=None,
                message=f"设备枚举失败: {exc}",
            )
        try:
            endpoint_ok = await asyncio.to_thread(
                self._task_api.agent_endpoint_configured
            )
        except Exception:
            endpoint_ok = None
        message = ""
        if not devices:
            message = "没有可用设备"
        elif self._unavailable_devices:
            message = (
                f"超时未退出的设备已被隔离: {sorted(self._unavailable_devices)}"
            )
        return RunnerHealth(
            ok=bool(devices) and not self._unavailable_devices,
            devices=sorted(devices),
            agent_endpoint_ok=endpoint_ok,
            message=message,
        )

    async def close(self) -> None:
        self._closed = True
        async with self._lock:
            attempts = list(self._active.values())
        for attempt in attempts:
            attempt.cancel_requested = True
            if attempt.task is not None:
                try:
                    await asyncio.to_thread(attempt.task.stop)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # 请求校验与遗留状态
    # ------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise RunnerClosedError("Runner 已关闭")

    def _validate_request(self, request: RunRequest) -> None:
        if request.schema_version != RUNNER_SCHEMA_VERSION:
            raise LocalRunnerConfigError(
                f"不支持的请求 Schema 版本: {request.schema_version}"
            )
        if not request.idempotency_key.strip():
            raise LocalRunnerConfigError("idempotency_key 不能为空")
        if not request.instruction.strip():
            raise LocalRunnerConfigError("instruction 不能为空")
        if not request.app.strip():
            raise LocalRunnerConfigError("app 不能为空")
        if request.timeout_sec <= 0:
            raise LocalRunnerConfigError("timeout_sec 必须是正数")
        if request.attempt_index < 0:
            raise LocalRunnerConfigError("attempt_index 不能为负数")

    @staticmethod
    def _running_result(idempotency_key: str) -> RunResult:
        return RunResult(
            schema_version=RUNNER_SCHEMA_VERSION,
            idempotency_key=idempotency_key,
            execution_status=ExecutionStatus.RUNNING,
            infrastructure_status=InfrastructureStatus.UNKNOWN,
            instruction="",
            app="",
            attempt_index=0,
            injected_plan_text="",
        )

    def _finalize_stale_attempt(
        self,
        idempotency_key: str = "",
        request: Optional[RunRequest] = None,
    ) -> RunResult:
        """把进程重启前遗留的进行中 attempt 定稿为 runner_error + retryable。"""
        key = idempotency_key or (request.idempotency_key if request else "")
        # 身份优先取落盘的 request.json；拿不到时退回当前请求。
        stored = self._store.load_request(key)
        source = stored or request
        result = RunResult(
            schema_version=RUNNER_SCHEMA_VERSION,
            idempotency_key=key,
            execution_status=ExecutionStatus.RUNNER_ERROR,
            infrastructure_status=InfrastructureStatus.RUNNER_ERROR,
            instruction=source.instruction if source else "",
            app=source.app if source else "",
            attempt_index=source.attempt_index if source else 0,
            injected_plan_text=source.plan_text if source else "",
            error={
                "type": "process_restarted",
                "message": "进程重启前 attempt 未结束，执行状态未知",
                "retryable": True,
            },
            runner_version=LOCAL_RUNNER_VERSION,
        )
        self._store.clear_in_progress(key)
        self._store.save_result(result)
        return result

    # ------------------------------------------------------------------
    # 执行主流程
    # ------------------------------------------------------------------

    async def _execute(self, request: RunRequest) -> RunResult:
        self._store.save_request(request)
        self._store.mark_in_progress(request.idempotency_key, request)
        started_at = datetime.now()
        timeout_sec = request.timeout_sec or self._default_timeout_sec
        async with self._semaphore:
            device_id = await self._select_device()
            if device_id is None:
                return self._persist(
                    self._compose_result(
                        request,
                        device_id="",
                        execution_status=ExecutionStatus.DEVICE_ERROR,
                        infrastructure_status=InfrastructureStatus.DEVICE_ERROR,
                        trajectory=None,
                        raw_ref=None,
                        error={
                            "type": "no_idle_device",
                            "message": "没有空闲设备可用",
                            "retryable": True,
                        },
                        started_at=started_at,
                        finished_at=datetime.now(),
                        artifacts={},
                    )
                )
            try:
                return await self._execute_on_device(
                    request, device_id, started_at, timeout_sec
                )
            finally:
                self._busy_devices.discard(device_id)

    async def _execute_on_device(
        self,
        request: RunRequest,
        device_id: str,
        started_at: datetime,
        timeout_sec: int,
    ) -> RunResult:
        if not self._config_ensured:
            # load_config 写进程级 os.environ，Runner 生命周期内只加载一次。
            await asyncio.to_thread(self._task_api.ensure_config)
            self._config_ensured = True
        try:
            task = await asyncio.to_thread(
                self._task_api.create_task,
                request.instruction,
                device_id,
                request.plan_text,
            )
        except Exception as exc:
            return self._persist(
                self._compose_result(
                    request,
                    device_id=device_id,
                    execution_status=ExecutionStatus.RUNNER_ERROR,
                    infrastructure_status=InfrastructureStatus.RUNNER_ERROR,
                    trajectory=None,
                    raw_ref=None,
                    error={
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "retryable": True,
                    },
                    started_at=started_at,
                    finished_at=datetime.now(),
                    artifacts={},
                )
            )

        attempt = _ActiveAttempt(
            idempotency_key=request.idempotency_key,
            device_id=device_id,
            task=task,
        )
        self._active[request.idempotency_key] = attempt

        def _worker() -> None:
            try:
                step_data_path = task.execute()
                attempt.finish_with("done", step_data_path=step_data_path)
            except BaseException as exc:  # noqa: BLE001 - 记录后由主流程分类
                attempt.finish_with("error", error=exc)

        worker_thread = threading.Thread(
            target=_worker,
            name=f"runner-attempt-{request.idempotency_key}",
            daemon=True,
        )
        worker_thread.start()
        try:
            return await self._await_attempt(
                request, attempt, device_id, started_at, timeout_sec
            )
        finally:
            self._active.pop(request.idempotency_key, None)

    async def _await_attempt(
        self,
        request: RunRequest,
        attempt: _ActiveAttempt,
        device_id: str,
        started_at: datetime,
        timeout_sec: int,
    ) -> RunResult:
        deadline = time.monotonic() + timeout_sec
        while not attempt.exit_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(_POLL_INTERVAL_SEC, remaining))

        if attempt.exit_event.is_set():
            if attempt.cancel_requested:
                return self._build_cancelled_result(request, device_id, started_at)
            status = attempt.outcome.get("status")
            if status == "done":
                return await self._build_finished_result(
                    request, device_id, started_at
                )
            return self._build_exception_result(
                request, device_id, started_at, attempt.outcome.get("error")
            )

        # 超时路径：协作式停止 + 有限宽限期。
        self._safe_stop_task(attempt)
        grace_deadline = time.monotonic() + self._stop_grace_sec
        while not attempt.exit_event.is_set():
            remaining = grace_deadline - time.monotonic()
            if remaining <= 0:
                # 宽限期后线程仍未退出：隔离设备，但不阻塞结果返回。
                self._unavailable_devices.add(device_id)
                break
            await asyncio.sleep(min(_POLL_INTERVAL_SEC, remaining))
        if attempt.exit_event.is_set() and attempt.cancel_requested:
            # 宽限期内外部取消先到：按取消定稿。
            return self._build_cancelled_result(request, device_id, started_at)
        return self._build_timeout_result(request, device_id, started_at)

    def _safe_stop_task(self, attempt: _ActiveAttempt) -> None:
        if attempt.task is None:
            return
        try:
            attempt.task.stop()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 结果构造
    # ------------------------------------------------------------------

    async def _build_finished_result(
        self,
        request: RunRequest,
        device_id: str,
        started_at: datetime,
    ) -> RunResult:
        attempt = self._active.get(request.idempotency_key)
        step_data_path = (attempt.outcome.get("step_data_path") if attempt else None)
        finished = datetime.now()
        if not step_data_path or not Path(step_data_path).is_file():
            # execute() 返回 None 或文件不存在：runner_error，不能当业务失败。
            return self._persist(
                self._compose_result(
                    request,
                    device_id=device_id,
                    execution_status=ExecutionStatus.RUNNER_ERROR,
                    infrastructure_status=InfrastructureStatus.RUNNER_ERROR,
                    trajectory=None,
                    raw_ref=None,
                    error={
                        "type": "no_step_data",
                        "message": "任务执行未产出 step_data.json",
                        "retryable": True,
                    },
                    started_at=started_at,
                    finished_at=finished,
                    artifacts={},
                )
            )

        try:
            raw_ref = await asyncio.to_thread(
                self._copy_raw_artifacts, request.idempotency_key, step_data_path
            )
            step_data = await asyncio.to_thread(self._read_step_data, raw_ref)
            trajectory = StepDataTrajectoryAdapter.step_data_to_trajectory(
                step_data, request
            )
        except (OSError, ValueError, LocalRunnerConfigError) as exc:
            return self._persist(
                self._compose_result(
                    request,
                    device_id=device_id,
                    execution_status=ExecutionStatus.RUNNER_ERROR,
                    infrastructure_status=InfrastructureStatus.RUNNER_ERROR,
                    trajectory=None,
                    raw_ref=None,
                    error={
                        "type": "step_data_invalid",
                        "message": str(exc),
                        "retryable": True,
                    },
                    started_at=started_at,
                    finished_at=datetime.now(),
                    artifacts={},
                )
            )

        trajectory_path = self._store.save_trajectory(
            request.idempotency_key, trajectory
        )
        artifacts = {"trajectory": str(trajectory_path)}
        return self._persist(
            self._compose_result(
                request,
                device_id=device_id,
                execution_status=ExecutionStatus.FINISHED,
                infrastructure_status=InfrastructureStatus.OK,
                trajectory=trajectory,
                raw_ref=raw_ref,
                error=None,
                started_at=started_at,
                finished_at=finished,
                artifacts=artifacts,
            )
        )

    def _copy_raw_artifacts(
        self, idempotency_key: str, step_data_path: str
    ) -> str:
        """把 ``DATA_DIR/<task_id>`` 复制到 ``result_root/<key>/raw/``。

        task_id 是秒级时间戳，原始目录可能被后续任务覆盖，因此必须留副本。
        """
        source_dir = Path(step_data_path).parent
        raw_dir = self._store.raw_dir(idempotency_key)
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        shutil.copytree(source_dir, raw_dir)
        return str(raw_dir / Path(step_data_path).name)

    @staticmethod
    def _read_step_data(raw_step_data_path: str) -> Dict[str, Any]:
        with open(raw_step_data_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _build_cancelled_result(
        self,
        request: RunRequest,
        device_id: str,
        started_at: datetime,
    ) -> RunResult:
        return self._persist(
            self._compose_result(
                request,
                device_id=device_id,
                execution_status=ExecutionStatus.CANCELLED,
                infrastructure_status=InfrastructureStatus.UNKNOWN,
                trajectory=None,
                raw_ref=None,
                error={
                    "type": "cancelled",
                    "message": "attempt 被取消",
                    "retryable": False,
                },
                started_at=started_at,
                finished_at=datetime.now(),
                artifacts={},
            )
        )

    def _build_timeout_result(
        self,
        request: RunRequest,
        device_id: str,
        started_at: datetime,
    ) -> RunResult:
        return self._persist(
            self._compose_result(
                request,
                device_id=device_id,
                execution_status=ExecutionStatus.TIMEOUT,
                infrastructure_status=InfrastructureStatus.TIMEOUT,
                trajectory=None,
                raw_ref=None,
                error={
                    "type": "timeout",
                    "message": f"attempt 超过 {request.timeout_sec}s 未结束",
                    "retryable": True,
                },
                started_at=started_at,
                finished_at=datetime.now(),
                artifacts={},
            )
        )

    def _build_exception_result(
        self,
        request: RunRequest,
        device_id: str,
        started_at: datetime,
        error: Optional[BaseException],
    ) -> RunResult:
        kind = "runner_error"
        if error is not None:
            try:
                kind = self._task_api.classify_error(error)
            except Exception:
                kind = "runner_error"
        if kind == "device_error":
            execution_status = ExecutionStatus.DEVICE_ERROR
            infrastructure_status = InfrastructureStatus.DEVICE_ERROR
        else:
            execution_status = ExecutionStatus.RUNNER_ERROR
            infrastructure_status = InfrastructureStatus.RUNNER_ERROR
        return self._persist(
            self._compose_result(
                request,
                device_id=device_id,
                execution_status=execution_status,
                infrastructure_status=infrastructure_status,
                trajectory=None,
                raw_ref=None,
                error={
                    "type": type(error).__name__ if error else "unknown",
                    "message": str(error) if error else "",
                    "retryable": True,
                },
                started_at=started_at,
                finished_at=datetime.now(),
                artifacts={},
            )
        )

    def _compose_result(
        self,
        request: RunRequest,
        device_id: str,
        execution_status: ExecutionStatus,
        infrastructure_status: InfrastructureStatus,
        trajectory: Optional[Dict[str, Any]],
        raw_ref: Optional[str],
        error: Optional[Dict[str, Any]],
        started_at: datetime,
        finished_at: datetime,
        artifacts: Dict[str, str],
    ) -> RunResult:
        key_dir = self._store.key_dir(request.idempotency_key)
        artifacts = dict(artifacts)
        artifacts.setdefault("request", str(key_dir / _REQUEST_FILE))
        artifacts.setdefault("result", str(key_dir / _RESULT_FILE))
        duration = (finished_at - started_at).total_seconds()
        return RunResult(
            schema_version=RUNNER_SCHEMA_VERSION,
            idempotency_key=request.idempotency_key,
            execution_status=execution_status,
            infrastructure_status=infrastructure_status,
            instruction=request.instruction,
            app=request.app,
            attempt_index=request.attempt_index,
            injected_plan_text=request.plan_text,
            trajectory=trajectory,
            raw_ref=raw_ref,
            device_meta={"device_id": device_id} if device_id else {},
            run_context={
                "experience_mode": request.experience_mode.value,
                "accepted_patch_ids": list(request.accepted_patch_ids),
                "candidate_patch_id": request.candidate_patch_id,
                "initial_state": request.initial_state,
                "device_constraints": dict(request.device_constraints),
            },
            error=error,
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            duration_sec=duration,
            runner_version=LOCAL_RUNNER_VERSION,
            artifacts=artifacts,
        )

    def _persist(self, result: RunResult) -> RunResult:
        self._store.clear_in_progress(result.idempotency_key)
        self._store.save_result(result)
        return result

    # ------------------------------------------------------------------
    # 设备选择
    # ------------------------------------------------------------------

    async def _select_device(self) -> Optional[str]:
        devices = await asyncio.to_thread(self._refresh_devices)
        for device_id in devices:
            if device_id in self._busy_devices:
                continue
            if device_id in self._unavailable_devices:
                continue
            self._busy_devices.add(device_id)
            return device_id
        return None

    def _refresh_devices(self, force: bool = False) -> List[str]:
        now = time.monotonic()
        if (
            not force
            and self._device_cache is not None
            and now - self._device_cache[0] < self._device_cache_ttl_sec
        ):
            return list(self._device_cache[1])
        devices = list(self._task_api.list_devices())
        self._device_cache = (now, devices)
        return devices
