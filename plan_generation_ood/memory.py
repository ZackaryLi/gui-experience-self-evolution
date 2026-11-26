import hashlib
import json
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .experiment_plan import ExperimentPlan
from .recovery_schemas import RecoveryContext
from .schemas import (
    AttemptResult,
    AttemptVerdict,
    BaselinePlan,
    DictSerializable,
    DiagnosisReport,
    EvidenceRef,
    ExperienceDraft,
    ExperimentCard,
    RecoveryState,
    RecoveryStateSnapshot,
    RecoveryValueAssessment,
    content_hash,
)
from .storage import append_jsonl, atomic_write_json


RECOVERY_MEMORY_SCHEMA_VERSION = "1.0"
STATE_ARTIFACTS = frozenset({
    "diagnosis.json",
    "recovery_value.json",
    "initial_experience.json",
    "initial_experience_error.json",
    "plan.json",
})


class RecoveryMemoryError(ValueError):
    pass


class StateConflictError(RecoveryMemoryError):
    pass


class ArtifactConflictError(RecoveryMemoryError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecoveryMemoryError(f"{name} 必须是非空字符串")
    return value.strip()


def _exact_fields(payload: Any, expected: set, label: str) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise RecoveryMemoryError(f"{label} 必须是 JSON 对象")
    actual = set(payload)
    if actual != expected:
        raise RecoveryMemoryError(
            f"{label} 字段不匹配，缺少 {sorted(expected - actual)}，"
            f"未知 {sorted(actual - expected)}"
        )
    return payload


def _read_json(path: Path, label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RecoveryMemoryError(f"{label} 不存在: {path}") from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RecoveryMemoryError(f"{label} 已损坏: {path}") from error
    if not isinstance(payload, dict):
        raise RecoveryMemoryError(f"{label} 必须是 JSON 对象")
    return payload


def _read_jsonl(path: Path, label: str) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise RecoveryMemoryError(f"{label} 编码损坏: {path}") from error
    records = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise RecoveryMemoryError(f"{label} 包含空行或部分写入: {line_number}")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise RecoveryMemoryError(
                f"{label} 包含损坏或部分写入行: {line_number}"
            ) from error
        if not isinstance(payload, dict):
            raise RecoveryMemoryError(f"{label} 第 {line_number} 行不是对象")
        records.append(payload)
    return records


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experience_version_id(experience: ExperienceDraft) -> str:
    if not isinstance(experience, ExperienceDraft):
        raise RecoveryMemoryError("experience 必须是 ExperienceDraft")
    _required_string(experience.experience_id, "experience_id")
    if (
        isinstance(experience.version, bool)
        or not isinstance(experience.version, int)
        or experience.version <= 0
    ):
        raise RecoveryMemoryError("经验 version 必须是正整数")
    return f"{experience.experience_id}:version-{experience.version}"


def _validate_artifact_hashes(root: Path, artifact_hashes: Dict[str, str]) -> None:
    if not isinstance(artifact_hashes, dict):
        raise RecoveryMemoryError("artifact_hashes 必须是对象")
    for name, expected_hash in artifact_hashes.items():
        if name not in STATE_ARTIFACTS:
            raise RecoveryMemoryError(f"未知状态产物: {name}")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise RecoveryMemoryError(f"产物哈希非法: {name}")
        path = root / name
        if not path.is_file():
            raise RecoveryMemoryError(f"状态引用的产物不存在: {path}")
        if file_sha256(path) != expected_hash:
            raise ArtifactConflictError(f"状态引用的产物已被修改: {path}")


class StateStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.path = self.root / "state.json"

    def initialize(
        self,
        plan: BaselinePlan,
        budget: Optional[Dict[str, Any]] = None,
    ) -> RecoveryStateSnapshot:
        if not isinstance(plan, BaselinePlan) or plan.status != "FROZEN":
            raise RecoveryMemoryError("初始化状态需要 FROZEN BaselinePlan")
        if self.path.exists():
            snapshot = self.load()
            if snapshot.intent_id != plan.intent_id:
                raise StateConflictError("恢复目录已属于另一个 intent")
            if snapshot.baseline_plan_id != plan.plan_id:
                raise StateConflictError("当前状态绑定的冻结基线计划不一致")
            if budget is not None and snapshot.budget != budget:
                raise StateConflictError("不能用新预算覆盖已持久化预算")
            return snapshot
        now = _utc_now()
        snapshot = RecoveryStateSnapshot(
            intent_id=_required_string(plan.intent_id, "intent_id"),
            state=RecoveryState.BASELINE_PENDING,
            baseline_plan_id=_required_string(plan.plan_id, "plan_id"),
            budget=dict(budget or {}),
            allowed_actions=["ANALYZE_BASELINE"],
            created_at=now,
            updated_at=now,
        )
        self._validate_snapshot(snapshot)
        atomic_write_json(self.path, snapshot.to_dict())
        return snapshot

    def load(self) -> RecoveryStateSnapshot:
        payload = _read_json(self.path, "Recovery 状态")
        _exact_fields(
            payload,
            {item.name for item in fields(RecoveryStateSnapshot)},
            "Recovery 状态",
        )
        data = dict(payload)
        try:
            data["state"] = RecoveryState(data["state"])
        except (TypeError, ValueError) as error:
            raise RecoveryMemoryError("Recovery 状态包含未知状态值") from error
        snapshot = RecoveryStateSnapshot(**data)
        self._validate_snapshot(snapshot)
        return snapshot

    def commit(
        self,
        snapshot: RecoveryStateSnapshot,
        expected_revision: int,
        transition_id: str,
    ) -> Tuple[RecoveryStateSnapshot, bool]:
        current = self.load()
        if current.last_transition_id == transition_id:
            if (
                current.state is snapshot.state
                and current.revision == snapshot.revision
            ):
                return current, False
            raise StateConflictError("状态已包含相同迁移 ID，但内容不一致")
        if current.revision != expected_revision:
            raise StateConflictError(
                f"状态版本冲突：期望 {expected_revision}，实际 {current.revision}"
            )
        if current.intent_id != snapshot.intent_id:
            raise StateConflictError("状态迁移不能修改 intent")
        if snapshot.revision != expected_revision + 1:
            raise StateConflictError("新状态 revision 必须恰好增加 1")
        if snapshot.last_transition_id != transition_id:
            raise StateConflictError("新状态缺少对应迁移 ID")
        self._validate_snapshot(snapshot)
        atomic_write_json(self.path, snapshot.to_dict())
        return snapshot, True

    @staticmethod
    def _validate_snapshot(snapshot: RecoveryStateSnapshot) -> None:
        _required_string(snapshot.intent_id, "state.intent_id")
        _required_string(snapshot.baseline_plan_id, "state.baseline_plan_id")
        if snapshot.schema_version != RECOVERY_MEMORY_SCHEMA_VERSION:
            raise RecoveryMemoryError("不支持的 Recovery 状态 Schema 版本")
        if not isinstance(snapshot.state, RecoveryState):
            raise RecoveryMemoryError("state 必须是 RecoveryState")
        if (
            isinstance(snapshot.revision, bool)
            or not isinstance(snapshot.revision, int)
            or snapshot.revision < 0
        ):
            raise RecoveryMemoryError("state.revision 必须是非负整数")
        if (
            isinstance(snapshot.round_index, bool)
            or not isinstance(snapshot.round_index, int)
            or snapshot.round_index < 0
        ):
            raise RecoveryMemoryError("state.round_index 必须是非负整数")
        if not isinstance(snapshot.budget, dict) or not isinstance(snapshot.history, list):
            raise RecoveryMemoryError("state.budget/history 类型非法")
        if not isinstance(snapshot.allowed_actions, list) or any(
            not isinstance(item, str) or not item.strip()
            for item in snapshot.allowed_actions
        ):
            raise RecoveryMemoryError("state.allowed_actions 必须是字符串数组")
        if snapshot.current_experience_id is not None:
            _required_string(snapshot.current_experience_id, "current_experience_id")
        if not isinstance(snapshot.current_hypothesis, str):
            raise RecoveryMemoryError("current_hypothesis 必须是字符串")
        if not isinstance(snapshot.blocker_reason, str):
            raise RecoveryMemoryError("blocker_reason 必须是字符串")
        if snapshot.last_transition_id and len(snapshot.last_transition_id) != 64:
            raise RecoveryMemoryError("last_transition_id 哈希非法")
        _required_string(snapshot.created_at, "state.created_at")
        _required_string(snapshot.updated_at, "state.updated_at")


def _journal_identity(
    intent_id: str,
    from_state: RecoveryState,
    to_state: RecoveryState,
    from_revision: int,
    stage: str,
    reason: str,
    artifact_hashes: Dict[str, str],
    experience_version_id: Optional[str],
    hypothesis: str,
    budget: Optional[Dict[str, Any]] = None,
    round_index: Optional[int] = None,
) -> str:
    payload = {
        "intent_id": intent_id,
        "from_state": from_state.value,
        "to_state": to_state.value,
        "from_revision": from_revision,
        "stage": stage,
        "reason": reason,
        "artifact_hashes": artifact_hashes,
        "experience_version_id": experience_version_id,
        "hypothesis": hypothesis,
    }
    if budget is not None:
        payload["budget"] = budget
    if round_index is not None:
        payload["round_index"] = round_index
    return content_hash(payload)


@dataclass(frozen=True)
class JournalEntry(DictSerializable):
    sequence: int
    transition_id: str
    intent_id: str
    from_state: RecoveryState
    to_state: RecoveryState
    from_revision: int
    to_revision: int
    stage: str
    reason: str
    artifact_hashes: Dict[str, str] = field(default_factory=dict)
    experience_version_id: Optional[str] = None
    hypothesis: str = ""
    budget: Optional[Dict[str, Any]] = None
    round_index: Optional[int] = None
    occurred_at: str = field(default_factory=_utc_now)

    @classmethod
    def create(
        cls,
        sequence: int,
        snapshot: RecoveryStateSnapshot,
        to_state: RecoveryState,
        stage: str,
        reason: str,
        artifact_hashes: Dict[str, str],
        experience_version_id: Optional[str] = None,
        hypothesis: str = "",
        budget: Optional[Dict[str, Any]] = None,
        round_index: Optional[int] = None,
    ) -> "JournalEntry":
        transition_id = _journal_identity(
            snapshot.intent_id,
            snapshot.state,
            to_state,
            snapshot.revision,
            stage,
            reason,
            artifact_hashes,
            experience_version_id,
            hypothesis,
            budget,
            round_index,
        )
        return cls(
            sequence=sequence,
            transition_id=transition_id,
            intent_id=snapshot.intent_id,
            from_state=snapshot.state,
            to_state=to_state,
            from_revision=snapshot.revision,
            to_revision=snapshot.revision + 1,
            stage=stage,
            reason=reason,
            artifact_hashes=dict(artifact_hashes),
            experience_version_id=experience_version_id,
            hypothesis=hypothesis,
            budget=dict(budget) if budget is not None else None,
            round_index=round_index,
        )

    def validate(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence <= 0:
            raise RecoveryMemoryError("journal sequence 必须是正整数")
        for name in ("intent_id", "stage", "reason", "occurred_at"):
            _required_string(getattr(self, name), f"journal.{name}")
        if not isinstance(self.from_state, RecoveryState) or not isinstance(
            self.to_state, RecoveryState
        ):
            raise RecoveryMemoryError("journal 状态类型非法")
        if self.to_revision != self.from_revision + 1 or self.from_revision < 0:
            raise RecoveryMemoryError("journal revision 不连续")
        if self.budget is not None and not isinstance(self.budget, dict):
            raise RecoveryMemoryError("journal.budget 必须是对象或 null")
        if self.round_index is not None and (
            isinstance(self.round_index, bool)
            or not isinstance(self.round_index, int)
            or self.round_index < 0
        ):
            raise RecoveryMemoryError("journal.round_index 必须是非负整数或 null")
        expected = _journal_identity(
            self.intent_id,
            self.from_state,
            self.to_state,
            self.from_revision,
            self.stage,
            self.reason,
            self.artifact_hashes,
            self.experience_version_id,
            self.hypothesis,
            self.budget,
            self.round_index,
        )
        if self.transition_id != expected:
            raise RecoveryMemoryError("journal 迁移 ID 与内容不一致")


class WorkflowJournal:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.path = self.root / "journal.jsonl"

    def load(self) -> List[JournalEntry]:
        entries = []
        for payload in _read_jsonl(self.path, "Workflow journal"):
            if "budget" not in payload:
                payload = {**payload, "budget": None}
            if "round_index" not in payload:
                payload = {**payload, "round_index": None}
            _exact_fields(
                payload,
                {item.name for item in fields(JournalEntry)},
                "journal 记录",
            )
            data = dict(payload)
            try:
                data["from_state"] = RecoveryState(data["from_state"])
                data["to_state"] = RecoveryState(data["to_state"])
            except (TypeError, ValueError) as error:
                raise RecoveryMemoryError("journal 包含未知状态") from error
            entry = JournalEntry(**data)
            entry.validate()
            entries.append(entry)
        for index, entry in enumerate(entries, start=1):
            if entry.sequence != index:
                raise RecoveryMemoryError("journal sequence 必须连续且从 1 开始")
            if index > 1:
                previous = entries[index - 2]
                if (
                    entry.from_state is not previous.to_state
                    or entry.from_revision != previous.to_revision
                ):
                    raise RecoveryMemoryError("journal 前后状态或 revision 不连续")
        return entries

    def append(self, entry: JournalEntry) -> Tuple[JournalEntry, bool]:
        entry.validate()
        entries = self.load()
        for existing in entries:
            if existing.transition_id == entry.transition_id:
                comparable = replace(existing, occurred_at=entry.occurred_at)
                if comparable != entry:
                    raise StateConflictError("相同迁移 ID 对应不同 journal 内容")
                return existing, False
        expected_sequence = len(entries) + 1
        if entry.sequence != expected_sequence:
            raise StateConflictError("journal 追加序号冲突")
        if entries:
            previous = entries[-1]
            if (
                entry.from_state is not previous.to_state
                or entry.from_revision != previous.to_revision
            ):
                raise StateConflictError("journal 追加与最后状态不连续")
        elif entry.from_revision != 0:
            raise StateConflictError("首条 journal 必须从 revision 0 开始")
        _validate_artifact_hashes(self.root, entry.artifact_hashes)
        append_jsonl(self.path, entry.to_dict())
        return entry, True


@dataclass(frozen=True)
class ExperienceVersionRecord(DictSerializable):
    intent_id: str
    experience_version_id: str
    experience: ExperienceDraft
    source: str
    validation_level: str
    supporting_evidence_refs: List[EvidenceRef]
    opposing_evidence_refs: List[EvidenceRef]
    selected_for_ind: bool
    source_failure_pattern_id: str
    source_details: Dict[str, Any]
    record_hash: str
    created_at: str

    @classmethod
    def create(
        cls,
        experience: ExperienceDraft,
        source: str,
        source_failure_pattern_id: str = "",
        source_details: Optional[Dict[str, Any]] = None,
    ) -> "ExperienceVersionRecord":
        data = {
            "intent_id": experience.intent_id,
            "experience_version_id": experience_version_id(experience),
            "experience": experience.to_dict(),
            "source": _required_string(source, "experience.source"),
            "validation_level": "CANDIDATE",
            "supporting_evidence_refs": [
                item.to_dict() for item in experience.evidence_refs
            ],
            "opposing_evidence_refs": [],
            "selected_for_ind": False,
            "source_failure_pattern_id": source_failure_pattern_id,
            "source_details": dict(source_details or {}),
        }
        return cls(
            intent_id=experience.intent_id,
            experience_version_id=experience_version_id(experience),
            experience=experience,
            source=data["source"],
            validation_level="CANDIDATE",
            supporting_evidence_refs=list(experience.evidence_refs),
            opposing_evidence_refs=[],
            selected_for_ind=False,
            source_failure_pattern_id=source_failure_pattern_id,
            source_details=dict(source_details or {}),
            record_hash=content_hash(data),
            created_at=_utc_now(),
        )


class ExperienceRegistry:
    def __init__(self, root: str | Path, intent_id: str):
        self.root = Path(root)
        self.intent_id = _required_string(intent_id, "intent_id")
        self.path = self.root / "experiences.jsonl"

    def register(
        self,
        experience: ExperienceDraft,
        source: str,
        source_failure_pattern_id: str = "",
        source_details: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ExperienceVersionRecord, bool]:
        if experience.intent_id != self.intent_id:
            raise RecoveryMemoryError("经验版本与恢复目录 intent 不一致")
        record = ExperienceVersionRecord.create(
            experience,
            source,
            source_failure_pattern_id,
            source_details,
        )
        records = self.load()
        version_id = record.experience_version_id
        for existing in records:
            if existing.experience_version_id != version_id:
                continue
            if (
                existing.experience == experience
                and existing.source == record.source
                and existing.source_failure_pattern_id == record.source_failure_pattern_id
                and existing.source_details == record.source_details
            ):
                return existing, False
            raise StateConflictError(f"经验版本 ID 已存在但内容不同: {version_id}")
        if experience.version == 1:
            if experience.parent_experience_id is not None:
                raise RecoveryMemoryError("version=1 不能有父经验")
        else:
            parent = next((
                item for item in records
                if item.experience.experience_id == experience.parent_experience_id
                and item.experience.version == experience.version - 1
            ), None)
            if parent is None:
                raise RecoveryMemoryError("后续经验的直接父版本未注册")
        append_jsonl(self.path, record.to_dict())
        return record, True

    def load(self) -> List[ExperienceVersionRecord]:
        records = []
        seen = set()
        for payload in _read_jsonl(self.path, "经验注册表"):
            _exact_fields(
                payload,
                {item.name for item in fields(ExperienceVersionRecord)},
                "经验版本",
            )
            data = dict(payload)
            data["experience"] = _experience_from_dict(data["experience"])
            data["supporting_evidence_refs"] = _evidence_from_dicts(
                data["supporting_evidence_refs"]
            )
            data["opposing_evidence_refs"] = _evidence_from_dicts(
                data["opposing_evidence_refs"]
            )
            record = ExperienceVersionRecord(**data)
            self._validate_record(record)
            if record.experience_version_id in seen:
                raise RecoveryMemoryError("经验注册表包含重复版本")
            seen.add(record.experience_version_id)
            records.append(record)
        for record in records:
            experience = record.experience
            if experience.version > 1 and not any(
                item.experience.experience_id == experience.parent_experience_id
                and item.experience.version == experience.version - 1
                for item in records
            ):
                raise RecoveryMemoryError("经验注册表存在缺失的父版本")
        return records

    def get(self, version_id: str) -> ExperienceVersionRecord:
        for record in self.load():
            if record.experience_version_id == version_id:
                return record
        raise RecoveryMemoryError(f"经验版本不存在: {version_id}")

    def _validate_record(self, record: ExperienceVersionRecord) -> None:
        if record.intent_id != self.intent_id or record.experience.intent_id != self.intent_id:
            raise RecoveryMemoryError("经验注册表包含其他 intent 的记录")
        if record.experience_version_id != experience_version_id(record.experience):
            raise RecoveryMemoryError("experience_version_id 与经验内容不一致")
        if record.validation_level != "CANDIDATE" or record.selected_for_ind:
            raise RecoveryMemoryError("候选经验不能伪造验证或 IND 状态")
        expected = ExperienceVersionRecord.create(
            record.experience,
            record.source,
            record.source_failure_pattern_id,
            record.source_details,
        )
        comparable = replace(record, record_hash=expected.record_hash)
        if comparable.to_dict() != replace(
            expected,
            created_at=record.created_at,
        ).to_dict():
            raise RecoveryMemoryError("经验版本内容或证据已被篡改")
        if record.record_hash != expected.record_hash:
            raise RecoveryMemoryError("经验版本 record_hash 校验失败")


@dataclass(frozen=True)
class ExperimentEvent(DictSerializable):
    sequence: int
    event_id: str
    intent_id: str
    experiment_id: str
    event_type: str
    experience_version_id: str
    payload: Dict[str, Any]
    event_hash: str
    occurred_at: str

    @classmethod
    def create(
        cls,
        sequence: int,
        intent_id: str,
        experiment_id: str,
        event_type: str,
        experience_version_id: str,
        payload: Dict[str, Any],
    ) -> "ExperimentEvent":
        event_id = content_hash({
            "intent_id": intent_id,
            "experiment_id": experiment_id,
            "event_type": event_type,
            "experience_version_id": experience_version_id,
            "payload": payload,
        })
        event_hash = content_hash({
            "event_id": event_id,
            "sequence": sequence,
        })
        return cls(
            sequence=sequence,
            event_id=event_id,
            intent_id=intent_id,
            experiment_id=experiment_id,
            event_type=event_type,
            experience_version_id=experience_version_id,
            payload=payload,
            event_hash=event_hash,
            occurred_at=_utc_now(),
        )

    def validate(self, intent_id: str) -> None:
        if self.intent_id != intent_id:
            raise RecoveryMemoryError("实验日志包含其他 intent 的记录")
        if self.event_type not in {
            "EXPERIMENT_PLANNED",
            "ATTEMPT_RESULT_RECORDED",
            "VERDICT_RECORDED",
        }:
            raise RecoveryMemoryError("实验日志包含未知事件类型")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence <= 0:
            raise RecoveryMemoryError("实验日志 sequence 非法")
        expected = ExperimentEvent.create(
            self.sequence,
            self.intent_id,
            self.experiment_id,
            self.event_type,
            self.experience_version_id,
            self.payload,
        )
        if self.event_id != expected.event_id or self.event_hash != expected.event_hash:
            raise RecoveryMemoryError("实验事件哈希校验失败")


class ExperimentStore:
    def __init__(
        self,
        root: str | Path,
        intent_id: str,
        registry: Optional[ExperienceRegistry] = None,
    ):
        self.root = Path(root)
        self.intent_id = _required_string(intent_id, "intent_id")
        self.registry = registry
        self.path = self.root / "experiments.jsonl"

    def record_plan(self, card: ExperimentCard) -> Tuple[ExperimentEvent, bool]:
        if not isinstance(card, ExperimentCard) or card.intent_id != self.intent_id:
            raise RecoveryMemoryError("ExperimentCard intent 不一致")
        version_id = experience_version_id(card.experience)
        if self.registry is not None:
            self.registry.get(version_id)
        payload = {"card": card.to_dict()}
        return self._append_unique(
            card.experiment_id,
            "EXPERIMENT_PLANNED",
            version_id,
            payload,
        )

    def record_attempt(
        self,
        experiment_id: str,
        result: AttemptResult,
    ) -> Tuple[ExperimentEvent, bool]:
        if not isinstance(result, AttemptResult):
            raise RecoveryMemoryError("result 必须是 AttemptResult")
        events = self.load()
        planned = next((
            event for event in events
            if event.experiment_id == experiment_id
            and event.event_type == "EXPERIMENT_PLANNED"
        ), None)
        if planned is None:
            raise RecoveryMemoryError("记录结果前必须先登记 ExperimentCard")
        card = planned.payload["card"]
        if result.request.intent_id != self.intent_id:
            raise RecoveryMemoryError("实验结果 intent 不一致")
        if result.request.experience is None:
            raise RecoveryMemoryError("候选经验实验结果必须包含实际使用的经验")
        if result.request.experience.to_dict() != card["experience"]:
            raise RecoveryMemoryError("实验结果使用的经验与 ExperimentCard 不一致")
        if (
            result.request.plan_id != card["baseline_plan_id"]
            or result.request.instruction_id not in card["instruction_ids"]
            or not 1 <= result.request.attempt_index <= card["attempts_per_instruction"]
        ):
            raise RecoveryMemoryError("实验结果请求与 ExperimentCard 的执行范围不一致")
        payload = {"result": result.to_dict()}
        return self._append_unique(
            experiment_id,
            "ATTEMPT_RESULT_RECORDED",
            planned.experience_version_id,
            payload,
            events=events,
        )

    def record_verdict(
        self,
        experiment_id: str,
        verdict: AttemptVerdict,
    ) -> Tuple[ExperimentEvent, bool]:
        if not isinstance(verdict, AttemptVerdict):
            raise RecoveryMemoryError("verdict 必须是 AttemptVerdict")
        events = self.load()
        planned = next((
            event for event in events
            if event.experiment_id == experiment_id
            and event.event_type == "EXPERIMENT_PLANNED"
        ), None)
        if planned is None:
            raise RecoveryMemoryError("记录 Verdict 前必须先登记实验卡")
        attempt = next((
            event for event in events
            if event.experiment_id == experiment_id
            and event.event_type == "ATTEMPT_RESULT_RECORDED"
            and event.payload["result"]["request"]["idempotency_key"] == verdict.attempt_key
        ), None)
        if attempt is None:
            raise RecoveryMemoryError("Verdict 必须对应已保存的 attempt")
        if attempt.payload["result"]["trajectory"].get("sessionid") != verdict.sessionid:
            raise RecoveryMemoryError("Verdict sessionid 与原始轨迹不一致")
        return self._append_unique(
            experiment_id,
            "VERDICT_RECORDED",
            planned.experience_version_id,
            {"verdict": verdict.to_dict()},
            events=events,
        )

    def query(
        self,
        experiment_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[ExperimentEvent]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
            raise RecoveryMemoryError("实验历史 limit 必须位于 1 和 50 之间")
        events = [
            event for event in self.load()
            if (experiment_id is None or event.experiment_id == experiment_id)
            and (event_type is None or event.event_type == event_type)
        ]
        return list(reversed(events[-limit:]))

    def _append_unique(
        self,
        experiment_id: str,
        event_type: str,
        version_id: str,
        payload: Dict[str, Any],
        events: Optional[List[ExperimentEvent]] = None,
    ) -> Tuple[ExperimentEvent, bool]:
        _required_string(experiment_id, "experiment_id")
        events = self.load() if events is None else events
        candidate = ExperimentEvent.create(
            len(events) + 1,
            self.intent_id,
            experiment_id,
            event_type,
            version_id,
            payload,
        )
        for existing in events:
            if existing.event_id == candidate.event_id:
                return existing, False
        if event_type == "EXPERIMENT_PLANNED" and any(
            event.experiment_id == experiment_id
            and event.event_type == "EXPERIMENT_PLANNED"
            for event in events
        ):
            raise StateConflictError("同一 experiment_id 不能登记不同实验卡")
        if event_type in {"ATTEMPT_RESULT_RECORDED", "VERDICT_RECORDED"}:
            payload_key = "result" if event_type == "ATTEMPT_RESULT_RECORDED" else "verdict"
            attempt_key = (
                payload[payload_key]["request"]["idempotency_key"]
                if payload_key == "result"
                else payload[payload_key]["attempt_key"]
            )
            for existing in events:
                if existing.experiment_id != experiment_id or existing.event_type != event_type:
                    continue
                existing_key = (
                    existing.payload["result"]["request"]["idempotency_key"]
                    if event_type == "ATTEMPT_RESULT_RECORDED"
                    else existing.payload["verdict"]["attempt_key"]
                )
                if existing_key == attempt_key:
                    raise StateConflictError("同一 attempt 不能登记不同的结果或 Verdict")
        append_jsonl(self.path, candidate.to_dict())
        return candidate, True

    def load(self) -> List[ExperimentEvent]:
        events = []
        for payload in _read_jsonl(self.path, "实验日志"):
            _exact_fields(payload, {item.name for item in fields(ExperimentEvent)}, "实验事件")
            event = ExperimentEvent(**payload)
            event.validate(self.intent_id)
            events.append(event)
        for index, event in enumerate(events, start=1):
            if event.sequence != index:
                raise RecoveryMemoryError("实验日志 sequence 必须连续")
        return events


class RecoveryMemory:
    def __init__(self, root: str | Path, intent_id: str):
        self.root = Path(root)
        self.intent_id = _required_string(intent_id, "intent_id")
        self.states = StateStore(self.root)
        self.journal = WorkflowJournal(self.root)
        self.experiences = ExperienceRegistry(self.root, self.intent_id)
        self.experiments = ExperimentStore(
            self.root,
            self.intent_id,
            registry=self.experiences,
        )

    def initialize(
        self,
        plan: BaselinePlan,
        budget: Optional[Dict[str, Any]] = None,
    ) -> RecoveryStateSnapshot:
        if plan.intent_id != self.intent_id:
            raise RecoveryMemoryError("冻结计划与恢复目录 intent 不一致")
        snapshot = self.states.initialize(plan, budget)
        self.recover()
        return snapshot

    def recover(self) -> RecoveryStateSnapshot:
        snapshot = self.states.load()
        if snapshot.intent_id != self.intent_id:
            raise RecoveryMemoryError("状态文件与恢复目录 intent 不一致")
        entries = self.journal.load()
        self.experiences.load()
        self.experiments.load()
        if snapshot.current_experience_id:
            self.experiences.get(snapshot.current_experience_id)
        if not entries:
            _validate_artifact_hashes(self.root, snapshot.artifact_hashes)
            if snapshot.revision != 0 or snapshot.last_transition_id:
                raise RecoveryMemoryError("状态已有提交但 journal 为空")
            return snapshot
        last = entries[-1]
        _validate_artifact_hashes(self.root, last.artifact_hashes)
        if snapshot.last_transition_id == last.transition_id:
            _validate_artifact_hashes(self.root, snapshot.artifact_hashes)
            if (
                snapshot.state is not last.to_state
                or snapshot.revision != last.to_revision
            ):
                raise RecoveryMemoryError("状态与最后 journal 提交不一致")
            return snapshot
        if (
            snapshot.state is not last.from_state
            or snapshot.revision != last.from_revision
        ):
            raise RecoveryMemoryError("状态和 journal 之间出现未知分叉")
        previous_id = entries[-2].transition_id if len(entries) > 1 else ""
        if snapshot.last_transition_id != previous_id:
            raise RecoveryMemoryError("待提交 journal 的来源状态不明确")
        return snapshot

    def pending_transition(self) -> Optional[JournalEntry]:
        snapshot = self.states.load()
        entries = self.journal.load()
        if entries and snapshot.last_transition_id != entries[-1].transition_id:
            return entries[-1]
        return None

    def save_plan(self, plan: ExperimentPlan) -> None:
        if not isinstance(plan, ExperimentPlan) or plan.intent_id != self.intent_id:
            raise RecoveryMemoryError("实验计划 intent 不一致")
        path = self.root / "plan.json"
        if path.exists():
            existing = ExperimentPlan.load(path)
            old_payload = existing.to_dict()
            new_payload = plan.to_dict()
            if old_payload == new_payload:
                return
            old_changes = old_payload["changes"]
            if (
                old_payload["created_at"] != new_payload["created_at"]
                or new_payload["changes"][:len(old_changes)] != old_changes
            ):
                raise ArtifactConflictError("实验计划只能追加合法变更，不能重写历史")
            atomic_write_json(path, new_payload)
            return
        plan.save(path)

    def load_plan(self) -> ExperimentPlan:
        plan = ExperimentPlan.load(self.root / "plan.json")
        if plan.intent_id != self.intent_id:
            raise RecoveryMemoryError("实验计划与恢复目录 intent 不一致")
        return plan

    def build_context(
        self,
        diagnosis: Optional[DiagnosisReport] = None,
        recovery_value: Optional[RecoveryValueAssessment] = None,
    ) -> RecoveryContext:
        snapshot = self.recover()
        current = None
        if snapshot.current_experience_id:
            current = self.experiences.get(snapshot.current_experience_id).experience
        return RecoveryContext(
            intent_id=self.intent_id,
            state=snapshot.state,
            diagnosis=diagnosis,
            recovery_value=recovery_value,
            current_experience=current,
            evidence_refs=list(current.evidence_refs) if current else [],
            budget=dict(snapshot.budget),
            history=[event.to_dict() for event in self.experiments.query()],
            metadata={
                "revision": snapshot.revision,
                "baseline_plan_id": snapshot.baseline_plan_id,
                "blocker_reason": snapshot.blocker_reason,
                "allowed_actions": list(snapshot.allowed_actions),
            },
        )


def _experience_from_dict(payload: Any) -> ExperienceDraft:
    _exact_fields(payload, {item.name for item in fields(ExperienceDraft)}, "经验草稿")
    data = dict(payload)
    data["evidence_refs"] = _evidence_from_dicts(data["evidence_refs"])
    return ExperienceDraft(**data)


def _evidence_from_dicts(payload: Any) -> List[EvidenceRef]:
    if not isinstance(payload, list):
        raise RecoveryMemoryError("证据引用必须是数组")
    refs = []
    for item in payload:
        _exact_fields(item, {"sessionid", "step_ids"}, "证据引用")
        refs.append(EvidenceRef(sessionid=item["sessionid"], step_ids=item["step_ids"]))
    return refs
