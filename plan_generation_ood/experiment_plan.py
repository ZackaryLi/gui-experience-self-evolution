import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .recovery_schemas import RecoveryContext
from .schemas import (
    EvidenceRef,
    RecoveryState,
    RecoveryValueDecision,
    content_hash,
)
from .storage import atomic_write_json


EXPERIMENT_PLAN_SCHEMA_VERSION = "1.0"


class ExperimentPlanError(ValueError):
    pass


class ExperimentPlanStatus(str, Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    BLOCKED = "BLOCKED"


CONCLUSION_STATUSES = frozenset({
    ExperimentPlanStatus.SUPPORTED,
    ExperimentPlanStatus.REFUTED,
    ExperimentPlanStatus.INCONCLUSIVE,
})
TERMINAL_STATUSES = CONCLUSION_STATUSES | {ExperimentPlanStatus.BLOCKED}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentPlanError(f"{field_name} 必须是非空字符串")
    return value.strip()


def _optional_string(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    return _required_string(value, field_name)


def _string_list(value: Any, field_name: str, required: bool = False) -> List[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ExperimentPlanError(f"{field_name} 必须是非空字符串数组")
    if required and not value:
        raise ExperimentPlanError(f"{field_name} 不能为空")
    return [item.strip() for item in value]


def _evidence_list(
    value: Any,
    field_name: str,
    required: bool = False,
) -> List[EvidenceRef]:
    if not isinstance(value, list) or any(
        not isinstance(item, EvidenceRef) for item in value
    ):
        raise ExperimentPlanError(f"{field_name} 必须是 EvidenceRef 数组")
    if required and not value:
        raise ExperimentPlanError(f"{field_name} 不能为空")
    for item in value:
        _required_string(item.sessionid, f"{field_name}.sessionid")
        _string_list(item.step_ids, f"{field_name}.step_ids")
    return list(value)


def _normalize_hypothesis(value: str) -> str:
    return " ".join(value.strip().lower().split())


@dataclass(frozen=True)
class ExperimentPlanItem:
    item_id: str
    intent_id: str
    hypothesis: str
    research_question: str
    expected_evidence: List[str]
    falsification_conditions: List[str]
    failure_pattern_refs: List[str]
    evidence_refs: List[EvidenceRef]
    reason: str
    status: ExperimentPlanStatus = ExperimentPlanStatus.PENDING
    experience_version_id: Optional[str] = None
    experiment_id: Optional[str] = None
    conclusion_reason: str = ""
    parent_item_id: Optional[str] = None
    result_evidence_refs: List[EvidenceRef] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now)
    hypothesis_key: str = field(init=False)

    def __post_init__(self):
        for name in ("item_id", "intent_id", "hypothesis", "research_question", "reason"):
            object.__setattr__(self, name, _required_string(getattr(self, name), name))
        for name in (
            "expected_evidence",
            "falsification_conditions",
            "failure_pattern_refs",
        ):
            object.__setattr__(
                self,
                name,
                _string_list(getattr(self, name), name, required=True),
            )
        object.__setattr__(
            self,
            "evidence_refs",
            _evidence_list(self.evidence_refs, "evidence_refs", required=True),
        )
        object.__setattr__(
            self,
            "result_evidence_refs",
            _evidence_list(self.result_evidence_refs, "result_evidence_refs"),
        )
        for name in ("experience_version_id", "experiment_id", "parent_item_id"):
            object.__setattr__(self, name, _optional_string(getattr(self, name), name))
        _required_string(self.created_at, "created_at")
        if not isinstance(self.status, ExperimentPlanStatus):
            raise ExperimentPlanError("status 必须是 ExperimentPlanStatus")
        if not isinstance(self.conclusion_reason, str):
            raise ExperimentPlanError("conclusion_reason 必须是字符串")
        object.__setattr__(self, "conclusion_reason", self.conclusion_reason.strip())
        object.__setattr__(
            self,
            "hypothesis_key",
            content_hash(_normalize_hypothesis(self.hypothesis)),
        )
        self._validate_state_fields()

    def _validate_state_fields(self) -> None:
        has_experience = self.experience_version_id is not None
        has_experiment = self.experiment_id is not None
        if self.status is ExperimentPlanStatus.PENDING:
            if has_experience or has_experiment:
                raise ExperimentPlanError("PENDING 项目不能预先绑定执行中的版本或实验")
            if self.conclusion_reason or self.result_evidence_refs:
                raise ExperimentPlanError("PENDING 项目不能包含实验结论")
        elif self.status is ExperimentPlanStatus.ACTIVE:
            if not has_experience or not has_experiment:
                raise ExperimentPlanError("ACTIVE 项目必须绑定经验版本和实验 ID")
            if self.conclusion_reason or self.result_evidence_refs:
                raise ExperimentPlanError("ACTIVE 项目不能提前包含实验结论")
        elif self.status in CONCLUSION_STATUSES:
            if not has_experience or not has_experiment:
                raise ExperimentPlanError("实验结论必须保留经验版本和实验 ID")
            if not self.conclusion_reason:
                raise ExperimentPlanError("实验结论必须包含结论理由")
            if not self.result_evidence_refs:
                raise ExperimentPlanError("实验结论必须包含结果证据")
        elif self.status is ExperimentPlanStatus.BLOCKED:
            if has_experience != has_experiment:
                raise ExperimentPlanError("BLOCKED 项目的版本和实验关联必须成对存在")
            if not self.conclusion_reason:
                raise ExperimentPlanError("BLOCKED 项目必须包含阻断理由")
            if self.result_evidence_refs:
                raise ExperimentPlanError("BLOCKED 项目不能伪造实验结果证据")

    def to_dict(self) -> Dict[str, Any]:
        from .schemas import _serialize_dataclass

        return _serialize_dataclass(self)


@dataclass(frozen=True)
class ExperimentPlanChange:
    sequence: int
    item_id: str
    change_type: str
    from_status: Optional[ExperimentPlanStatus]
    to_status: ExperimentPlanStatus
    reason: str
    experience_version_id: Optional[str] = None
    experiment_id: Optional[str] = None
    occurred_at: str = field(default_factory=_utc_now)

    def __post_init__(self):
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence <= 0:
            raise ExperimentPlanError("变更 sequence 必须是正整数")
        object.__setattr__(
            self,
            "item_id",
            _required_string(self.item_id, "change.item_id"),
        )
        object.__setattr__(
            self,
            "change_type",
            _required_string(self.change_type, "change_type"),
        )
        if self.change_type not in {"ADDED", "ACTIVATED", "CONCLUDED", "BLOCKED"}:
            raise ExperimentPlanError(f"未知计划变更类型: {self.change_type}")
        if self.from_status is not None and not isinstance(
            self.from_status,
            ExperimentPlanStatus,
        ):
            raise ExperimentPlanError("from_status 必须是 ExperimentPlanStatus 或 None")
        if not isinstance(self.to_status, ExperimentPlanStatus):
            raise ExperimentPlanError("to_status 必须是 ExperimentPlanStatus")
        object.__setattr__(
            self,
            "reason",
            _required_string(self.reason, "change.reason"),
        )
        for name in ("experience_version_id", "experiment_id"):
            object.__setattr__(
                self,
                name,
                _optional_string(getattr(self, name), f"change.{name}"),
            )
        _required_string(self.occurred_at, "occurred_at")

    def to_dict(self) -> Dict[str, Any]:
        from .schemas import _serialize_dataclass

        return _serialize_dataclass(self)


class ExperimentPlan:
    def __init__(self, intent_id: str):
        self.intent_id = _required_string(intent_id, "intent_id")
        self.created_at = _utc_now()
        self._items: Dict[str, ExperimentPlanItem] = {}
        self._changes: List[ExperimentPlanChange] = []

    @classmethod
    def create(cls, context: RecoveryContext) -> "ExperimentPlan":
        if not isinstance(context, RecoveryContext):
            raise ExperimentPlanError("创建实验计划需要 RecoveryContext")
        if context.state is not RecoveryState.EXPERIENCE_EXPLORING:
            raise ExperimentPlanError("只有 EXPERIENCE_EXPLORING 状态可以创建实验计划")
        assessment = context.recovery_value
        if assessment is None or assessment.decision is not RecoveryValueDecision.START_RECOVERY:
            raise ExperimentPlanError("第二层修复价值判断必须为 START_RECOVERY")
        if assessment.intent_id != context.intent_id:
            raise ExperimentPlanError("修复价值判断与当前 intent 不一致")
        return cls(context.intent_id)

    @property
    def items(self) -> Tuple[ExperimentPlanItem, ...]:
        return tuple(self._items.values())

    @property
    def changes(self) -> Tuple[ExperimentPlanChange, ...]:
        return tuple(self._changes)

    @property
    def active_item(self) -> Optional[ExperimentPlanItem]:
        return next(
            (item for item in self._items.values() if item.status is ExperimentPlanStatus.ACTIVE),
            None,
        )

    def get_item(self, item_id: str) -> ExperimentPlanItem:
        try:
            return self._items[item_id]
        except KeyError as error:
            raise ExperimentPlanError(f"未知实验计划项: {item_id}") from error

    def add_item(
        self,
        hypothesis: str,
        research_question: str,
        expected_evidence: List[str],
        falsification_conditions: List[str],
        failure_pattern_refs: List[str],
        evidence_refs: List[EvidenceRef],
        reason: str,
        parent_item_id: Optional[str] = None,
        item_id: Optional[str] = None,
    ) -> ExperimentPlanItem:
        normalized_hypothesis = _required_string(hypothesis, "hypothesis")
        hypothesis_key = content_hash(_normalize_hypothesis(normalized_hypothesis))
        duplicates = [
            item for item in self._items.values()
            if item.hypothesis_key == hypothesis_key
        ]
        if parent_item_id is not None:
            parent = self.get_item(parent_item_id)
            if parent.status not in TERMINAL_STATUSES:
                raise ExperimentPlanError("只有已终结项目才能派生后续实验")
        for duplicate in duplicates:
            if duplicate.status in {
                ExperimentPlanStatus.PENDING,
                ExperimentPlanStatus.ACTIVE,
                ExperimentPlanStatus.REFUTED,
            }:
                raise ExperimentPlanError("待验证、活动中或已否定的相同假设不能重复加入")
            if parent_item_id is None:
                raise ExperimentPlanError("重复验证既有假设必须指定 parent_item_id")
            if parent_item_id != duplicate.item_id:
                raise ExperimentPlanError("重复假设必须直接派生自同一假设的历史项目")

        new_item_id = item_id or f"experiment-plan-{len(self._items) + 1}"
        if new_item_id in self._items:
            raise ExperimentPlanError(f"实验计划项重复: {new_item_id}")
        item = ExperimentPlanItem(
            item_id=new_item_id,
            intent_id=self.intent_id,
            hypothesis=normalized_hypothesis,
            research_question=research_question,
            expected_evidence=expected_evidence,
            falsification_conditions=falsification_conditions,
            failure_pattern_refs=failure_pattern_refs,
            evidence_refs=evidence_refs,
            reason=reason,
            parent_item_id=parent_item_id,
        )
        self._items[item.item_id] = item
        self._record(
            item=item,
            change_type="ADDED",
            from_status=None,
            to_status=ExperimentPlanStatus.PENDING,
            reason=item.reason,
        )
        return item

    def activate(
        self,
        item_id: str,
        reason: str,
        experience_version_id: str,
        experiment_id: str,
    ) -> ExperimentPlanItem:
        item = self.get_item(item_id)
        if item.status is not ExperimentPlanStatus.PENDING:
            raise ExperimentPlanError("只有 PENDING 项目可以转为 ACTIVE")
        if self.active_item is not None:
            raise ExperimentPlanError("同一时间最多只能有一个 ACTIVE 实验")
        updated = replace(
            item,
            status=ExperimentPlanStatus.ACTIVE,
            experience_version_id=_required_string(
                experience_version_id,
                "experience_version_id",
            ),
            experiment_id=_required_string(experiment_id, "experiment_id"),
        )
        self._items[item_id] = updated
        self._record(
            item=updated,
            change_type="ACTIVATED",
            from_status=item.status,
            to_status=updated.status,
            reason=reason,
        )
        return updated

    def conclude(
        self,
        item_id: str,
        outcome: ExperimentPlanStatus,
        reason: str,
        result_evidence_refs: List[EvidenceRef],
    ) -> ExperimentPlanItem:
        item = self.get_item(item_id)
        if item.status is not ExperimentPlanStatus.ACTIVE:
            raise ExperimentPlanError("只有 ACTIVE 项目可以形成实验结论")
        if not isinstance(outcome, ExperimentPlanStatus):
            try:
                outcome = ExperimentPlanStatus(outcome)
            except (TypeError, ValueError) as error:
                raise ExperimentPlanError("未知实验结论") from error
        if outcome not in CONCLUSION_STATUSES:
            raise ExperimentPlanError("实验结论只能是 SUPPORTED、REFUTED 或 INCONCLUSIVE")
        updated = replace(
            item,
            status=outcome,
            conclusion_reason=_required_string(reason, "conclusion_reason"),
            result_evidence_refs=_evidence_list(
                result_evidence_refs,
                "result_evidence_refs",
                required=True,
            ),
        )
        self._items[item_id] = updated
        self._record(
            item=updated,
            change_type="CONCLUDED",
            from_status=item.status,
            to_status=updated.status,
            reason=updated.conclusion_reason,
        )
        return updated

    def block(self, item_id: str, reason: str) -> ExperimentPlanItem:
        item = self.get_item(item_id)
        if item.status not in {ExperimentPlanStatus.PENDING, ExperimentPlanStatus.ACTIVE}:
            raise ExperimentPlanError("只有 PENDING 或 ACTIVE 项目可以被阻断")
        updated = replace(
            item,
            status=ExperimentPlanStatus.BLOCKED,
            conclusion_reason=_required_string(reason, "block.reason"),
        )
        self._items[item_id] = updated
        self._record(
            item=updated,
            change_type="BLOCKED",
            from_status=item.status,
            to_status=updated.status,
            reason=updated.conclusion_reason,
        )
        return updated

    def _record(
        self,
        item: ExperimentPlanItem,
        change_type: str,
        from_status: Optional[ExperimentPlanStatus],
        to_status: ExperimentPlanStatus,
        reason: str,
    ) -> None:
        self._changes.append(ExperimentPlanChange(
            sequence=len(self._changes) + 1,
            item_id=item.item_id,
            change_type=change_type,
            from_status=from_status,
            to_status=to_status,
            reason=reason,
            experience_version_id=item.experience_version_id,
            experiment_id=item.experiment_id,
        ))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": EXPERIMENT_PLAN_SCHEMA_VERSION,
            "intent_id": self.intent_id,
            "created_at": self.created_at,
            "items": [item.to_dict() for item in self.items],
            "changes": [change.to_dict() for change in self.changes],
        }

    def save(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentPlan":
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ExperimentPlanError(f"实验计划文件损坏: {source}") from error
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: Any) -> "ExperimentPlan":
        if not isinstance(payload, dict):
            raise ExperimentPlanError("实验计划必须是 JSON 对象")
        _exact_fields(payload, {
            "schema_version",
            "intent_id",
            "created_at",
            "items",
            "changes",
        }, "实验计划")
        if payload["schema_version"] != EXPERIMENT_PLAN_SCHEMA_VERSION:
            raise ExperimentPlanError("不支持的实验计划 Schema 版本")
        plan = cls(payload["intent_id"])
        plan.created_at = _required_string(payload["created_at"], "created_at")
        if not isinstance(payload["items"], list) or not isinstance(payload["changes"], list):
            raise ExperimentPlanError("items 和 changes 必须是数组")
        for raw_item in payload["items"]:
            item = _item_from_dict(raw_item)
            if item.intent_id != plan.intent_id:
                raise ExperimentPlanError("计划项 intent 不一致")
            if item.item_id in plan._items:
                raise ExperimentPlanError("持久化计划包含重复 item_id")
            plan._items[item.item_id] = item
        for raw_change in payload["changes"]:
            plan._changes.append(_change_from_dict(raw_change))
        plan._validate_loaded()
        return plan

    def _validate_loaded(self) -> None:
        if sum(
            item.status is ExperimentPlanStatus.ACTIVE
            for item in self._items.values()
        ) > 1:
            raise ExperimentPlanError("持久化计划包含多个 ACTIVE 实验")
        for index, change in enumerate(self._changes, start=1):
            if change.sequence != index:
                raise ExperimentPlanError("计划变更 sequence 必须连续且从 1 开始")
            if change.item_id not in self._items:
                raise ExperimentPlanError("计划变更引用了不存在的项目")
            _validate_change_transition(change)
        for item in self._items.values():
            related = [
                change for change in self._changes
                if change.item_id == item.item_id
            ]
            if not related or related[0].change_type != "ADDED":
                raise ExperimentPlanError("每个计划项必须从 ADDED 变更开始")
            for previous, current in zip(related, related[1:]):
                if current.from_status is not previous.to_status:
                    raise ExperimentPlanError("计划变更前后状态不连续")
            last = related[-1]
            if last.to_status is not item.status:
                raise ExperimentPlanError("计划项状态与最后变更不一致")
            if (
                last.experience_version_id != item.experience_version_id
                or last.experiment_id != item.experiment_id
            ):
                raise ExperimentPlanError("计划项关联 ID 与最后变更不一致")
            if item.parent_item_id is not None:
                parent = self.get_item(item.parent_item_id)
                if parent.status not in TERMINAL_STATUSES:
                    raise ExperimentPlanError("派生项目的父项目必须已经终结")


def _validate_change_transition(change: ExperimentPlanChange) -> None:
    source = change.from_status
    target = change.to_status
    valid = (
        change.change_type == "ADDED"
        and source is None
        and target is ExperimentPlanStatus.PENDING
    ) or (
        change.change_type == "ACTIVATED"
        and source is ExperimentPlanStatus.PENDING
        and target is ExperimentPlanStatus.ACTIVE
    ) or (
        change.change_type == "CONCLUDED"
        and source is ExperimentPlanStatus.ACTIVE
        and target in CONCLUSION_STATUSES
    ) or (
        change.change_type == "BLOCKED"
        and source in {ExperimentPlanStatus.PENDING, ExperimentPlanStatus.ACTIVE}
        and target is ExperimentPlanStatus.BLOCKED
    )
    if not valid:
        raise ExperimentPlanError("持久化计划包含非法状态迁移")


def _exact_fields(payload: Dict[str, Any], fields: set, label: str) -> None:
    actual = set(payload)
    if actual != fields:
        raise ExperimentPlanError(
            f"{label} 字段不匹配，缺少 {sorted(fields - actual)}，"
            f"未知 {sorted(actual - fields)}"
        )


def _item_from_dict(payload: Any) -> ExperimentPlanItem:
    if not isinstance(payload, dict):
        raise ExperimentPlanError("计划项必须是对象")
    fields = {
        "item_id",
        "intent_id",
        "hypothesis",
        "research_question",
        "expected_evidence",
        "falsification_conditions",
        "failure_pattern_refs",
        "evidence_refs",
        "reason",
        "status",
        "experience_version_id",
        "experiment_id",
        "conclusion_reason",
        "parent_item_id",
        "result_evidence_refs",
        "created_at",
        "hypothesis_key",
    }
    _exact_fields(payload, fields, "计划项")
    data = dict(payload)
    try:
        data["status"] = ExperimentPlanStatus(data["status"])
    except (TypeError, ValueError) as error:
        raise ExperimentPlanError("持久化计划项包含未知状态") from error
    data["evidence_refs"] = _evidence_from_dicts(data["evidence_refs"])
    data["result_evidence_refs"] = _evidence_from_dicts(data["result_evidence_refs"])
    expected_key = data.pop("hypothesis_key")
    item = ExperimentPlanItem(**data)
    if item.hypothesis_key != expected_key:
        raise ExperimentPlanError("计划项 hypothesis_key 与假设内容不一致")
    return item


def _change_from_dict(payload: Any) -> ExperimentPlanChange:
    if not isinstance(payload, dict):
        raise ExperimentPlanError("计划变更必须是对象")
    _exact_fields(payload, {
        "sequence",
        "item_id",
        "change_type",
        "from_status",
        "to_status",
        "reason",
        "experience_version_id",
        "experiment_id",
        "occurred_at",
    }, "计划变更")
    data = dict(payload)
    try:
        data["from_status"] = (
            None
            if data["from_status"] is None
            else ExperimentPlanStatus(data["from_status"])
        )
        data["to_status"] = ExperimentPlanStatus(data["to_status"])
    except (TypeError, ValueError) as error:
        raise ExperimentPlanError("持久化计划变更包含未知状态") from error
    return ExperimentPlanChange(**data)


def _evidence_from_dicts(payload: Any) -> List[EvidenceRef]:
    if not isinstance(payload, list):
        raise ExperimentPlanError("证据引用必须是数组")
    refs = []
    for item in payload:
        if not isinstance(item, dict):
            raise ExperimentPlanError("证据引用必须是对象")
        _exact_fields(item, {"sessionid", "step_ids"}, "证据引用")
        refs.append(EvidenceRef(sessionid=item["sessionid"], step_ids=item["step_ids"]))
    return refs
