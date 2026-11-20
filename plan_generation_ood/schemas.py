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
