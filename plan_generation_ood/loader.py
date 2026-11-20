import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schemas import (
    InconsistentRecord,
    IntentKey,
    IntentWorkItem,
    LoadedTrajectory,
    LoadOutcome,
    ValidationIssue,
    content_hash,
)


class OODLoader:
    def load(self, input_file: str | Path) -> LoadOutcome:
        entries, inconsistent = self._read_input(input_file)
        groups: Dict[IntentKey, IntentWorkItem] = {}
        seen_sessions: Dict[str, Tuple[IntentWorkItem, str]] = {}
        skipped_cluster_mismatch = 0
        duplicate_records = 0

        for line_number, raw_record in entries:
            if not isinstance(raw_record, dict):
                inconsistent.append(InconsistentRecord(
                    record=raw_record,
                    issues=[ValidationIssue(
                        code="INVALID_RECORD_TYPE",
                        message="轨迹记录必须是 JSON 对象",
                        line_number=line_number,
                    )],
                ))
                continue

            cluster_effectiveness = raw_record.get("cluster_effectiveness")
            is_match = (
                cluster_effectiveness.get("is_match")
                if isinstance(cluster_effectiveness, dict) else None
            )
            if is_match is not True:
                skipped_cluster_mismatch += 1
                continue

            key, key_issues = self._extract_intent_key(raw_record, line_number)
            if key is None:
                inconsistent.append(InconsistentRecord(
                    record=raw_record,
                    issues=key_issues,
                ))
                continue

            item = groups.setdefault(key, IntentWorkItem(key=key))
            trajectory = LoadedTrajectory(
                record=raw_record,
                content_hash=content_hash(raw_record),
                line_number=line_number,
            )
            issues, include_in_input = self._validate_record(trajectory)
            sessionid = trajectory.sessionid

            if sessionid:
                previous = seen_sessions.get(sessionid)
                if previous is not None:
                    previous_item, previous_hash = previous
                    if previous_hash == trajectory.content_hash:
                        duplicate_records += 1
                        continue

                    conflict = ValidationIssue(
                        code="CONFLICTING_SESSION_ID",
                        message="同一 sessionid 对应不同记录内容",
                        sessionid=sessionid,
                        line_number=line_number,
                    )
                    item.issues.append(conflict)
                    previous_item.issues.append(conflict)
                    inconsistent.append(InconsistentRecord(
                        record=raw_record,
                        issues=[conflict],
                        intent_id=item.intent_id,
                    ))
                    continue
                seen_sessions[sessionid] = (item, trajectory.content_hash)

            if issues:
                item.issues.extend(issues)
                inconsistent.append(InconsistentRecord(
                    record=raw_record,
                    issues=issues,
                    intent_id=item.intent_id,
                ))

            if include_in_input:
                item.records.append(trajectory)

        items = sorted(
            groups.values(),
            key=lambda item: (
                item.key.category,
                item.key.app,
                item.key.intent_name,
            ),
        )
        return LoadOutcome(
            items=items,
            inconsistent_records=inconsistent,
            total_input_records=len(entries) + self._parse_error_count(inconsistent),
            skipped_cluster_mismatch=skipped_cluster_mismatch,
            duplicate_records=duplicate_records,
        )

    @staticmethod
    def _parse_error_count(inconsistent: List[InconsistentRecord]) -> int:
        return sum(
            1
            for entry in inconsistent
            for issue in entry.issues
            if issue.code == "INVALID_JSON"
        )

    def _read_input(
        self,
        input_file: str | Path,
    ) -> Tuple[List[Tuple[int, Any]], List[InconsistentRecord]]:
        path = Path(input_file)
        text = path.read_text(encoding="utf-8")
        inconsistent: List[InconsistentRecord] = []

        try:
            data = json.loads(text)
            if not isinstance(data, list):
                raise ValueError("JSON 输入文件必须是轨迹对象数组")
            return list(enumerate(data, start=1)), inconsistent
        except json.JSONDecodeError:
            pass

        entries: List[Tuple[int, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entries.append((line_number, json.loads(line)))
            except json.JSONDecodeError as error:
                inconsistent.append(InconsistentRecord(
                    record=line,
                    issues=[ValidationIssue(
                        code="INVALID_JSON",
                        message=f"JSONL 行解析失败: {error.msg}",
                        line_number=line_number,
                    )],
                ))
        return entries, inconsistent

    @staticmethod
    def _extract_intent_key(
        record: Dict[str, Any],
        line_number: int,
    ) -> Tuple[Optional[IntentKey], List[ValidationIssue]]:
        category_obj = record.get("category")
        category = (
            category_obj.get("category")
            if isinstance(category_obj, dict) else None
        )
        app = record.get("app")
        intent_name = record.get("intent_cluster_name")

        missing = []
        if not isinstance(category, str) or not category.strip():
            missing.append("category.category")
        if not isinstance(app, str) or not app.strip():
            missing.append("app")
        if not isinstance(intent_name, str) or not intent_name.strip():
            missing.append("intent_cluster_name")

        if missing:
            return None, [ValidationIssue(
                code="MISSING_GROUP_FIELD",
                message=f"缺少意图分组字段: {', '.join(missing)}",
                sessionid=str(record.get("sessionid", "")),
                line_number=line_number,
            )]

        return IntentKey(
            category=category.strip(),
            app=app.strip(),
            intent_name=intent_name.strip(),
        ), []

    @staticmethod
    def _validate_record(
        trajectory: LoadedTrajectory,
    ) -> Tuple[List[ValidationIssue], bool]:
        record = trajectory.record
        issues: List[ValidationIssue] = []
        include_in_input = True

        def add_issue(code: str, message: str, include: bool = False):
            nonlocal include_in_input
            issues.append(ValidationIssue(
                code=code,
                message=message,
                sessionid=trajectory.sessionid,
                line_number=trajectory.line_number,
            ))
            if not include:
                include_in_input = False

        for field_name in ("sessionid", "instruction"):
            value = record.get(field_name)
            if not isinstance(value, str) or not value.strip():
                add_issue(
                    "MISSING_REQUIRED_FIELD",
                    f"必填字段 {field_name} 缺失或不是非空字符串",
                )

        step_keys = [
            key for key in record
            if isinstance(key, str) and key.startswith("step")
        ]
        valid_step_keys = [
            key for key in step_keys
            if re.fullmatch(r"step\d+", key) and isinstance(record.get(key), dict)
        ]
        if not valid_step_keys or len(valid_step_keys) != len(step_keys):
            add_issue(
                "INVALID_STEPS",
                "步骤必须使用 stepN 键且每个步骤必须是 JSON 对象",
            )

        verdict = record.get("verdict")
        if not isinstance(verdict, dict):
            add_issue("MISSING_VERDICT", "verdict 必须是 JSON 对象")
            return issues, include_in_input

        completion_status = verdict.get("completion_status")
        task_completed = verdict.get("task_completed")
        if completion_status not in {"completed", "failed", "uncertain"}:
            add_issue(
                "INVALID_VERDICT",
                "completion_status 必须为 completed、failed 或 uncertain",
            )
        if not isinstance(task_completed, bool):
            add_issue("INVALID_VERDICT", "task_completed 必须为布尔值")

        if not issues and (completion_status != "failed" or task_completed is not False):
            add_issue(
                "OOD_STATUS_CONFLICT",
                "OOD 组内只允许 completion_status=failed 且 task_completed=false",
                include=True,
            )

        return issues, include_in_input
