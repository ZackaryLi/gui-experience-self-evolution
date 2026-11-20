import math
from collections import Counter
from typing import Any, Dict, List

from .config import OODTriageConfig
from .schemas import (
    IntentWorkItem,
    LoadedTrajectory,
    TriageResult,
    TriageStatus,
)


class DeterministicTriage:
    ROOT_CAUSES = ("A", "B", "C", "D", "E", "UNKNOWN")

    def __init__(self, config: OODTriageConfig | None = None):
        self.config = config or OODTriageConfig()

    def evaluate(self, item: IntentWorkItem) -> TriageResult:
        distribution = self._build_root_distribution(item.records)
        total = len(item.records)

        if item.issues:
            return self._result(
                item,
                TriageStatus.MANUAL_REVIEW,
                "DATA_INCONSISTENT",
                "意图组存在非法记录、完成状态冲突或 sessionid 冲突，禁止自动探索",
                distribution,
                total,
            )
        if total == 0:
            return self._result(
                item,
                TriageStatus.MANUAL_REVIEW,
                "EMPTY_INTENT",
                "意图组没有通过校验的轨迹",
                distribution,
                total,
            )

        ratios = {key: distribution[key]["ratio"] for key in self.ROOT_CAUSES}
        subcategories = {
            key: distribution[key]["subcategory_counts"]
            for key in self.ROOT_CAUSES
        }

        if ratios["B"] >= self.config.b_dominant_ratio:
            return self._result(
                item,
                TriageStatus.UNSOLVABLE,
                "B_CAPABILITY_BLOCKED",
                "B 类能力不可达根因达到主导阈值，不消耗后续探索预算",
                distribution,
                total,
            )

        a2_ratio = subcategories["A"].get("A2", 0) / total
        if a2_ratio > self.config.majority_ratio:
            return self._result(
                item,
                TriageStatus.NEEDS_CLARIFICATION,
                "A2_INFORMATION_MISSING",
                "A2 信息缺失占主导，需要澄清，不允许编造用户信息",
                distribution,
                total,
            )

        a1_ratio = subcategories["A"].get("A1", 0) / total
        a3_ratio = subcategories["A"].get("A3", 0) / total
        if a1_ratio >= self.config.instruction_issue_ratio or a3_ratio >= self.config.instruction_issue_ratio:
            is_multi_task = a3_ratio > a1_ratio
            return self._result(
                item,
                TriageStatus.REWRITING,
                "A3_MULTI_TASK" if is_multi_task else "A1_AMBIGUOUS_INSTRUCTION",
                "A3 多任务问题占主导，需要拆分" if is_multi_task else "A1 语义模糊占主导，需要语义保持改写",
                distribution,
                total,
            )

        for subcategory, rule_id, message in (
            ("A4", "A4_CONTENT_GENERATION", "A4 内容生成或槽位问题占主导，转人工审核"),
            ("A5", "A5_INTENT_MISUNDERSTANDING", "A5 意图理解问题占主导，转人工审核"),
        ):
            if subcategories["A"].get(subcategory, 0) / total > self.config.majority_ratio:
                return self._result(
                    item,
                    TriageStatus.MANUAL_REVIEW,
                    rule_id,
                    message,
                    distribution,
                    total,
                )

        if ratios["D"] > self.config.d_dominant_ratio:
            candidates = [
                record for record in item.records
                if record.primary_parent == "D"
            ]
            return self._result(
                item,
                TriageStatus.EXPLORABLE,
                "D_DOMINANT",
                "D 类执行规划问题占绝对多数，仅将 D 类主根因轨迹交给后续诊断",
                distribution,
                total,
                candidates,
            )

        if ratios["C"] > self.config.majority_ratio:
            return self._result(
                item,
                TriageStatus.MANUAL_REVIEW,
                "C_DOMINANT",
                "C 类界面感知问题占主导，当前阶段不进入 D 类执行规划探索",
                distribution,
                total,
            )
        if ratios["E"] > self.config.majority_ratio:
            return self._result(
                item,
                TriageStatus.MANUAL_REVIEW,
                "E_DOMINANT",
                "E 类跨应用问题占主导，当前阶段不进入 D 类执行规划探索",
                distribution,
                total,
            )
        if ratios["UNKNOWN"] > self.config.majority_ratio:
            return self._result(
                item,
                TriageStatus.MANUAL_REVIEW,
                "UNKNOWN_DOMINANT",
                "未知根因占主导，不消耗真机探索预算",
                distribution,
                total,
            )

        return self._result(
            item,
            TriageStatus.MANUAL_REVIEW,
            "D_NOT_DOMINANT",
            "D 类执行规划问题未达到绝对多数，混合根因意图转人工审核",
            distribution,
            total,
        )

    def _result(
        self,
        item: IntentWorkItem,
        status: TriageStatus,
        rule_id: str,
        reason: str,
        distribution: Dict[str, Any],
        total: int,
        candidates: List[LoadedTrajectory] | None = None,
    ) -> TriageResult:
        return TriageResult(
            key=item.key,
            status=status,
            rule_id=rule_id,
            reason=reason,
            total_records=total,
            root_distribution=distribution,
            thresholds=self.config.to_dict(),
            issues=list(item.issues),
            candidate_records=candidates or [],
        )

    def _build_root_distribution(
        self,
        records: List[LoadedTrajectory],
    ) -> Dict[str, Any]:
        total = len(records)
        grouped: Dict[str, List[LoadedTrajectory]] = {
            key: [] for key in self.ROOT_CAUSES
        }
        for record in records:
            grouped[record.primary_parent].append(record)

        distribution: Dict[str, Any] = {}
        for root_cause, root_records in grouped.items():
            confidence_sum = 0.0
            subcategories = Counter()
            for record in root_records:
                verdict = record.record.get("verdict", {})
                confidence = verdict.get("confidence") if isinstance(verdict, dict) else None
                if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
                    if math.isfinite(confidence) and 0.0 <= confidence <= 1.0:
                        confidence_sum += float(confidence)
                if record.primary_subcategory:
                    subcategories[record.primary_subcategory] += 1

            distribution[root_cause] = {
                "count": len(root_records),
                "ratio": len(root_records) / total if total else 0.0,
                "confidence_sum": round(confidence_sum, 6),
                "subcategory_counts": dict(sorted(subcategories.items())),
                "evidence_session_ids": [record.sessionid for record in root_records],
            }
        return distribution
