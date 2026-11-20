import math
from datetime import datetime, timezone
from typing import Any, List, Optional

from .baseline import BaselinePlanner, normalize_instruction
from .config import OODBaselineConfig, OODBatchValueConfig
from .llm import LLMAgent, extract_json
from .prompts import BATCH_VALUE_PROMPT_VERSION, build_batch_value_prompt
from .schemas import (
    BatchRunConstraints,
    BatchRunDecision,
    BatchRunValueAssessment,
    EvidenceRef,
    IntentKey,
    LoadedTrajectory,
    content_hash,
)


class BatchRunValueAssessor:
    LEVELS = {"HIGH", "MEDIUM", "LOW", "UNKNOWN"}
    BLOCKER_TYPES = {
        "",
        "LOGIN",
        "PERMISSION",
        "CAPTCHA",
        "MISSING_TEST_DATA",
        "UNSUPPORTED_ENVIRONMENT",
        "AUTHORIZATION",
        "SAFETY",
        "CAPABILITY_UNSUPPORTED",
        "BUDGET",
        "OTHER",
    }

    def __init__(
        self,
        llm: LLMAgent,
        config: OODBatchValueConfig | None = None,
        baseline_config: OODBaselineConfig | None = None,
    ):
        self.llm = llm
        self.config = config or OODBatchValueConfig()
        self.baseline_config = baseline_config or OODBaselineConfig()
        self.planner = BaselinePlanner(self.baseline_config)

    async def assess(
        self,
        key: IntentKey,
        records: List[LoadedTrajectory],
        constraints: BatchRunConstraints | None = None,
    ) -> BatchRunValueAssessment:
        constraints = constraints or BatchRunConstraints(
            max_attempts=self.config.max_total_attempts
        )
        planned_attempts = self.planner.planned_attempt_count(records)
        unique_count = len({
            normalize_instruction(str(record.record.get("instruction", "")))
            for record in records
        })
        fingerprint = self._source_fingerprint(records)

        rule_result = self._apply_pre_llm_rules(
            key=key,
            records=records,
            constraints=constraints,
            planned_attempts=planned_attempts,
            unique_count=unique_count,
            fingerprint=fingerprint,
        )
        if rule_result is not None:
            return rule_result

        previous_error = ""
        last_error = ""
        for _ in range(self.config.max_llm_attempts):
            prompt = build_batch_value_prompt(
                key=key,
                records=records,
                planned_attempt_count=planned_attempts,
                constraints=constraints,
                previous_error=previous_error,
            )
            try:
                raw_response = await self.llm.ainvoke(prompt, temperature=0.1)
                if isinstance(raw_response, tuple):
                    raw_response = raw_response[0]
                payload = extract_json(raw_response or "", default={})
                assessment = self._parse_assessment(
                    payload=payload,
                    key=key,
                    records=records,
                    planned_attempts=planned_attempts,
                    unique_count=unique_count,
                    fingerprint=fingerprint,
                )
                if assessment.confidence < self.config.min_llm_confidence:
                    llm_decision = assessment.decision
                    assessment.llm_decision = llm_decision
                    assessment.decision = BatchRunDecision.MANUAL_REVIEW
                    assessment.rule_findings.append("LLM_CONFIDENCE_BELOW_THRESHOLD")
                    assessment.reason = (
                        f"LLM 置信度 {assessment.confidence} 低于阈值 "
                        f"{self.config.min_llm_confidence}，转人工复核；原判断：{assessment.reason}"
                    )
                return assessment
            except Exception as error:
                last_error = str(error)
                previous_error = last_error

        return BatchRunValueAssessment(
            intent_id=key.intent_id,
            intent_key=key,
            decision=BatchRunDecision.MANUAL_REVIEW,
            reason=f"LLM 批跑价值判断失败或输出持续不合法: {last_error}",
            information_value="UNKNOWN",
            repair_potential="UNKNOWN",
            confidence=0.0,
            evidence_refs=[],
            candidate_count=len(records),
            unique_instruction_count=unique_count,
            planned_attempt_count=planned_attempts,
            source_fingerprint=fingerprint,
            assessment_mode="RULE_FALLBACK_AFTER_LLM_ERROR",
            rule_findings=["LLM_ASSESSMENT_UNAVAILABLE"],
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def _apply_pre_llm_rules(
        self,
        key: IntentKey,
        records: List[LoadedTrajectory],
        constraints: BatchRunConstraints,
        planned_attempts: int,
        unique_count: int,
        fingerprint: str,
    ) -> Optional[BatchRunValueAssessment]:
        finding = ""
        decision = BatchRunDecision.MANUAL_REVIEW
        reason = ""
        blocker_type = ""

        if not records or planned_attempts == 0:
            finding = "EMPTY_OR_INVALID_CANDIDATES"
            reason = "候选轨迹为空或缺少有效指令，无法制定基线计划"
        elif any(record.primary_parent != "D" for record in records):
            finding = "NON_D_CANDIDATE"
            reason = "批跑价值输入包含非 D 类主根因轨迹，禁止自动选择候选"
        elif not constraints.execution_authorized:
            finding = "EXECUTION_NOT_AUTHORIZED"
            decision = BatchRunDecision.STOP_BLOCKED
            reason = "任务未获得执行授权"
            blocker_type = "AUTHORIZATION"
        elif not constraints.safety_cleared:
            finding = "SAFETY_NOT_CLEARED"
            decision = BatchRunDecision.STOP_BLOCKED
            reason = "任务尚未通过安全检查"
            blocker_type = "SAFETY"
        elif not constraints.environment_ready:
            finding = "ENVIRONMENT_NOT_READY"
            decision = BatchRunDecision.STOP_BLOCKED
            reason = "执行环境或必要前置条件尚未就绪"
            blocker_type = "UNSUPPORTED_ENVIRONMENT"
        elif constraints.has_reusable_baseline:
            finding = "REUSABLE_BASELINE_EXISTS"
            decision = BatchRunDecision.SKIP_NO_BATCH_VALUE
            reason = "已有可复用基线，本次重复批跑没有新增信息收益"
        elif constraints.max_attempts is not None and planned_attempts > constraints.max_attempts:
            finding = "BASELINE_BUDGET_EXCEEDED"
            reason = (
                f"计划需要 {planned_attempts} 个 attempt，超过当前上限 "
                f"{constraints.max_attempts}"
            )
            blocker_type = "BUDGET"

        if not finding:
            return None
        return BatchRunValueAssessment(
            intent_id=key.intent_id,
            intent_key=key,
            decision=decision,
            reason=reason,
            information_value="LOW" if decision != BatchRunDecision.MANUAL_REVIEW else "UNKNOWN",
            repair_potential="UNKNOWN",
            confidence=1.0,
            evidence_refs=[],
            candidate_count=len(records),
            unique_instruction_count=unique_count,
            planned_attempt_count=planned_attempts,
            source_fingerprint=fingerprint,
            blocker_type=blocker_type,
            assessment_mode="DETERMINISTIC_PRECHECK",
            rule_findings=[finding],
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def _parse_assessment(
        self,
        payload: Any,
        key: IntentKey,
        records: List[LoadedTrajectory],
        planned_attempts: int,
        unique_count: int,
        fingerprint: str,
    ) -> BatchRunValueAssessment:
        if not isinstance(payload, dict):
            raise ValueError("LLM 输出必须是 JSON 对象")
        decision = BatchRunDecision(str(payload.get("decision", "")))
        reason = payload.get("reason")
        information_value = str(payload.get("information_value", ""))
        repair_potential = str(payload.get("repair_potential", ""))
        confidence = payload.get("confidence")
        blocker_type = str(payload.get("blocker_type", ""))

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason 必须是非空字符串")
        if information_value not in self.LEVELS or repair_potential not in self.LEVELS:
            raise ValueError("information_value/repair_potential 枚举值非法")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError("confidence 必须是 0 到 1 之间的有限数值")
        if blocker_type not in self.BLOCKER_TYPES:
            raise ValueError("blocker_type 枚举值非法")
        if decision is BatchRunDecision.RUN_BASELINE and information_value == "LOW":
            raise ValueError("RUN_BASELINE 与 LOW 信息价值矛盾")
        if decision is BatchRunDecision.STOP_BLOCKED and not blocker_type:
            raise ValueError("STOP_BLOCKED 必须提供 blocker_type")
        if (
            decision is BatchRunDecision.STOP_UNSOLVABLE
            and blocker_type != "CAPABILITY_UNSUPPORTED"
        ):
            raise ValueError("STOP_UNSOLVABLE 必须明确 CAPABILITY_UNSUPPORTED")

        evidence_refs = self._parse_evidence(payload.get("evidence_refs"), records)
        return BatchRunValueAssessment(
            intent_id=key.intent_id,
            intent_key=key,
            decision=decision,
            reason=reason.strip(),
            information_value=information_value,
            repair_potential=repair_potential,
            confidence=float(confidence),
            evidence_refs=evidence_refs,
            candidate_count=len(records),
            unique_instruction_count=unique_count,
            planned_attempt_count=planned_attempts,
            source_fingerprint=fingerprint,
            blocker_type=blocker_type,
            llm_decision=decision,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _parse_evidence(
        raw_refs: Any,
        records: List[LoadedTrajectory],
    ) -> List[EvidenceRef]:
        if not isinstance(raw_refs, list) or not raw_refs:
            raise ValueError("evidence_refs 必须是非空数组")
        by_session = {record.sessionid: record for record in records}
        refs = []
        seen = set()
        for raw_ref in raw_refs:
            if not isinstance(raw_ref, dict):
                raise ValueError("evidence_refs 元素必须是对象")
            sessionid = raw_ref.get("sessionid")
            step_ids = raw_ref.get("step_ids", [])
            if sessionid not in by_session:
                raise ValueError(f"引用了候选集中不存在的 sessionid: {sessionid}")
            if not isinstance(step_ids, list) or any(
                not isinstance(step_id, str) for step_id in step_ids
            ):
                raise ValueError("step_ids 必须是字符串数组")
            available_steps = set(by_session[sessionid].record)
            unknown_steps = [step_id for step_id in step_ids if step_id not in available_steps]
            if unknown_steps:
                raise ValueError(f"引用了不存在的步骤: {unknown_steps}")
            identity = (sessionid, tuple(sorted(step_ids)))
            if identity not in seen:
                refs.append(EvidenceRef(sessionid=sessionid, step_ids=list(step_ids)))
                seen.add(identity)
        return refs

    @staticmethod
    def _source_fingerprint(records: List[LoadedTrajectory]) -> str:
        return content_hash([
            {
                "sessionid": record.sessionid,
                "content_hash": record.content_hash,
            }
            for record in sorted(records, key=lambda item: item.sessionid)
        ])
