import asyncio
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._analysis_llm import StructuredAnalysisClient, StructuredAnalysisError
from .config import OODRecoveryValueConfig
from .schemas import (
    AttemptAnalysis,
    DiagnosisReport,
    EvidenceRef,
    RecoveryValueAssessment,
    RecoveryValueDecision,
)
from .storage import atomic_write_json


RECOVERY_VALUE_PROMPT_VERSION = "recovery_value_v1"
RECOVERY_VALUE_SYSTEM_PROMPT = """你是 OOD 第二层探索性修复价值评估器。
你只根据输入的结构化基线诊断判断：通过向 GUI Agent 注入一条候选经验，是否可能引导其避错或完成任务，并值得启动后续实验。
你是主要语义判断者，但不能修改诊断统计、重新判定完成度、编造证据或生成经验正文。
策略可干预通常包括搜索与关键词、功能入口、页面导航、候选选择、状态检查、循环退出和完成验证。
登录、权限、验证码、测试数据缺失、设备或 Runner 异常不属于普通经验可直接解决的问题。
单指令明确失败可以形成低置信度假设，不要求失败跨指令一致；多指令稳定模式价值更高。
一次成功不能写成稳定能力；部分成功部分失败仍可能具有修复价值。
只有充分、多次且一致的能力缺失证据才能 STOP_UNSOLVABLE；一次搜索无结果、一次 B1 或环境阻断都不够。
混合环境阻断与策略失败、证据冲突或无法可靠判断时使用 MANUAL_REVIEW。
START_RECOVERY 时只能选择一个当前最有价值的失败模式和一个可证伪假设，不生成“大而全”的方案。

只输出 JSON 对象，不要 Markdown，字段必须且只能包含：
decision、reason、confidence、repair_potential、intervention_rationale、
primary_failure_pattern_id、primary_hypothesis、expected_stage、
falsification_conditions、evidence_refs、blocker_type。
decision 为 START_RECOVERY、NO_REPAIR_NEEDED、STOP_BLOCKED、STOP_UNSOLVABLE、STOP_RETRYABLE 或 MANUAL_REVIEW。
repair_potential 为 HIGH、MEDIUM、LOW 或 UNKNOWN；confidence 为 0 到 1。
START_RECOVERY 必须给出真实 primary_failure_pattern_id、非空 intervention_rationale 和 primary_hypothesis、
共享阶段中的 expected_stage、非空 falsification_conditions 以及真实 evidence_refs。
其他决策不需要假设的字段使用空字符串或空数组。
blocker_type 可为空字符串、LOGIN、PERMISSION、CAPTCHA、MISSING_TEST_DATA、
UNSUPPORTED_ENVIRONMENT、CAPABILITY_UNSUPPORTED、BUDGET 或 OTHER。
"""


class RecoveryValueError(ValueError):
    pass


class RecoveryValueAssessor:
    ENVIRONMENT_BLOCKERS = {
        "LOGIN",
        "PERMISSION",
        "CAPTCHA",
        "MISSING_TEST_DATA",
        "UNSUPPORTED_ENVIRONMENT",
    }
    BLOCKER_TYPES = ENVIRONMENT_BLOCKERS | {
        "",
        "CAPABILITY_UNSUPPORTED",
        "BUDGET",
        "OTHER",
    }
    LEVELS = {"HIGH", "MEDIUM", "LOW", "UNKNOWN"}

    def __init__(
        self,
        llm_agent,
        model: str,
        config: OODRecoveryValueConfig | None = None,
    ):
        self.config = config or OODRecoveryValueConfig()
        self.client = StructuredAnalysisClient(llm_agent, model, self.config)

    def assess(
        self,
        diagnosis: DiagnosisReport,
        output_path: str | Path | None = None,
    ) -> RecoveryValueAssessment:
        return asyncio.run(self.aassess(diagnosis, output_path=output_path))

    async def aassess(
        self,
        diagnosis: DiagnosisReport,
        output_path: str | Path | None = None,
    ) -> RecoveryValueAssessment:
        if not isinstance(diagnosis, DiagnosisReport):
            raise RecoveryValueError("diagnosis 必须是 DiagnosisReport")
        precheck = self._apply_prechecks(diagnosis)
        if precheck is not None:
            self._save(precheck, output_path)
            return precheck

        payload = self._input_payload(diagnosis)

        def validate(raw: Any) -> RecoveryValueAssessment:
            return self._parse_assessment(raw, diagnosis)

        try:
            assessment, _ = await self.client.request(
                RECOVERY_VALUE_SYSTEM_PROMPT,
                payload,
                validate,
            )
        except StructuredAnalysisError as error:
            assessment = self._manual_assessment(
                diagnosis,
                reason=f"第二层模型判断失败或输出持续不合法：{error}",
                findings=["LLM_ASSESSMENT_UNAVAILABLE"],
                mode="RULE_FALLBACK_AFTER_LLM_ERROR",
            )
        if assessment.confidence < self.config.min_llm_confidence:
            original = assessment.decision
            assessment.llm_decision = original
            assessment.decision = RecoveryValueDecision.MANUAL_REVIEW
            assessment.assessment_mode = "LLM_WITH_RULE_GUARDS"
            assessment.rule_findings.append("LLM_CONFIDENCE_BELOW_THRESHOLD")
            assessment.reason = (
                f"LLM 置信度 {assessment.confidence} 低于阈值 "
                f"{self.config.min_llm_confidence}，转人工复核；原判断：{assessment.reason}"
            )
        self._save(assessment, output_path)
        return assessment

    def _apply_prechecks(
        self,
        diagnosis: DiagnosisReport,
    ) -> Optional[RecoveryValueAssessment]:
        stats = diagnosis.overall_statistics
        if stats is None:
            return self._manual_assessment(
                diagnosis,
                reason="诊断缺少 overall_statistics，无法进行第二层价值判断",
                findings=["DIAGNOSIS_CONTRACT_INVALID"],
            )
        if diagnosis.evidence_quality == "CONFLICT" or stats.conflict_count > 0:
            return self._manual_assessment(
                diagnosis,
                reason="基线存在完成状态或 Verdict 冲突，需先解决数据一致性",
                findings=["BASELINE_CONFLICT"],
            )
        if stats.valid_completed_count + stats.valid_failed_count == 0:
            return self._precheck_assessment(
                diagnosis,
                RecoveryValueDecision.STOP_RETRYABLE,
                "没有有效 completed/failed，需补充基线或处理执行问题，不能宣称不可解",
                "NO_VALID_BASELINE",
            )
        if (
            stats.valid_failed_count == 0
            and diagnosis.evidence_quality == "SUFFICIENT"
            and stats.valid_completed_count == stats.planned_count
            and stats.planned_count > 0
        ):
            return self._precheck_assessment(
                diagnosis,
                RecoveryValueDecision.NO_REPAIR_NEEDED,
                "全部计划 attempt 均有效完成，不强行生成修复经验；后续仍需稳定性验证",
                "ALL_PLANNED_ATTEMPTS_COMPLETED",
                evidence=self._evidence_for_analyses(self._valid_analyses(diagnosis)),
            )
        if stats.valid_failed_count == 0:
            return self._precheck_assessment(
                diagnosis,
                RecoveryValueDecision.STOP_RETRYABLE,
                "已回收结果成功但基线尚不完整，等待剩余结果后再判断",
                "PARTIAL_BASELINE_ALL_COMPLETED",
                evidence=self._evidence_for_analyses(self._valid_analyses(diagnosis)),
            )

        failed = self._failed_analyses(diagnosis)
        if failed and len(failed) == stats.valid_failed_count:
            blockers = {
                analysis.verdict.blocker_type
                for analysis in failed
                if analysis.verdict is not None
            }
            if blockers and all(
                blocker in self.ENVIRONMENT_BLOCKERS for blocker in blockers
            ):
                blocker_type = next(iter(blockers)) if len(blockers) == 1 else "OTHER"
                return self._precheck_assessment(
                    diagnosis,
                    RecoveryValueDecision.STOP_BLOCKED,
                    "所有有效失败均由当前环境前置条件阻断，普通经验不能绕过",
                    "ENVIRONMENT_BLOCKED_BASELINE",
                    blocker_type=blocker_type,
                    evidence=self._evidence_for_analyses(failed),
                )
            if any(blocker in self.ENVIRONMENT_BLOCKERS for blocker in blockers):
                return self._manual_assessment(
                    diagnosis,
                    reason="环境阻断与策略性失败混合，需先拆分子集，不能生成大而全的经验",
                    findings=["MIXED_ENVIRONMENT_AND_STRATEGY_FAILURES"],
                    evidence=self._evidence_for_analyses(failed),
                )
        return None

    def _parse_assessment(
        self,
        payload: Any,
        diagnosis: DiagnosisReport,
    ) -> RecoveryValueAssessment:
        fields = {
            "decision",
            "reason",
            "confidence",
            "repair_potential",
            "intervention_rationale",
            "primary_failure_pattern_id",
            "primary_hypothesis",
            "expected_stage",
            "falsification_conditions",
            "evidence_refs",
            "blocker_type",
        }
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ValueError("第二层输出字段不完整或包含未知字段")
        try:
            decision = RecoveryValueDecision(payload["decision"])
        except (TypeError, ValueError) as error:
            raise ValueError("decision 枚举值非法") from error
        for name in (
            "reason",
            "repair_potential",
            "intervention_rationale",
            "primary_failure_pattern_id",
            "primary_hypothesis",
            "expected_stage",
            "blocker_type",
        ):
            if not isinstance(payload[name], str):
                raise ValueError(f"{name} 必须是字符串")
        if not payload["reason"].strip():
            raise ValueError("reason 不能为空")
        if payload["repair_potential"] not in self.LEVELS:
            raise ValueError("repair_potential 枚举值非法")
        blocker_type = payload["blocker_type"]
        if blocker_type not in self.BLOCKER_TYPES:
            raise ValueError("blocker_type 枚举值非法")
        confidence = payload["confidence"]
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError("confidence 必须是 0 到 1 之间的有限数值")
        falsification = payload["falsification_conditions"]
        if not isinstance(falsification, list) or any(
            not isinstance(item, str) or not item.strip() for item in falsification
        ):
            raise ValueError("falsification_conditions 必须是非空字符串组成的数组")
        evidence_refs = self._parse_evidence(
            payload["evidence_refs"],
            self._valid_analyses(diagnosis),
        )
        stats = diagnosis.overall_statistics

        if decision is RecoveryValueDecision.START_RECOVERY:
            if stats.valid_failed_count == 0:
                raise ValueError("没有有效失败时不能 START_RECOVERY")
            if payload["repair_potential"] not in {"HIGH", "MEDIUM"}:
                raise ValueError("START_RECOVERY 必须具有 HIGH 或 MEDIUM 修复潜力")
            for name in ("intervention_rationale", "primary_hypothesis"):
                if not payload[name].strip():
                    raise ValueError(f"START_RECOVERY 必须提供 {name}")
            pattern_ids = {pattern.pattern_id for pattern in diagnosis.failure_patterns}
            if payload["primary_failure_pattern_id"] not in pattern_ids:
                raise ValueError("primary_failure_pattern_id 不在当前诊断中")
            stages = {
                stage.stage_name for stage in diagnosis.stage_rubric.stages
            } if diagnosis.stage_rubric is not None else set()
            if payload["expected_stage"] not in stages:
                raise ValueError("expected_stage 不在共享业务阶段中")
            if not falsification or not evidence_refs:
                raise ValueError("START_RECOVERY 必须具有否定条件和证据")
        elif decision is RecoveryValueDecision.NO_REPAIR_NEEDED:
            if stats.valid_failed_count != 0:
                raise ValueError("存在有效失败时不能 NO_REPAIR_NEEDED")
        elif decision is RecoveryValueDecision.STOP_UNSOLVABLE:
            if blocker_type != "CAPABILITY_UNSUPPORTED" or not self._strong_unsolvable(diagnosis):
                raise ValueError("STOP_UNSOLVABLE 缺少多次一致的能力缺失强证据")
        elif decision is RecoveryValueDecision.STOP_BLOCKED:
            if blocker_type not in self.ENVIRONMENT_BLOCKERS:
                raise ValueError("STOP_BLOCKED 必须提供明确环境阻断类型")
            failed = self._failed_analyses(diagnosis)
            if not failed or any(
                analysis.verdict.blocker_type != blocker_type for analysis in failed
            ):
                raise ValueError("STOP_BLOCKED 与有效失败的阻断证据不一致")

        return RecoveryValueAssessment(
            intent_id=diagnosis.intent_id,
            baseline_plan_id=diagnosis.baseline_plan_id,
            decision=decision,
            reason=payload["reason"].strip(),
            evidence_refs=evidence_refs,
            confidence=float(confidence),
            repair_potential=payload["repair_potential"],
            intervention_rationale=payload["intervention_rationale"].strip(),
            primary_failure_pattern_id=payload["primary_failure_pattern_id"].strip(),
            primary_hypothesis=payload["primary_hypothesis"].strip(),
            expected_stage=payload["expected_stage"].strip(),
            falsification_conditions=list(falsification),
            blocker_type=blocker_type,
            llm_decision=decision,
            assessment_mode="LLM_WITH_RULE_GUARDS",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _input_payload(diagnosis: DiagnosisReport) -> Dict[str, Any]:
        return {
            "intent_id": diagnosis.intent_id,
            "baseline_plan_id": diagnosis.baseline_plan_id,
            "evidence_quality": diagnosis.evidence_quality,
            "overall_statistics": diagnosis.overall_statistics.to_dict(),
            "statistics_by_instruction": {
                key: value.to_dict()
                for key, value in diagnosis.statistics_by_instruction.items()
            },
            "attempt_analyses": [item.to_dict() for item in diagnosis.attempt_analyses],
            "progress_by_attempt": {
                key: value.to_dict()
                for key, value in diagnosis.progress_by_attempt.items()
            },
            "failure_patterns": [item.to_dict() for item in diagnosis.failure_patterns],
            "stage_rubric": (
                diagnosis.stage_rubric.to_dict()
                if diagnosis.stage_rubric is not None else None
            ),
            "issues": [item.to_dict() for item in diagnosis.issues],
        }

    @staticmethod
    def _valid_analyses(diagnosis: DiagnosisReport) -> List[AttemptAnalysis]:
        return [
            analysis for analysis in diagnosis.attempt_analyses
            if analysis.included_in_strategy and analysis.verdict is not None
        ]

    @staticmethod
    def _failed_analyses(diagnosis: DiagnosisReport) -> List[AttemptAnalysis]:
        return [
            analysis for analysis in RecoveryValueAssessor._valid_analyses(diagnosis)
            if analysis.verdict.task_completed is False
        ]

    @staticmethod
    def _evidence_for_analyses(analyses: List[AttemptAnalysis]) -> List[EvidenceRef]:
        return [
            EvidenceRef(
                sessionid=analysis.verdict.sessionid,
                step_ids=list(analysis.verdict.key_evidence_steps),
            )
            for analysis in analyses
        ]

    @staticmethod
    def _parse_evidence(
        raw_refs: Any,
        analyses: List[AttemptAnalysis],
    ) -> List[EvidenceRef]:
        if not isinstance(raw_refs, list):
            raise ValueError("evidence_refs 必须是数组")
        allowed: Dict[str, set] = {}
        for analysis in analyses:
            allowed.setdefault(analysis.verdict.sessionid, set()).update(
                analysis.verdict.key_evidence_steps
            )
        refs = []
        for raw in raw_refs:
            if not isinstance(raw, dict) or set(raw) != {"sessionid", "step_ids"}:
                raise ValueError("证据引用只能包含 sessionid 和 step_ids")
            sessionid = raw["sessionid"]
            steps = raw["step_ids"]
            if sessionid not in allowed:
                raise ValueError("证据 sessionid 不在有效基线分析中")
            if not isinstance(steps, list) or not steps or any(
                not isinstance(step, str) for step in steps
            ):
                raise ValueError("证据 step_ids 必须是非空字符串数组")
            unknown = set(steps) - allowed[sessionid]
            if unknown:
                raise ValueError(f"引用了未校验的证据步骤: {sorted(unknown)}")
            refs.append(EvidenceRef(sessionid=sessionid, step_ids=list(steps)))
        return refs

    @staticmethod
    def _strong_unsolvable(diagnosis: DiagnosisReport) -> bool:
        failed = RecoveryValueAssessor._failed_analyses(diagnosis)
        stats = diagnosis.overall_statistics
        matches = [
            analysis for analysis in failed
            if analysis.verdict.primary_failure_subcategory == "B1"
            and analysis.verdict.blocker_type == "CAPABILITY_UNSUPPORTED"
        ]
        return (
            stats.valid_completed_count == 0
            and stats.valid_failed_count >= 2
            and len(matches) == stats.valid_failed_count
        )

    @staticmethod
    def _precheck_assessment(
        diagnosis: DiagnosisReport,
        decision: RecoveryValueDecision,
        reason: str,
        finding: str,
        blocker_type: str = "",
        evidence: Optional[List[EvidenceRef]] = None,
    ) -> RecoveryValueAssessment:
        return RecoveryValueAssessment(
            intent_id=diagnosis.intent_id,
            baseline_plan_id=diagnosis.baseline_plan_id,
            decision=decision,
            reason=reason,
            evidence_refs=evidence or [],
            confidence=1.0,
            blocker_type=blocker_type,
            rule_findings=[finding],
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _manual_assessment(
        diagnosis: DiagnosisReport,
        reason: str,
        findings: List[str],
        mode: str = "DETERMINISTIC_PRECHECK",
        evidence: Optional[List[EvidenceRef]] = None,
    ) -> RecoveryValueAssessment:
        return RecoveryValueAssessment(
            intent_id=diagnosis.intent_id,
            baseline_plan_id=diagnosis.baseline_plan_id,
            decision=RecoveryValueDecision.MANUAL_REVIEW,
            reason=reason,
            evidence_refs=evidence or [],
            assessment_mode=mode,
            rule_findings=findings,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _save(
        assessment: RecoveryValueAssessment,
        output_path: str | Path | None,
    ) -> None:
        if output_path is not None:
            atomic_write_json(output_path, assessment.to_dict())
