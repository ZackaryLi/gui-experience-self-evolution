import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ._analysis_llm import StructuredAnalysisClient, StructuredAnalysisError
from .config import OODInitialExperienceConfig
from .schemas import (
    DiagnosisReport,
    DictSerializable,
    EvidenceRef,
    ExperienceDraft,
    FailurePattern,
    RecoveryValueAssessment,
    RecoveryValueDecision,
)
from .skills import SkillLoader, SkillLoadError
from .storage import atomic_write_json


INITIAL_EXPERIENCE_PROMPT_VERSION = "initial_experience_v1"
INITIAL_EXPERIENCE_BASE_PROMPT = """你是 OOD 固定第一条候选经验生成器。
你只根据第二层已经选定的假设生成一条候选经验，不重新选择根因，不调用工具，不执行真机。
不得修改诊断、指定经验 ID 或版本、创建实验、宣布经验有效或任务成功。
以下 Skill 是当前任务的程序性规范，必须遵守。
只输出 JSON 对象，不要 Markdown，字段必须且只能包含：
hypothesis、applicable_scope、trigger_conditions、strategy、expected_stage、
evidence_refs、forbidden_behaviors、exit_conditions。
hypothesis 和 expected_stage 必须原样复制第二层结论；evidence_refs 只能引用输入白名单中的轨迹步骤。
"""


class InitialExperienceError(ValueError):
    pass


@dataclass(frozen=True)
class InitialExperienceResult(DictSerializable):
    experience: ExperienceDraft
    source_failure_pattern_id: str
    llm_calls: int
    prompt_version: str
    skill_name: str
    skill_version: str
    skill_content_hash: str


class InitialExperienceGenerator:
    def __init__(
        self,
        llm_agent,
        model: str,
        config: OODInitialExperienceConfig | None = None,
        skill_loader: Optional[SkillLoader] = None,
    ):
        if skill_loader is not None and not isinstance(skill_loader, SkillLoader):
            raise ValueError("skill_loader 必须是 SkillLoader 或 None")
        self.config = config or OODInitialExperienceConfig()
        self.client = StructuredAnalysisClient(llm_agent, model, self.config)
        self.skill_loader = skill_loader or SkillLoader()

    def generate(
        self,
        diagnosis: DiagnosisReport,
        assessment: RecoveryValueAssessment,
        output_path: str | Path | None = None,
    ) -> InitialExperienceResult:
        return asyncio.run(self.agenerate(
            diagnosis=diagnosis,
            assessment=assessment,
            output_path=output_path,
        ))

    async def agenerate(
        self,
        diagnosis: DiagnosisReport,
        assessment: RecoveryValueAssessment,
        output_path: str | Path | None = None,
    ) -> InitialExperienceResult:
        pattern, allowed_evidence = self._validate_inputs(diagnosis, assessment)
        try:
            document = self.skill_loader.load("initial_experience")
        except SkillLoadError as error:
            raise InitialExperienceError(f"第一条经验 Skill 不可用：{error}") from error
        payload = {
            "diagnosis": self._diagnosis_payload(diagnosis),
            "recovery_value": assessment.to_dict(),
            "selected_failure_pattern": pattern.to_dict(),
            "allowed_evidence_refs": [
                EvidenceRef(sessionid=sessionid, step_ids=sorted(steps)).to_dict()
                for sessionid, steps in sorted(allowed_evidence.items())
            ],
        }

        def validate(raw: Any) -> Dict[str, Any]:
            return self._validate_payload(raw, assessment, allowed_evidence)

        try:
            validated, llm_calls = await self.client.request(
                INITIAL_EXPERIENCE_BASE_PROMPT + "\n\n" + document.content,
                payload,
                validate,
            )
        except StructuredAnalysisError as error:
            raise InitialExperienceError(
                f"第一条候选经验生成失败：{error}"
            ) from error

        experience = ExperienceDraft(
            experience_id=f"experience-{uuid.uuid4().hex[:16]}",
            intent_id=diagnosis.intent_id,
            version=1,
            parent_experience_id=None,
            **validated,
        )
        result = InitialExperienceResult(
            experience=experience,
            source_failure_pattern_id=pattern.pattern_id,
            llm_calls=llm_calls,
            prompt_version=INITIAL_EXPERIENCE_PROMPT_VERSION,
            skill_name=document.metadata.name,
            skill_version=document.metadata.version,
            skill_content_hash=document.content_hash,
        )
        if output_path is not None:
            atomic_write_json(output_path, result.to_dict())
        return result

    @staticmethod
    def _validate_inputs(
        diagnosis: DiagnosisReport,
        assessment: RecoveryValueAssessment,
    ) -> Tuple[FailurePattern, Dict[str, set]]:
        if not isinstance(diagnosis, DiagnosisReport):
            raise InitialExperienceError("diagnosis 必须是 DiagnosisReport")
        if not isinstance(assessment, RecoveryValueAssessment):
            raise InitialExperienceError("assessment 必须是 RecoveryValueAssessment")
        if assessment.decision is not RecoveryValueDecision.START_RECOVERY:
            raise InitialExperienceError("只有 START_RECOVERY 才能生成第一条经验")
        if (
            assessment.intent_id != diagnosis.intent_id
            or assessment.baseline_plan_id != diagnosis.baseline_plan_id
        ):
            raise InitialExperienceError("第二层结论与诊断身份不一致")
        pattern = next((
            item for item in diagnosis.failure_patterns
            if item.pattern_id == assessment.primary_failure_pattern_id
        ), None)
        if pattern is None:
            raise InitialExperienceError("主要失败模式不在当前诊断中")
        if not assessment.primary_hypothesis.strip():
            raise InitialExperienceError("第一条经验缺少明确假设")
        stages = {
            stage.stage_name for stage in diagnosis.stage_rubric.stages
        } if diagnosis.stage_rubric is not None else set()
        if assessment.expected_stage not in stages:
            raise InitialExperienceError("预期阶段不在共享业务阶段中")
        if not assessment.falsification_conditions:
            raise InitialExperienceError("第一条经验缺少假设否定条件")

        available: Dict[str, set] = {}
        for analysis in diagnosis.attempt_analyses:
            if not analysis.included_in_strategy or analysis.verdict is None:
                continue
            available.setdefault(analysis.verdict.sessionid, set()).update(
                analysis.verdict.key_evidence_steps
            )
        allowed: Dict[str, set] = {}
        for ref in list(assessment.evidence_refs) + list(pattern.evidence_refs):
            if ref.sessionid not in available:
                raise InitialExperienceError("第二层结论引用了无效证据 sessionid")
            unknown = set(ref.step_ids) - available[ref.sessionid]
            if unknown:
                raise InitialExperienceError(
                    f"第二层结论引用了无效证据步骤: {sorted(unknown)}"
                )
            allowed.setdefault(ref.sessionid, set()).update(ref.step_ids)
        if not allowed:
            raise InitialExperienceError("第一条经验缺少可用证据白名单")
        return pattern, allowed

    @staticmethod
    def _diagnosis_payload(diagnosis: DiagnosisReport) -> Dict[str, Any]:
        return {
            "intent_id": diagnosis.intent_id,
            "baseline_plan_id": diagnosis.baseline_plan_id,
            "evidence_quality": diagnosis.evidence_quality,
            "overall_statistics": diagnosis.overall_statistics.to_dict(),
            "statistics_by_instruction": {
                key: value.to_dict()
                for key, value in diagnosis.statistics_by_instruction.items()
            },
            "progress_by_attempt": {
                key: value.to_dict()
                for key, value in diagnosis.progress_by_attempt.items()
            },
            "failure_patterns": [item.to_dict() for item in diagnosis.failure_patterns],
            "stage_rubric": (
                diagnosis.stage_rubric.to_dict()
                if diagnosis.stage_rubric is not None else None
            ),
        }

    @staticmethod
    def _validate_payload(
        payload: Any,
        assessment: RecoveryValueAssessment,
        allowed_evidence: Dict[str, set],
    ) -> Dict[str, Any]:
        fields = {
            "hypothesis",
            "applicable_scope",
            "trigger_conditions",
            "strategy",
            "expected_stage",
            "evidence_refs",
            "forbidden_behaviors",
            "exit_conditions",
        }
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ValueError("候选经验字段不完整或包含未知字段")
        for name in (
            "hypothesis",
            "applicable_scope",
            "strategy",
            "expected_stage",
        ):
            if not isinstance(payload[name], str) or not payload[name].strip():
                raise ValueError(f"{name} 必须是非空字符串")
        if payload["hypothesis"].strip() != assessment.primary_hypothesis:
            raise ValueError("候选经验假设与第二层选定假设不一致")
        if payload["expected_stage"].strip() != assessment.expected_stage:
            raise ValueError("候选经验预期阶段与第二层结论不一致")
        for name in ("trigger_conditions", "forbidden_behaviors", "exit_conditions"):
            if not isinstance(payload[name], list) or any(
                not isinstance(item, str) or not item.strip() for item in payload[name]
            ):
                raise ValueError(f"{name} 必须是非空字符串组成的数组")
        if not payload["trigger_conditions"] or not payload["exit_conditions"]:
            raise ValueError("trigger_conditions 和 exit_conditions 不能为空")
        evidence_refs = InitialExperienceGenerator._parse_evidence(
            payload["evidence_refs"],
            allowed_evidence,
        )
        return {
            "hypothesis": assessment.primary_hypothesis,
            "applicable_scope": payload["applicable_scope"].strip(),
            "trigger_conditions": list(payload["trigger_conditions"]),
            "strategy": payload["strategy"].strip(),
            "expected_stage": assessment.expected_stage,
            "evidence_refs": evidence_refs,
            "forbidden_behaviors": list(payload["forbidden_behaviors"]),
            "exit_conditions": list(payload["exit_conditions"]),
        }

    @staticmethod
    def _parse_evidence(
        raw_refs: Any,
        allowed: Dict[str, set],
    ) -> List[EvidenceRef]:
        if not isinstance(raw_refs, list) or not raw_refs:
            raise ValueError("evidence_refs 必须是非空数组")
        refs = []
        for raw in raw_refs:
            if not isinstance(raw, dict) or set(raw) != {"sessionid", "step_ids"}:
                raise ValueError("证据引用只能包含 sessionid 和 step_ids")
            sessionid = raw["sessionid"]
            steps = raw["step_ids"]
            if sessionid not in allowed:
                raise ValueError("候选经验引用了白名单之外的 sessionid")
            if not isinstance(steps, list) or not steps or any(
                not isinstance(step, str) for step in steps
            ):
                raise ValueError("候选经验 step_ids 必须是非空字符串数组")
            unknown = set(steps) - allowed[sessionid]
            if unknown:
                raise ValueError(f"候选经验引用了白名单之外的步骤: {sorted(unknown)}")
            refs.append(EvidenceRef(sessionid=sessionid, step_ids=list(steps)))
        return refs
