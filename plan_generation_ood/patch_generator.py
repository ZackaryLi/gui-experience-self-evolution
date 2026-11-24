import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .config import OODPatchGeneratorConfig
from .experiment_plan import ExperimentPlan, ExperimentPlanStatus
from .llm import extract_json
from .recovery_schemas import RecoveryDecision
from .schemas import (
    AttemptResult,
    AttemptStatus,
    DiagnosisReport,
    DictSerializable,
    EvidenceRef,
    ExperienceDraft,
)
from .skills import SkillLoader

if TYPE_CHECKING:
    from openai import AsyncOpenAI


PATCH_SYSTEM_PROMPT = """你是 OOD Recovery 内部的经验起草器，不负责选择探索方向或判断实验成功。
围绕输入中的当前活动假设，只起草一个候选经验版本。
只输出 JSON 对象，不要输出 Markdown 或额外解释。
JSON 必须且只能包含：
hypothesis、applicable_scope、trigger_conditions、strategy、expected_stage、
evidence_refs、forbidden_behaviors、exit_conditions、change_summary。
hypothesis 必须原样复制当前活动假设；evidence_refs 只能引用输入中存在的轨迹和步骤。
REVISE_EXPERIENCE 应保留已有经验中仍有效的部分，只针对最新失败锚点增量修改。
PROPOSE_EXPERIENCE 只能用于原方向已被否定后的新方向，不得重复已否定假设。
不得把设备、Runner、超时或结果不确定解释为经验无效，不得编造页面、功能或证据。
"""

PATCH_BASE_SYSTEM_PROMPT = """你是 OOD Recovery 内部的经验起草器，不负责选择探索方向或判断实验成功。
不得调用工具、执行真机、修改实验历史或宣布成功。
以下 Skill 是当前任务的程序性规范，必须遵守其中的分析和输出要求。
"""


class PatchGenerationError(ValueError):
    pass


@dataclass(frozen=True)
class PatchGenerationResult(DictSerializable):
    experience: ExperienceDraft
    change_summary: str
    mode: RecoveryDecision
    plan_item_id: str
    parent_experience_id: str
    llm_calls: int
    skill_name: Optional[str] = None
    skill_version: Optional[str] = None
    skill_content_hash: Optional[str] = None


class PatchGenerator:
    def __init__(
        self,
        llm_agent: "AsyncOpenAI",
        model: str,
        config: OODPatchGeneratorConfig | None = None,
        skill_loader: Optional[SkillLoader] = None,
    ):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model 必须是非空字符串")
        if skill_loader is not None and not isinstance(skill_loader, SkillLoader):
            raise ValueError("skill_loader 必须是 SkillLoader 或 None")
        self.llm = llm_agent
        self.model = model.strip()
        self.config = config or OODPatchGeneratorConfig()
        self.skill_loader = skill_loader

    def generate(
        self,
        plan: ExperimentPlan,
        diagnosis: DiagnosisReport,
        current_experience: ExperienceDraft,
        experiment_feedback: List[AttemptResult],
        mode: RecoveryDecision,
        version_diff: Optional[Dict[str, Any]] = None,
    ) -> PatchGenerationResult:
        return asyncio.run(self.agenerate(
            plan=plan,
            diagnosis=diagnosis,
            current_experience=current_experience,
            experiment_feedback=experiment_feedback,
            mode=mode,
            version_diff=version_diff,
        ))

    async def agenerate(
        self,
        plan: ExperimentPlan,
        diagnosis: DiagnosisReport,
        current_experience: ExperienceDraft,
        experiment_feedback: List[AttemptResult],
        mode: RecoveryDecision,
        version_diff: Optional[Dict[str, Any]] = None,
    ) -> PatchGenerationResult:
        active_item, feedback = self._validate_inputs(
            plan,
            diagnosis,
            current_experience,
            experiment_feedback,
            mode,
            version_diff,
        )
        allowed_evidence = self._allowed_evidence(
            plan,
            diagnosis,
            current_experience,
            feedback,
        )
        refuted = [
            {
                "item_id": item.item_id,
                "hypothesis": item.hypothesis,
                "conclusion_reason": item.conclusion_reason,
                "result_evidence_refs": [ref.to_dict() for ref in item.result_evidence_refs],
            }
            for item in plan.items
            if item.status is ExperimentPlanStatus.REFUTED
        ]
        input_payload = {
            "mode": mode.value,
            "intent_id": plan.intent_id,
            "active_hypothesis": active_item.to_dict(),
            "diagnosis": diagnosis.to_dict(),
            "current_experience": current_experience.to_dict(),
            "version_diff": version_diff or {},
            "relevant_feedback": [result.to_dict() for result in feedback],
            "refuted_hypotheses": refuted,
        }
        skill_document = (
            self.skill_loader.load("experience_revision")
            if self.skill_loader is not None
            else None
        )
        system_prompt = (
            PATCH_BASE_SYSTEM_PROMPT + "\n\n" + skill_document.content
            if skill_document is not None
            else PATCH_SYSTEM_PROMPT
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(input_payload, ensure_ascii=False, default=str),
            },
        ]
        llm_calls = 0
        last_error = ""
        for _ in range(self.config.max_invalid_responses):
            response = await self._request(messages)
            llm_calls += 1
            content = ""
            try:
                if response.get("role", "assistant") != "assistant":
                    raise ValueError("响应 role 必须是 assistant")
                if response.get("tool_calls"):
                    raise ValueError("PatchGenerator 不允许工具调用")
                content = response.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("候选经验必须是 assistant content 中的非空 JSON")
                payload = extract_json(content, default={})
                validated, change_summary = self._validate_payload(
                    payload,
                    active_item.hypothesis,
                    allowed_evidence,
                )
            except (TypeError, ValueError) as error:
                last_error = str(error)
                messages.append({"role": "assistant", "content": content or ""})
                messages.append({
                    "role": "user",
                    "content": f"上一轮候选经验未通过校验，请只返回修正后的完整 JSON：{last_error}",
                })
                continue

            experience = ExperienceDraft(
                experience_id=self._new_experience_id(current_experience),
                intent_id=plan.intent_id,
                version=current_experience.version + 1,
                parent_experience_id=current_experience.experience_id,
                **validated,
            )
            return PatchGenerationResult(
                experience=experience,
                change_summary=change_summary,
                mode=mode,
                plan_item_id=active_item.item_id,
                parent_experience_id=current_experience.experience_id,
                llm_calls=llm_calls,
                skill_name=skill_document.metadata.name if skill_document else None,
                skill_version=skill_document.metadata.version if skill_document else None,
                skill_content_hash=skill_document.content_hash if skill_document else None,
            )
        raise PatchGenerationError(
            f"候选经验连续未通过校验，已停止生成：{last_error}"
        )

    async def _request(self, messages) -> Dict[str, Any]:
        try:
            completion = await self.llm.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.config.temperature,
            )
            message = completion.choices[0].message
            if hasattr(message, "model_dump"):
                return message.model_dump(exclude_none=True)
            if isinstance(message, dict):
                return message
            raise ValueError("LLM 响应不是 OpenAI assistant message 对象")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise PatchGenerationError(
                f"PatchGenerator 模型调用失败：{type(error).__name__}"
            ) from error

    def _validate_inputs(
        self,
        plan: ExperimentPlan,
        diagnosis: DiagnosisReport,
        current_experience: ExperienceDraft,
        experiment_feedback: List[AttemptResult],
        mode: RecoveryDecision,
        version_diff: Optional[Dict[str, Any]],
    ) -> Tuple[Any, List[AttemptResult]]:
        if not isinstance(plan, ExperimentPlan):
            raise PatchGenerationError("plan 必须是 ExperimentPlan")
        if not isinstance(diagnosis, DiagnosisReport):
            raise PatchGenerationError("diagnosis 必须是 DiagnosisReport")
        if not isinstance(current_experience, ExperienceDraft):
            raise PatchGenerationError("current_experience 必须是 ExperienceDraft")
        if mode not in {RecoveryDecision.REVISE_EXPERIENCE, RecoveryDecision.PROPOSE_EXPERIENCE}:
            raise PatchGenerationError("PatchGenerator 只支持修订或新方向经验")
        if diagnosis.intent_id != plan.intent_id or current_experience.intent_id != plan.intent_id:
            raise PatchGenerationError("诊断、当前经验和计划必须属于同一 intent")
        if current_experience.version < 1:
            raise PatchGenerationError("当前经验版本必须大于 0")
        if version_diff is not None and not isinstance(version_diff, dict):
            raise PatchGenerationError("version_diff 必须是字典或 None")
        active_item = plan.active_item
        if active_item is None:
            raise PatchGenerationError("必须存在一个 ACTIVE 实验计划项")
        expected_version_id = (
            f"{current_experience.experience_id}:version-{current_experience.version}"
        )
        if active_item.experience_version_id != expected_version_id:
            raise PatchGenerationError("活动实验绑定的经验版本与当前经验不一致")
        refuted = [
            item for item in plan.items
            if item.status is ExperimentPlanStatus.REFUTED
        ]
        if mode is RecoveryDecision.PROPOSE_EXPERIENCE and not refuted:
            raise PatchGenerationError("只有原方向被否定后才能生成新方向草稿")
        if not isinstance(experiment_feedback, list) or any(
            not isinstance(item, AttemptResult) for item in experiment_feedback
        ):
            raise PatchGenerationError("experiment_feedback 必须是 AttemptResult 数组")
        feedback = [
            item for item in experiment_feedback
            if item.status in {AttemptStatus.COMPLETED, AttemptStatus.FAILED}
            and item.request.intent_id == plan.intent_id
            and item.request.experience is not None
            and item.request.experience.experience_id == current_experience.experience_id
            and item.request.experience.version == current_experience.version
        ]
        if not feedback:
            raise PatchGenerationError(
                "没有与当前经验版本匹配的正常完成或失败反馈，不能生成修订"
            )
        return active_item, feedback

    @staticmethod
    def _new_experience_id(current: ExperienceDraft) -> str:
        return (
            f"{current.experience_id}__v{current.version + 1}"
            f"__{uuid.uuid4().hex[:12]}"
        )

    def _allowed_evidence(
        self,
        plan: ExperimentPlan,
        diagnosis: DiagnosisReport,
        current_experience: ExperienceDraft,
        feedback: List[AttemptResult],
    ) -> Dict[str, set]:
        refs = list(current_experience.evidence_refs)
        if plan.active_item is not None:
            refs.extend(plan.active_item.evidence_refs)
        for pattern in diagnosis.failure_patterns:
            refs.extend(pattern.evidence_refs)
        for item in plan.items:
            if item.status is ExperimentPlanStatus.REFUTED:
                refs.extend(item.evidence_refs)
                refs.extend(item.result_evidence_refs)
        allowed = self._index_evidence(refs)
        for result in list(diagnosis.attempt_results) + list(feedback):
            if result.status not in {AttemptStatus.COMPLETED, AttemptStatus.FAILED}:
                continue
            ref = self._attempt_evidence(result)
            if ref is not None:
                allowed.setdefault(ref.sessionid, set()).update(ref.step_ids)
        return allowed

    @staticmethod
    def _attempt_evidence(result: AttemptResult) -> Optional[EvidenceRef]:
        sessionid = result.trajectory.get("sessionid")
        if not isinstance(sessionid, str) or not sessionid.strip():
            return None
        step_ids = [
            key for key in result.trajectory
            if isinstance(key, str)
            and key.startswith("step")
            and key[4:].isdigit()
        ]
        return EvidenceRef(sessionid=sessionid.strip(), step_ids=step_ids)

    @staticmethod
    def _index_evidence(refs: List[EvidenceRef]) -> Dict[str, set]:
        allowed: Dict[str, set] = {}
        for ref in refs:
            allowed.setdefault(ref.sessionid, set()).update(ref.step_ids)
        return allowed

    def _validate_payload(
        self,
        payload: Any,
        active_hypothesis: str,
        allowed_evidence: Dict[str, set],
    ) -> Tuple[Dict[str, Any], str]:
        if not isinstance(payload, dict):
            raise ValueError("候选经验必须是 JSON 对象")
        fields = {
            "hypothesis",
            "applicable_scope",
            "trigger_conditions",
            "strategy",
            "expected_stage",
            "evidence_refs",
            "forbidden_behaviors",
            "exit_conditions",
            "change_summary",
        }
        if set(payload) != fields:
            raise ValueError(
                f"候选经验字段不匹配，缺少 {sorted(fields - set(payload))}，"
                f"未知 {sorted(set(payload) - fields)}"
            )
        for name in (
            "hypothesis",
            "applicable_scope",
            "strategy",
            "expected_stage",
            "change_summary",
        ):
            if not isinstance(payload[name], str) or not payload[name].strip():
                raise ValueError(f"{name} 必须是非空字符串")
        if payload["hypothesis"].strip() != active_hypothesis:
            raise ValueError("候选经验必须原样保留当前活动假设")
        for name in ("trigger_conditions", "forbidden_behaviors", "exit_conditions"):
            if not isinstance(payload[name], list) or any(
                not isinstance(item, str) or not item.strip() for item in payload[name]
            ):
                raise ValueError(f"{name} 必须是非空字符串数组")
        if not payload["trigger_conditions"] or not payload["exit_conditions"]:
            raise ValueError("trigger_conditions 和 exit_conditions 不能为空")
        evidence_refs = self._parse_evidence_refs(payload["evidence_refs"], allowed_evidence)
        validated = {
            "hypothesis": active_hypothesis,
            "applicable_scope": payload["applicable_scope"].strip(),
            "trigger_conditions": payload["trigger_conditions"],
            "strategy": payload["strategy"].strip(),
            "expected_stage": payload["expected_stage"].strip(),
            "evidence_refs": evidence_refs,
            "forbidden_behaviors": payload["forbidden_behaviors"],
            "exit_conditions": payload["exit_conditions"],
        }
        return validated, payload["change_summary"].strip()

    @staticmethod
    def _parse_evidence_refs(
        raw_refs: Any,
        allowed: Dict[str, set],
    ) -> List[EvidenceRef]:
        if not isinstance(raw_refs, list) or not raw_refs:
            raise ValueError("evidence_refs 必须是非空数组")
        refs = []
        for raw_ref in raw_refs:
            if not isinstance(raw_ref, dict) or set(raw_ref) != {"sessionid", "step_ids"}:
                raise ValueError("证据引用只能包含 sessionid 和 step_ids")
            sessionid = raw_ref["sessionid"]
            step_ids = raw_ref["step_ids"]
            if not isinstance(sessionid, str) or sessionid not in allowed:
                raise ValueError(f"引用了不存在的证据 sessionid: {sessionid}")
            if not isinstance(step_ids, list) or any(
                not isinstance(step_id, str) for step_id in step_ids
            ):
                raise ValueError("step_ids 必须是字符串数组")
            unknown_steps = set(step_ids) - allowed[sessionid]
            if unknown_steps:
                raise ValueError(f"引用了不存在的证据步骤: {sorted(unknown_steps)}")
            refs.append(EvidenceRef(sessionid=sessionid, step_ids=list(step_ids)))
        return refs
