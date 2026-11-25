from typing import Any, Dict, List, Optional

from ._analysis_llm import StructuredAnalysisClient, StructuredAnalysisError
from .config import OODBaselineAnalysisConfig
from .schemas import (
    AttemptResult,
    AttemptStatus,
    AttemptVerdict,
    BusinessStage,
    BusinessStageRubric,
    StageProgress,
    canonical_json,
    content_hash,
)
from .verdict import result_sessionid, trajectory_steps


RUBRIC_SYSTEM_PROMPT = """你是 OOD 基线业务阶段定义器。
根据同一 intent 的指令和有效轨迹，建立一套供所有 attempt 共用的有序业务阶段。
阶段必须表示实际业务进展，不能按轨迹长度、动作数量或耗时划分。
最后一个阶段必须表示用户任务已经真正完成；中间阶段应能通过页面观察或操作结果验证。
不要生成修复经验，不判断任务是否值得修复，不把设备异常或结果不确定纳入阶段依据。
只输出 JSON 对象，且只能包含 stages 数组。每个阶段只能包含：
stage_name、stage_order、description、completion_evidence。
stage_order 从 1 开始连续递增；completion_evidence 是判断到达该阶段所需的非空证据说明数组。
"""

PROGRESS_SYSTEM_PROMPT = """你是 OOD 单条轨迹业务进度分析器。
只能使用输入给定的共享业务阶段，不得创造新阶段或改变阶段顺序。
找出最后一个具有实际业务进展证据的步骤，而不是简单选择轨迹最后一步。
evidence_steps 和 furthest_anchor 必须引用真实 stepN；furthest_anchor 必须包含在 evidence_steps 中。
first_deviation_step 是首个关键偏离步骤；没有充分证据时输出 null，不能猜测。
completed 必须到达最后一个任务完成阶段；failed 不得声称已经到达该阶段。
不使用连续进度分数，不生成经验，不修改 Verdict。
只输出 JSON 对象，字段必须且只能包含：
stage_name、evidence_steps、furthest_anchor、first_deviation_step。
"""


class BusinessStageAnalyzer:
    def __init__(
        self,
        llm_agent,
        model: str,
        config: OODBaselineAnalysisConfig | None = None,
    ):
        self.client = StructuredAnalysisClient(llm_agent, model, config)

    async def abuild_rubric(
        self,
        intent_id: str,
        results: List[AttemptResult],
    ) -> BusinessStageRubric:
        if not results:
            raise StructuredAnalysisError("没有有效轨迹，无法建立业务阶段")
        payload = {
            "intent_id": intent_id,
            "instructions": list(dict.fromkeys(
                result.request.instruction for result in results
            )),
            "attempts": [
                {
                    "attempt_key": result.request.idempotency_key,
                    "instruction": result.request.instruction,
                    "sessionid": result_sessionid(result),
                    "trajectory": result.trajectory,
                }
                for result in results
            ],
        }

        def validate(raw: Any) -> BusinessStageRubric:
            rubric = self._validate_rubric_payload(raw, intent_id)
            return rubric

        rubric, _ = await self.client.request(
            RUBRIC_SYSTEM_PROMPT,
            payload,
            validate,
        )
        return rubric

    def validate_rubric(
        self,
        rubric: BusinessStageRubric,
        intent_id: str,
    ) -> BusinessStageRubric:
        if not isinstance(rubric, BusinessStageRubric) or rubric.intent_id != intent_id:
            raise StructuredAnalysisError("stage_rubric 必须属于当前 intent")
        self._validate_stages(rubric.stages)
        expected_id = self._rubric_id(intent_id, rubric.stages)
        if rubric.rubric_id != expected_id:
            raise StructuredAnalysisError("stage_rubric 的 rubric_id 校验失败")
        return rubric

    async def aanalyze(
        self,
        result: AttemptResult,
        rubric: BusinessStageRubric,
        verdict: AttemptVerdict,
    ) -> StageProgress:
        if verdict.attempt_key != result.request.idempotency_key:
            raise StructuredAnalysisError("Verdict 与 attempt 身份不一致")
        if verdict.completion_status not in {AttemptStatus.COMPLETED, AttemptStatus.FAILED}:
            raise StructuredAnalysisError("只有明确 completed/failed 才能分析有效业务进度")
        sessionid = result_sessionid(result)
        available_steps = trajectory_steps(result.trajectory)
        payload = {
            "attempt_key": result.request.idempotency_key,
            "instruction": result.request.instruction,
            "sessionid": sessionid,
            "verdict": verdict.to_dict(),
            "stage_rubric": rubric.to_dict(),
            "available_steps": available_steps,
            "trajectory": result.trajectory,
        }

        def validate(raw: Any) -> StageProgress:
            return self._validate_progress_payload(
                raw,
                rubric,
                verdict,
                available_steps,
                result.trajectory,
            )

        progress, _ = await self.client.request(
            PROGRESS_SYSTEM_PROMPT,
            payload,
            validate,
        )
        return progress

    def _validate_rubric_payload(
        self,
        payload: Any,
        intent_id: str,
    ) -> BusinessStageRubric:
        if not isinstance(payload, dict) or set(payload) != {"stages"}:
            raise ValueError("阶段定义只能包含 stages")
        raw_stages = payload["stages"]
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError("stages 必须是非空数组")
        stages = []
        fields = {"stage_name", "stage_order", "description", "completion_evidence"}
        for raw in raw_stages:
            if not isinstance(raw, dict) or set(raw) != fields:
                raise ValueError("业务阶段字段不完整或包含未知字段")
            for name in ("stage_name", "description"):
                if not isinstance(raw[name], str) or not raw[name].strip():
                    raise ValueError(f"{name} 必须是非空字符串")
            evidence = raw["completion_evidence"]
            if (
                not isinstance(evidence, list)
                or not evidence
                or any(not isinstance(item, str) or not item.strip() for item in evidence)
            ):
                raise ValueError("completion_evidence 必须是非空字符串数组")
            stages.append(BusinessStage(
                stage_name=raw["stage_name"].strip(),
                stage_order=raw["stage_order"],
                description=raw["description"].strip(),
                completion_evidence=list(evidence),
            ))
        self._validate_stages(stages)
        return BusinessStageRubric(
            intent_id=intent_id,
            stages=stages,
            rubric_id=self._rubric_id(intent_id, stages),
        )

    @staticmethod
    def _validate_stages(stages: List[BusinessStage]) -> None:
        if not stages or any(not isinstance(stage, BusinessStage) for stage in stages):
            raise ValueError("stages 必须是非空 BusinessStage 数组")
        orders = [stage.stage_order for stage in stages]
        if any(isinstance(order, bool) or not isinstance(order, int) for order in orders):
            raise ValueError("stage_order 必须是整数")
        if orders != list(range(1, len(stages) + 1)):
            raise ValueError("stage_order 必须从 1 开始连续递增")
        names = [stage.stage_name for stage in stages]
        if len(set(names)) != len(names) or any(not name.strip() for name in names):
            raise ValueError("stage_name 必须非空且不能重复")
        for stage in stages:
            if not isinstance(stage.description, str) or not stage.description.strip():
                raise ValueError("description 必须是非空字符串")
            if (
                not isinstance(stage.completion_evidence, list)
                or not stage.completion_evidence
                or any(
                    not isinstance(item, str) or not item.strip()
                    for item in stage.completion_evidence
                )
            ):
                raise ValueError("completion_evidence 必须是非空字符串数组")

    @staticmethod
    def _rubric_id(intent_id: str, stages: List[BusinessStage]) -> str:
        return content_hash({
            "intent_id": intent_id,
            "stages": [stage.to_dict() for stage in stages],
        })[:24]

    @staticmethod
    def _validate_progress_payload(
        payload: Any,
        rubric: BusinessStageRubric,
        verdict: AttemptVerdict,
        available_steps: List[str],
        trajectory: Dict[str, Any],
    ) -> StageProgress:
        fields = {"stage_name", "evidence_steps", "furthest_anchor", "first_deviation_step"}
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ValueError("进度结果字段不完整或包含未知字段")
        by_name = {stage.stage_name: stage for stage in rubric.stages}
        stage_name = payload["stage_name"]
        if stage_name not in by_name:
            raise ValueError("stage_name 不在共享业务阶段中")
        evidence_steps = payload["evidence_steps"]
        if (
            not isinstance(evidence_steps, list)
            or not evidence_steps
            or any(not isinstance(step, str) for step in evidence_steps)
        ):
            raise ValueError("evidence_steps 必须是非空字符串数组")
        unknown = set(evidence_steps) - set(available_steps)
        if unknown:
            raise ValueError(f"进度引用了不存在的步骤: {sorted(unknown)}")
        anchor = payload["furthest_anchor"]
        if not isinstance(anchor, str) or anchor not in evidence_steps:
            raise ValueError("furthest_anchor 必须是 evidence_steps 中的真实步骤")
        deviation = payload["first_deviation_step"]
        if deviation in (None, ""):
            deviation = None
        elif not isinstance(deviation, str) or deviation not in available_steps:
            raise ValueError("first_deviation_step 必须是真实步骤或 null")
        stage = by_name[stage_name]
        is_final = stage.stage_order == len(rubric.stages)
        if verdict.completion_status is AttemptStatus.COMPLETED and not is_final:
            raise ValueError("completed 必须到达最后一个任务完成阶段")
        if verdict.completion_status is AttemptStatus.FAILED and is_final:
            raise ValueError("failed 不能到达最后一个任务完成阶段")
        return StageProgress(
            stage_name=stage_name,
            stage_order=stage.stage_order,
            evidence_steps=list(dict.fromkeys(evidence_steps)),
            furthest_anchor=anchor,
            first_deviation_step=deviation,
            repeated_action_steps=repeated_action_steps(trajectory),
            page_oscillation_steps=page_oscillation_steps(trajectory),
        )


def _action_signature(step: Dict[str, Any]) -> str:
    values = {
        name: step.get(name)
        for name in ("action", "manager_action", "actionPurpose")
        if step.get(name) not in (None, "", {})
    }
    return canonical_json(values) if values else ""


def repeated_action_steps(trajectory: Dict[str, Any]) -> List[str]:
    steps = trajectory_steps(trajectory)
    repeated = []
    for previous, current in zip(steps, steps[1:]):
        previous_step = trajectory[previous]
        current_step = trajectory[current]
        if not isinstance(previous_step, dict) or not isinstance(current_step, dict):
            continue
        previous_signature = _action_signature(previous_step)
        if previous_signature and previous_signature == _action_signature(current_step):
            repeated.extend([previous, current])
    return list(dict.fromkeys(repeated))


def _page_signature(step: Dict[str, Any]) -> str:
    for container in (step, step.get("observation")):
        if not isinstance(container, dict):
            continue
        for name in ("page_id", "page_name", "activity", "current_page"):
            value = container.get(name)
            if isinstance(value, (str, int)) and str(value).strip():
                return f"{name}:{value}"
    return ""


def page_oscillation_steps(trajectory: Dict[str, Any]) -> List[str]:
    steps = trajectory_steps(trajectory)
    oscillating = []
    for index in range(len(steps) - 2):
        window = steps[index:index + 3]
        values = []
        for step_id in window:
            step = trajectory[step_id]
            values.append(_page_signature(step) if isinstance(step, dict) else "")
        if values[0] and values[0] == values[2] and values[0] != values[1]:
            oscillating.extend(window)
    return list(dict.fromkeys(oscillating))
