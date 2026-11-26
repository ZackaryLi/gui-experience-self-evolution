import math
import re
from typing import Any, Dict, List

from ._analysis_llm import StructuredAnalysisClient, StructuredAnalysisError
from .config import OODBaselineAnalysisConfig
from .schemas import AttemptResult, AttemptStatus, AttemptVerdict


VERDICT_PROMPT_VERSION = "ood_attempt_verdict_v1"

VERDICT_SYSTEM_PROMPT = """你是独立的 GUI Agent 单条轨迹任务完成判定器。
你只判断当前 attempt 是否满足用户原始任务，并在明确失败时进行 A～E 根因分类。
不得参考 Runner 预设的完成状态，不得生成经验、决定后续探索或修改任何执行事实。
必须依据最终页面观察和实际动作证据，不能仅凭 Agent 声称完成、停止运行或到达相关页面就判成功。
找对页面但未最终确认、只完成部分任务均不是 completed；证据不足时输出 uncertain。
设备或 Runner 故障不属于本组件输入，不得把普通不确定结果强行判为 failed。

失败根因 taxonomy：
A 用户指令表达问题：A1 语义模糊，A2 信息缺失未澄清，A3 多任务指令，A4 内容生成或槽位填充错误，A5 意图理解错误。
B 任务能力或客观环境限制：B1 App、功能或目标内容不存在，B2 系统权限、网络、登录或人工验证限制，B3 功能无法通过 UI 自动化。
C 界面感知问题：C1 OCR 识别错误，C2 UI 结构变化，C3 控件识别失败，C4 误触。
D 执行规划问题：D1 路径选择或交互时序错误，D2 循环探索，D3 未找到关键入口，D4 探索不足或过早终止。
E 跨应用任务问题：E1 跨 App 流程，E2 数据跨 App 传递。
一次搜索无结果或一次失败不足以证明 B1；登录、权限、验证码等应标记相应 blocker_type，不得直接等同于 App 功能缺失。

只输出 JSON 对象，不要 Markdown，字段必须且只能包含：
completion_status、task_completed、completion_analysis、failure_analysis、
primary_failure_parent_category、primary_failure_subcategory、
key_evidence_steps、confidence、blocker_type。
completion_status 为 completed/failed/uncertain；completed 时 task_completed=true，failed 时为 false，uncertain 时为 null。
completed 时失败分类和 failure_analysis 必须为空；failed 时必须给出有效父子分类和 failure_analysis。
key_evidence_steps 只能使用输入中真实存在的 stepN，且至少包含一个关键依据。
confidence 为 0 到 1；blocker_type 可为空字符串或 LOGIN、PERMISSION、CAPTCHA、MISSING_TEST_DATA、UNSUPPORTED_ENVIRONMENT、CAPABILITY_UNSUPPORTED、OTHER。
"""

SUBCATEGORIES = {
    "A": {"A1", "A2", "A3", "A4", "A5"},
    "B": {"B1", "B2", "B3"},
    "C": {"C1", "C2", "C3", "C4"},
    "D": {"D1", "D2", "D3", "D4"},
    "E": {"E1", "E2"},
}
BLOCKER_TYPES = {
    "",
    "LOGIN",
    "PERMISSION",
    "CAPTCHA",
    "MISSING_TEST_DATA",
    "UNSUPPORTED_ENVIRONMENT",
    "CAPABILITY_UNSUPPORTED",
    "OTHER",
}


def trajectory_steps(trajectory: Dict[str, Any]) -> List[str]:
    steps = [
        key for key in trajectory
        if isinstance(key, str) and re.fullmatch(r"step\d+", key)
    ]
    return sorted(steps, key=lambda key: int(key[4:]))


def result_sessionid(result: AttemptResult) -> str:
    value = result.trajectory.get("sessionid", "")
    return value.strip() if isinstance(value, str) else ""


class AttemptVerdictAnalyzer:
    def __init__(
        self,
        llm_agent,
        model: str,
        config: OODBaselineAnalysisConfig | None = None,
    ):
        self.client = StructuredAnalysisClient(llm_agent, model, config)

    async def aanalyze(self, result: AttemptResult) -> AttemptVerdict:
        if not isinstance(result, AttemptResult):
            raise StructuredAnalysisError("result 必须是 AttemptResult")
        if result.status not in {
            AttemptStatus.FINISHED,
            AttemptStatus.COMPLETED,
            AttemptStatus.FAILED,
        }:
            raise StructuredAnalysisError("只有正常 finished/completed/failed 才能进行语义 Verdict")
        sessionid = result_sessionid(result)
        steps = trajectory_steps(result.trajectory)
        if not sessionid or not steps:
            raise StructuredAnalysisError("正常 attempt 缺少 sessionid 或有效 stepN")

        payload = {
            "prompt_version": VERDICT_PROMPT_VERSION,
            "attempt_key": result.request.idempotency_key,
            "instruction": result.request.instruction,
            "sessionid": sessionid,
            "available_steps": steps,
            "environment": result.environment,
            "trajectory": result.trajectory,
        }

        def validate(raw: Any) -> AttemptVerdict:
            return self._validate_payload(raw, result, sessionid, steps)

        verdict, _ = await self.client.request(
            VERDICT_SYSTEM_PROMPT,
            payload,
            validate,
        )
        return verdict

    @staticmethod
    def _validate_payload(
        payload: Any,
        result: AttemptResult,
        sessionid: str,
        available_steps: List[str],
    ) -> AttemptVerdict:
        if not isinstance(payload, dict):
            raise ValueError("Verdict 必须是 JSON 对象")
        fields = {
            "completion_status",
            "task_completed",
            "completion_analysis",
            "failure_analysis",
            "primary_failure_parent_category",
            "primary_failure_subcategory",
            "key_evidence_steps",
            "confidence",
            "blocker_type",
        }
        if set(payload) != fields:
            raise ValueError("Verdict 字段不完整或包含未知字段")
        try:
            status = AttemptStatus(payload["completion_status"])
        except (TypeError, ValueError) as error:
            raise ValueError("completion_status 枚举值非法") from error
        if status not in {AttemptStatus.COMPLETED, AttemptStatus.FAILED, AttemptStatus.UNCERTAIN}:
            raise ValueError("Verdict 只能输出 completed/failed/uncertain")
        expected_flag = {
            AttemptStatus.COMPLETED: True,
            AttemptStatus.FAILED: False,
            AttemptStatus.UNCERTAIN: None,
        }[status]
        if payload["task_completed"] is not expected_flag:
            raise ValueError("task_completed 与 completion_status 不一致")
        for name in ("completion_analysis", "failure_analysis"):
            if not isinstance(payload[name], str):
                raise ValueError(f"{name} 必须是字符串")
        if not payload["completion_analysis"].strip():
            raise ValueError("completion_analysis 不能为空")
        parent = payload["primary_failure_parent_category"]
        subcategory = payload["primary_failure_subcategory"]
        if not isinstance(parent, str) or not isinstance(subcategory, str):
            raise ValueError("失败分类必须是字符串")
        parent = parent.strip().upper()
        subcategory = subcategory.strip().upper()
        if status is AttemptStatus.FAILED:
            if parent not in SUBCATEGORIES or subcategory not in SUBCATEGORIES[parent]:
                raise ValueError("failed 必须提供匹配的 A～E 父子分类")
            if not payload["failure_analysis"].strip():
                raise ValueError("failed 必须提供 failure_analysis")
        elif parent or subcategory or payload["failure_analysis"].strip():
            raise ValueError("非 failed 结果不能携带失败分类或失败归因")
        blocker_type = payload["blocker_type"]
        if blocker_type not in BLOCKER_TYPES:
            raise ValueError("blocker_type 枚举值非法")
        steps = payload["key_evidence_steps"]
        if (
            not isinstance(steps, list)
            or not steps
            or any(not isinstance(step, str) for step in steps)
        ):
            raise ValueError("key_evidence_steps 必须是非空字符串数组")
        unknown = set(steps) - set(available_steps)
        if unknown:
            raise ValueError(f"引用了不存在的证据步骤: {sorted(unknown)}")
        confidence = payload["confidence"]
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError("confidence 必须是 0 到 1 之间的有限数值")
        return AttemptVerdict(
            attempt_key=result.request.idempotency_key,
            sessionid=sessionid,
            completion_status=status,
            task_completed=expected_flag,
            completion_analysis=payload["completion_analysis"].strip(),
            failure_analysis=payload["failure_analysis"].strip(),
            primary_failure_parent_category=parent,
            primary_failure_subcategory=subcategory,
            key_evidence_steps=list(dict.fromkeys(steps)),
            confidence=float(confidence),
            blocker_type=blocker_type,
        )
