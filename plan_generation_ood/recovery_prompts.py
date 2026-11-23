import json
from typing import Any, Dict, List

from .recovery_schemas import RecoveryContext, RecoveryDecision


RECOVERY_PROMPT_VERSION = "recovery_agent_s01_v1"


def build_recovery_prompt(
    context: RecoveryContext,
    transcript: List[Dict[str, Any]],
) -> str:
    context_preview = {
        "intent_id": context.intent_id,
        "state": context.state.value,
        "has_diagnosis": context.diagnosis is not None,
        "has_recovery_value": context.recovery_value is not None,
        "current_experience_id": (
            context.current_experience.experience_id
            if context.current_experience else None
        ),
        "budget": context.budget,
        "available_evidence_refs": [ref.to_dict() for ref in context.evidence_refs],
    }
    decisions = " | ".join(decision.value for decision in RecoveryDecision)
    return f"""你是 OOD Recovery Agent 的最小决策内核。你只能提出结构化动作，不能直接修改实验事实、Recovery 状态、预算或任务完成结论。

当前上下文摘要：
{json.dumps(context_preview, ensure_ascii=False, indent=2)}

本阶段只允许一个只读工具：
- get_recovery_context：输入 {{"intent_id": "当前意图ID"}}，返回当前意图的完整结构化上下文。不能查询其他意图。

每次只输出一个 JSON 对象，不要使用 Markdown。

需要读取上下文时：
{{
  "action": "call_tool",
  "tool": "get_recovery_context",
  "input": {{"intent_id": "{context.intent_id}"}},
  "reason": "为什么需要读取",
  "evidence_refs": []
}}

已经有足够信息提出本轮建议时：
{{
  "action": "finish",
  "decision": "{decisions}",
  "reason": "决策理由",
  "evidence_refs": [{{"sessionid": "可引用证据sessionid", "step_ids": ["可选stepN"]}}]
}}

约束：
1. call_tool 和 finish 只能二选一，不能同时输出。
2. 只能引用 available_evidence_refs 或工具结果中已有的证据；没有证据时 evidence_refs 为空数组。
3. ACCEPT_EXPERIENCE 只是请求后续验证，不代表任务成功。
4. 不生成经验正文，不提交真机实验，不把 uncertain 当作确定失败。
5. 工具或校验错误会作为下一条记录返回，请根据错误修正动作。

已有动作和工具结果：
{json.dumps(transcript, ensure_ascii=False, indent=2, default=str)}
"""
