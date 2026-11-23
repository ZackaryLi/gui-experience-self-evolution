import json
from typing import Any, Dict, List

from .recovery_schemas import RecoveryContext, RecoveryDecision
from .tools import ToolRegistry


RECOVERY_PROMPT_VERSION = "recovery_agent_s02_v1"


def build_recovery_prompt(
    context: RecoveryContext,
    transcript: List[Dict[str, Any]],
    tool_registry: ToolRegistry,
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
    tool_descriptions = json.dumps(
        tool_registry.describe(),
        ensure_ascii=False,
        indent=2,
    )
    return f"""你是 OOD Recovery Agent 的决策内核。你只能提出结构化动作，不能直接修改实验事实、Recovery 状态、预算或任务完成结论。

当前上下文摘要：
{json.dumps(context_preview, ensure_ascii=False, indent=2)}

当前注册的工具及其输入 Schema：
{tool_descriptions}

每次只输出一个 JSON 对象，不要使用 Markdown。

需要调用工具时：
{{
  "action": "call_tool",
  "tool": "从当前注册工具中选择的名称",
  "input": {{"参数": "必须符合所选工具的 input_schema"}},
  "reason": "为什么需要调用",
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
2. 只能使用上方列出的工具，input 必须严格符合对应 Schema；不要调用未注册能力。
3. 只能引用 available_evidence_refs 或工具结果中已有的证据；没有证据时 evidence_refs 为空数组。
4. ACCEPT_EXPERIENCE 只是请求后续验证，不代表任务成功。
5. 不生成经验正文，不提交真机实验，不把 uncertain 当作确定失败。
6. 工具或校验错误会作为下一条记录返回，请根据错误修正动作。

已有动作和工具结果：
{json.dumps(transcript, ensure_ascii=False, indent=2, default=str)}
"""
