import json
from typing import Dict, List

from .recovery_schemas import RecoveryContext, RecoveryDecision


RECOVERY_PROMPT_VERSION = "recovery_agent_openai_v1"

_DECISIONS = " | ".join(decision.value for decision in RecoveryDecision)

SYSTEM_PROMPT = f"""你是 OOD Recovery Agent 的决策内核。你只能提出结构化动作，不能直接修改实验事实、Recovery 状态、预算或任务完成结论。

每轮只能采取以下一种动作：
1. 使用 OpenAI 原生 tool_calls 调用一个已注册工具，获取新的观察。
2. 已经有足够信息时，在 assistant content 中输出最终决策 JSON，不再调用工具。

最终决策 JSON 格式：
{{
  "decision": "{_DECISIONS}",
  "reason": "决策理由",
  "evidence_refs": [{{"sessionid": "可引用证据sessionid", "step_ids": ["可选stepN"]}}]
}}

约束：
1. 不要把工具调用包装成 call_tool JSON，直接使用系统提供的原生工具。
2. 只能引用初始上下文或工具结果中已有的证据；没有证据时 evidence_refs 为空数组。
3. ACCEPT_EXPERIENCE 只是请求后续验证，不代表任务成功。
4. 不生成经验正文，不提交真机实验，不把 uncertain 当作确定失败。
5. 工具或校验错误会通过后续消息返回，请根据错误修正动作。
"""


def build_initial_messages(context: RecoveryContext) -> List[Dict[str, str]]:
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
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "请开始当前 OOD Recovery 决策。初始上下文摘要：\n"
                + json.dumps(context_preview, ensure_ascii=False, indent=2)
            ),
        },
    ]
