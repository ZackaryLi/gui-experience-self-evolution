import json
from typing import Any, Dict, List

from .schemas import BatchRunConstraints, IntentKey, LoadedTrajectory, canonical_json


BATCH_VALUE_PROMPT_VERSION = "batch_value_v1"


def build_batch_value_prompt(
    key: IntentKey,
    records: List[LoadedTrajectory],
    planned_attempt_count: int,
    constraints: BatchRunConstraints,
    previous_error: str = "",
) -> str:
    prompt_records = [_prompt_record(record) for record in records]
    error_section = (
        f"\n上一次输出未通过校验：{previous_error}\n请修正后重新输出。\n"
        if previous_error else ""
    )
    return f"""你是 OOD 轨迹基线批跑价值评估器。你的判断是主要决策依据，确定性规则只负责数据、安全、预算和输出合法性约束。

任务：阅读同一 OOD 意图下的历史 D 类失败轨迹，判断现在是否值得进行无经验基线批跑。这里不生成修复经验，也不判断 Recovery Agent 后续是否应持续探索。

意图信息：
{canonical_json(key.to_dict())}

执行约束：
{canonical_json(constraints.to_dict())}

若建议批跑，计划执行 {planned_attempt_count} 个 attempt。批跑目的，是以合理成本获得能够影响后续决策的新信息。

请按以下清单进行语义判断：
1. 当前是否仍失败尚不明确、历史失败可能随环境变化、需要判断失败稳定性或比较不同指令时，通常具有批跑价值。
2. 错误入口、搜索策略、重复无效动作、页面状态理解、完成验证、循环退出等可能的策略问题，通常值得通过基线进一步确认。
3. 单条指令也可以有批跑价值；指令重复不等于没有价值，重复 attempt 可用于稳定性判断。
4. 未登录、无权限、验证码、缺少必要测试数据、账号或地区条件不满足，且 GUI Agent 无法解决时，应 STOP_BLOCKED。
5. 已有充分且一致的证据表明 App 确实不支持目标能力，并排除环境因素时，才 STOP_UNSOLVABLE。一次搜索无结果或一次失败不够。
6. 已有兼容基线或重复执行不会带来新信息时，应 SKIP_NO_BATCH_VALUE。
7. 意图不清、证据冲突、授权或安全范围不明、多根因无法作为同批判断、证据不足且批跑也无法解决该不确定性时，应 MANUAL_REVIEW。
8. 不要把“历史轨迹没有说明当前是否失败”本身当作人工复核理由，这正是基线可以补充的信息。
9. 所有结论必须引用真实 sessionid；引用步骤时只能使用输入中实际存在的 stepN。
10. 不要因为想进入后续流程而默认 RUN_BASELINE，也不要因为任务困难而默认不可解。

历史轨迹：
{json.dumps(prompt_records, ensure_ascii=False, indent=2, default=str)}
{error_section}
只输出一个 JSON 对象，不要使用 Markdown：
{{
  "decision": "RUN_BASELINE | SKIP_NO_BATCH_VALUE | STOP_BLOCKED | STOP_UNSOLVABLE | MANUAL_REVIEW",
  "reason": "主要判断理由",
  "information_value": "HIGH | MEDIUM | LOW | UNKNOWN",
  "repair_potential": "HIGH | MEDIUM | LOW | UNKNOWN",
  "confidence": 0.0,
  "blocker_type": "空字符串或 LOGIN | PERMISSION | CAPTCHA | MISSING_TEST_DATA | UNSUPPORTED_ENVIRONMENT | AUTHORIZATION | SAFETY | CAPABILITY_UNSUPPORTED | BUDGET | OTHER",
  "evidence_refs": [
    {{"sessionid": "真实sessionid", "step_ids": ["可选的真实stepN"]}}
  ]
}}
"""


def _prompt_record(record: LoadedTrajectory) -> Dict[str, Any]:
    data = {
        key: value
        for key, value in record.record.items()
        if key != "intent_embedding"
    }
    data["sessionid"] = record.sessionid
    return data
