import typing as t
import re

TAXONOMY_DESC = """
你需要从以下 taxonomy 中选择一个最合适的父类和子类。

A 用户指令表达问题（Query表达层）
  A1 语义模糊：指令表达不清晰，难以确定具体任务目标
  A2 信息缺失未澄清：用户提供的信息不足以唯一确定任务内容，但 Agent 没有追问或做合理澄清
  A3 多任务指令：一个指令中同时包含多个任务目标
  A4 内容生成/槽位填充错误：Agent在生成最终文本内容时出现重复、冗余、错填、漏填等问题
  A5 意图理解错误：Agent对意图的理解有误

B 任务能力不可达（Agent能力层）
  B1 App及其功能不存在：没有找到应用或者应用中没有该功能或入口
  B2 系统权限限制：系统设置或权限限制导致无法操作
  B3 功能不可自动化：功能存在，但Agent无法通过UI完成

C 界面感知问题（UI感知层）
  C1 OCR识别错误：页面文字识别不准确
  C2 UI结构变化：页面布局变化导致元素定位失败
  C3 控件识别失败：按钮、入口或组件检测不到
  C4 误触：Agent目的正确，但点击位置错误，导致进入错误页面或触发错误操作

D 执行规划问题（Agent决策层）
  D1 路径选择错误：Agent选择了错误的操作路径
  D2 循环探索：Agent在多个页面之间重复操作（循环探索、重复点击与重复返回等）
  D3 未找到关键入口：入口存在但未被发现
  D4 探索不足/过早终止：当前路径失败后，Agent没有继续尝试其他可行路径

E 跨应用任务问题（任务结构层）    
  E1 跨App流程：任务需要在多个应用之间跳转完成
  E2 数据跨App传递：如文件、图片或文本在不同App之间传输
"""

CATEGORY_NAME = {
    "A": "用户指令表达问题",
    "B": "任务能力不可达",
    "C": "界面感知问题",
    "D": "执行规划问题",
    "E": "跨应用任务问题",
}

SUBCATEGORY_NAME = {
    "A1": "语义模糊",
    "A2": "信息缺失未澄清",
    "A3": "多任务指令",
    "A4": "内容生成/槽位填充错误",
    "A5": "意图理解错误",
    "B1": "App功能不存在",
    "B2": "系统权限限制",
    "B3": "功能不可自动化",
    "C1": "OCR识别错误",
    "C2": "UI结构变化",
    "C3": "控件识别失败",
    "C4": "误触",
    "D1": "路径选择错误",
    "D2": "循环探索",
    "D3": "未找到关键入口",
    "D4": "探索不足/过早终止",
    "E1": "跨App流程",
    "E2": "数据跨App传递",
}

def resolve_task_completed(verdict: t.Any) -> t.Tuple[t.Optional[bool], t.Optional[str]]:
    """
    显式解析 verdict 中的 task_completed 完成状态，只信任真正的布尔值。

    返回 (task_completed, invalid_reason)：
    - task_completed 为 True/False 时 invalid_reason 为 None，可直接用于 IND/OOD 分流；
    - 无法确定时 task_completed 为 None，invalid_reason 说明原因。调用方应按解析失败处理，
      严禁默认当作成功——否则未完成判定的轨迹会误入 success_trajectories，污染 IND 经验池。

    背景：旧写法 verdict.get('task_completed', "None") == "None" 只能拦住"键缺失"的情况；
    若 LLM 输出 JSON null（解析为 Python None），会绕过该校验，并因 "None is False"
    不成立而落入 success 分支，导致完成状态不明的轨迹被当作成功。
    """
    if not isinstance(verdict, dict):
        return None, f"verdict 结构异常（类型: {type(verdict).__name__}）"

    task_completed = verdict.get('task_completed')

    if isinstance(task_completed, bool):
        return task_completed, None

    # 非布尔时尝试从 completion_status 文本兜底恢复，
    # 避免 LLM 漏输出布尔字段导致整批轨迹直接报废
    status_text = str(verdict.get('completion_status', '')).strip().lower()
    if status_text == 'completed':
        return True, None
    if status_text == 'failed':
        return False, None
    if status_text == 'uncertain':
        # LLM 明确判定"无法确定完成状态"：合法结果但不可用于 IND/OOD 分流
        return None, "verdict 判定为 uncertain，完成状态无法确定"

    return None, f"verdict 缺少合法的 task_completed 布尔值（实际值: {task_completed!r}）"


def build_batch_verdict_prompt(trajectories: t.List[t.Dict[str, str]]) -> str:
    """
    构建用于批量轨迹判定的 Prompt。
    输入 trajectories 格式示例: [{"id": "traj_001", "trajectory_text": "..."}, ...]
    """
    trajectories_str = ""
    for item in trajectories:
        trajectories_str += f"\n\n====================\n[轨迹 ID: {item['id']}]\n====================\n{item['trajectory_text']}\n"

    prompt = f"""
你是 GUI Agent 执行轨迹任务完成判定与失败诊断专家。你需要批量分析多条 GUI Agent 执行轨迹。

对于提供的**每一条轨迹**，你需要独立判断它是否完成了用户原始任务，并判断失败轨迹是否适合进一步挖掘。

====================
一、你需要判断的内容及判定规则
====================
1. 任务是否完成 (completion_status)：
   - completed / failed / uncertain
   - 规则：不可仅因Agent停止就判完成。必须根据观察内容看是否满足最终意图。若找对页面但未点击最终确认、仅完成部分任务等均判 failed。
2. 若任务失败，具体失败的根因类别 (依据 TAXONOMY_DESC)。

====================
二、输出要求
====================
必须输出一个合法的 JSON 数组 (Array)，为每一条输入的轨迹生成对应的判定对象。不要 markdown 标记，不要额外解释。

[
  {{
    "id": "对应的轨迹ID",
    "task_completion_analysis": "分析过程，不少于40字，不多于100字",
    "completion_status": "completed/failed/uncertain",
    "failure_analysis": "失败的具体根因分析；若完成则为空",
    "primary_failure_parent_category": "A/B/C... 或空",
    "primary_failure_subcategory": "如A1/D2，若无则为空",
    "secondary_failure_parent_categories": ["C", "D"],
    "secondary_failure_subcategories": ["C3", "D4"],
    "task_completed": true,
    "key_evidence_steps": [1, 2],
    "final_judgement": "面向表格展示的一句话结论，不超过80字",
    "confidence": 0.92
  }},
  ...
]

====================
三、待判定的批量轨迹
====================
以下是需要你判定的多条轨迹。请务必为每一个 ID 输出对应的结果。
{trajectories_str}
""".strip()
    return prompt




def get_sorted_steps(trajectory: t.Dict[str, t.Any]) -> t.List[t.Dict[str, t.Any]]:
    step_items = []

    for k, v in trajectory.items():
        if re.fullmatch(r"step\d+", k) and isinstance(v, dict):
            step_num = int(k.replace("step", ""))
            step_items.append((step_num, v))

    step_items.sort(key=lambda x: x[0])

    return [
        {
            "step_num": step_num,
            **step_data
        }
        for step_num, step_data in step_items
    ]

def build_trajectory_text(
        trajectory: t.Dict[str, t.Any],
        max_steps: int = 80,
        head_keep_steps: int = 20
) -> str:
    app = trajectory.get("app", "")
    instruction = trajectory.get("instruction", "")
    steps = get_sorted_steps(trajectory)

    header_lines = [
        f"app: {app}",
        f"instruction: {instruction}",
        "",
        "轨迹如下："
    ]

    # 1. 先格式化所有非空步骤
    blocks: t.List[t.Tuple[int, str]] = []
    for step in steps:
        action = str(step.get("action", "")).strip()
        thought = str(step.get("thought", "")).strip()
        role = str(step.get("role", "")).strip()
        manager_action = str(step.get("manager_action", "None")).strip()
        action_purpose = str(step.get("actionPurpose", "")).strip()

        # 过滤完全空的无效 step
        if not action and not thought and not action_purpose:
            continue

        step_num = step["step_num"]
        blocks.append((step_num, (
            f"""
            [step{step_num}]
            manager_action={manager_action}
            thought={thought}
            action={action}
            role={role}
            """.strip()
        )))

    # 2. 超长轨迹保留头部摘要 + 完整尾部。任务完成与否由最终状态决定（是否点了
    #    最终确认按钮、停在哪个页面），旧实现固定保留前 max_steps 步、丢弃尾部，
    #    恰好把对判定最关键的收尾步骤截掉，导致长轨迹（往往是最有诊断价值的
    #    失败轨迹）的 verdict 质量系统性偏低。
    if max_steps is not None and len(blocks) > max_steps:
        n_head = max(0, min(head_keep_steps, max_steps))
        n_tail = max(0, max_steps - n_head)
        head_blocks = blocks[:n_head]
        tail_blocks = blocks[len(blocks) - n_tail:] if n_tail > 0 else []
        omitted = len(blocks) - len(head_blocks) - len(tail_blocks)

        body_parts = [b[1] for b in head_blocks]
        body_parts.append(
            f"[中间省略 {omitted} 步：为保证完成度判定的准确性，已保留前 {len(head_blocks)} 步"
            f"与最后 {len(tail_blocks)} 步的完整执行记录，任务最终状态在尾部步骤中]"
        )
        body_parts.extend(b[1] for b in tail_blocks)
    else:
        body_parts = [b[1] for b in blocks]

    return "\n\n".join(header_lines + body_parts)