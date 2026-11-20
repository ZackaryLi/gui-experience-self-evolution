from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# =========================
# 数据结构定义
# =========================
@dataclass
class TrajectoryStep:
    """单条轨迹的单个步骤"""
    step_num: int
    manager_action: str
    thought: str
    action: str
    role: str


@dataclass
class TrajectoryRecord:
    """单条轨迹记录"""
    trajectory_id: str
    task_cls: str
    app: str
    instruction: str
    intent_name: str
    raw_plan: str
    steps: List[TrajectoryStep]
    task_status: str = "success"
    issue_level: str = "none"
    raw_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def get_actions(self) -> List[str]:
        return [s.action for s in self.steps if s.action]

    def get_manager_actions(self) -> List[str]:
        return [s.manager_action for s in self.steps if s.manager_action]

    def get_thoughts(self) -> List[str]:
        return [s.thought for s in self.steps if s.thought]

    def get_raw_plan(self) -> str:
        return self.raw_plan


@dataclass
class IntentGroup:
    """同一意图下的一组轨迹"""
    task_cls: str
    app: str
    intent_name: str
    trajectories: List[TrajectoryRecord]


@dataclass
class GoldenPathPattern:
    """最优路径及其对应的轨迹index列表"""
    golden_path: List[str]
    relevant_trajectories: List[TrajectoryRecord]

    def to_json(self) -> Dict[str, Any]:
        return {
            "golden_path": self.golden_path,
            "relevant_trajectories": [self.relevant_trajectories],
        }


@dataclass
class RedundantPattern:
    """识别出的冗余模式"""
    trajectory_id: str
    instruction: str
    redundancy_analysis: str
    redundancy_types: List[str]
    redundant_steps: List[str]
    detailed_analysis: str
    plan_defect_analysis: str = ""  # 新增：记录基线Plan在这条轨迹中暴露出的指导缺陷

    def to_json(self):
        return {
            "trajectory_id": self.trajectory_id,
            "instruction": self.instruction,
            "redundancy_analysis": self.redundancy_analysis,
            "redundancy_types": self.redundancy_types,
            "redundant_steps": self.redundant_steps,
            "detailed_analysis": self.detailed_analysis,
            "plan_defect_analysis": self.plan_defect_analysis,
        }

    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> "RedundantPattern":
        return RedundantPattern(**json_data)


@dataclass
class PlanPhase:
    """执行计划的单一阶段"""
    phase_idx: int
    phase_goal: str
    mapped_actions: List[str]
    expected_visual_state: str
    attention_warning: str


@dataclass
class EfficientPlan:
    """高效执行Plan (已重构)"""
    intent: str
    app: str
    summary: str
    execution_plan: List[PlanPhase]  # 结构化的执行阶段

    # --- 新增的溯源与迭代元数据 ---
    baseline_trajectory_id: Optional[str] = None  # 基线旧 Plan 来源于哪条轨迹
    generation_mode: str = "from_scratch"  # 取值: 'patched' (打补丁) 或 'from_scratch' (从零生成)
    diff_review: Optional[Dict[str, Any]] = None  # 新旧 Plan 的 Diff 评审报告

    # --- 新增：置信度评估 ---
    confidence: Optional[float] = None  # 0-1，该 plan 解决此类意图任务的成功概率估计
    confidence_reason: str = ""  # 置信度的核心理由（来自独立评审）
    confidence_breakdown: Optional[Dict[str, Any]] = None  # 置信度分解：统计分/LLM 分/各信号值
    filtered: bool = False  # confidence < 阈值时标记 True，由下游决定是否过滤