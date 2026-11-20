"""
置信度计算模块
计算生成的 Plan 对某类意图任务成功概率的估计（0-1）。

三层结构：
  1. 统计信号层：簇规模、步数一致性、缩减潜力、冗余比例、缺陷严重度、基线信号、轨迹一致性
  2. LLM 评估层：生成时自评 (initial) + 独立评审 (review)，分歧惩罚
  3. 融合层：final = λ·stat_score + (1-λ)·llm_score

所有权重均可配置 (ConfidenceConfig)，预留 Calibrator 接口用于未来接入真实执行标签校准。
"""
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .schemas import EfficientPlan, TrajectoryRecord


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass
class ConfidenceConfig:
    """置信度计算配置。调整权重即可改变评估侧重点。"""
    # 信号层权重（统计信号 → stat_score，内部自动归一化，不要求和为 1）
    signal_weights: Dict[str, float] = field(default_factory=lambda: {
        "cluster_size": 0.15,
        "step_consistency": 0.20,
        "reduction_potential": 0.15,
        "redundancy_ratio": 0.20,
        "baseline_signal": 0.10,
        "trajectory_consistency": 0.20,
    })
    # LLM 层：llm_score = alpha * initial + (1 - alpha) * review
    alpha: float = 0.3
    # 融合层：final = lambda_ * stat_score + (1 - lambda_) * llm_score
    lambda_: float = 0.3
    # 分歧惩罚：|initial - review| > disagreement_threshold 时开始扣减
    disagreement_threshold: float = 0.2
    disagreement_penalty: float = 0.15
    # 过滤阈值：confidence < threshold → filtered = True
    threshold: float = 0.7

OPTIMAL_CLUSTER_SIZE = 20
MINIMUM_SCORE = 0.3
def cluster_size_score(total: int, optimal_cluster_size: int=OPTIMAL_CLUSTER_SIZE, minimal_score: float=MINIMUM_SCORE) -> float:
    """
    簇规模信号：轨迹数越多越可信（平滑递增，20+ 视为满分）。
    分数被非线性映射到 [0.4, 1.0] 区间，避免少量轨迹得分过低。
    """
    if total <= 0:
        return 0.4
    # 1. 限制最大计算值为 20
    capped_total = min(total, optimal_cluster_size)
    # 2. 计算基础比例 (0.0 ~ 1.0)
    base_ratio = capped_total / float(optimal_cluster_size)
    # 3. 引入平方根进行非线性平滑，提权小数值
    smooth_ratio = base_ratio ** 0.5
    # 4. 映射到 0.4 ~ 1.0 的范围
    score = minimal_score + 0.6 * smooth_ratio

    return _clip01(score)


def step_consistency_score(step_counts: List[int], minimum_score:float=MINIMUM_SCORE) -> float:
    """
    步数一致性信号：变异系数越小越稳定。
    通过平滑的非线性映射，避免正常波动导致分数断崖式下跌，下限控制在 0.4。
    """
    if len(step_counts) < 2:
        return 0.7
    avg = float(np.mean(step_counts))
    if avg <= 0:
        return 0.4  # 异常情况（如步数全为0），触发最低分保底
    # 1. 计算变异系数 CV
    cv = float(np.std(step_counts)) / avg
    # 2. 封顶 CV（假设 CV >= 1.0 也就是标准差大于等于均值时，视为极度不稳定，给最低分）
    capped_cv = min(cv, 1.0)
    # 3. 计算基础一致性度 (1.0 代表完美一致，0.0 代表极度离散)
    base_consistency = 1.0 - capped_cv
    # 4. 引入平方根平滑（对轻微和中度的波动更加宽容）
    smooth_consistency = base_consistency ** 0.5
    # 5. 映射到 0.4 ~ 1.0 区间
    score = minimum_score + 0.6 * smooth_consistency

    return _clip01(score)


def reduction_potential_score(potential: float) -> float:
    """缩减潜力信号：适中偏好（过高说明轨迹分歧大、plan 可能激进）"""
    peak, width = 0.3, 0.4
    return _clip01(math.exp(-(((potential - peak) / width) ** 2)))


def redundancy_ratio_score(total_trajectories: int, redundant_count: int) -> float:
    """冗余比例信号：冗余轨迹占比越低越好"""
    if total_trajectories <= 0:
        return 0.5
    return _clip01(1.0 - redundant_count / total_trajectories)


def baseline_signal_score(has_baseline: bool, diff_improved: Optional[bool]) -> float:
    """基线信号：有基线且补丁评审通过 → 高分；从零生成 → 中性"""
    if not has_baseline:
        return 0.7
    if diff_improved is None:
        return 0.7
    return 0.9 if diff_improved else 0.4


def _action_seq_similarity(actions_a: List[str], actions_b: List[str]) -> float:
    """两条动作序列的相似度：最长公共前缀 + 动作集合 Jaccard"""
    if not actions_a or not actions_b:
        return 0.0
    max_len = max(len(actions_a), len(actions_b))
    prefix = 0
    for a, b in zip(actions_a, actions_b):
        if a == b:
            prefix += 1
        else:
            break
    prefix_score = prefix / max_len
    set_a, set_b = set(actions_a), set(actions_b)
    union = len(set_a | set_b)
    jaccard = (len(set_a & set_b) / union) if union else 0.0
    return 0.5 * prefix_score + 0.5 * jaccard


def trajectory_consistency_score(
        trajectories: List[TrajectoryRecord], max_pairs: int = 50
) -> float:
    """簇内轨迹一致性：平均 pairwise 动作序列相似度（抽样控制成本）"""
    action_lists = [t.get_actions() for t in trajectories if t.get_actions()]
    if len(action_lists) < 2:
        return 0.5
    pairs = [(i, j) for i in range(len(action_lists)) for j in range(i + 1, len(action_lists))]
    if len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)
    scores = [_action_seq_similarity(action_lists[i], action_lists[j]) for i, j in pairs]
    return _clip01(float(np.mean(scores)))


def compute_stat_score(
        config: ConfidenceConfig,
        total_trajectories: int,
        step_counts: List[int],
        reduction_potential: float,
        redundant_patterns: List[Any],
        has_baseline: bool,
        diff_improved: Optional[bool],
        trajectories: List[TrajectoryRecord],
) -> Tuple[float, Dict[str, float]]:
    """统计信号层：计算加权统计分，返回 (stat_score, 各信号值 dict)"""
    redundant_count = sum(
        1 for p in redundant_patterns
        if getattr(p, "redundancy_types", None) or getattr(p, "redundant_steps", None)
    )
    signals = {
        # 尺寸越大越可靠
        "cluster_size": cluster_size_score(total_trajectories),
        # 步骤变异系数越小越可靠
        "step_consistency": step_consistency_score(step_counts),
        # 缩减潜力信号：适中偏好（过高说明轨迹分歧大、plan 可能激进）
        "reduction_potential": reduction_potential_score(reduction_potential),
        # 冗余信号
        "redundancy_ratio": redundancy_ratio_score(total_trajectories, redundant_count),
        # 有基线且补丁评审通过 → 高分；从零生成 → 中性
        "baseline_signal": baseline_signal_score(has_baseline, diff_improved),
        # 簇内轨迹一致性：平均 pairwise 动作序列相似度
        "trajectory_consistency": trajectory_consistency_score(trajectories),
    }
    wsum = sum(config.signal_weights.get(k, 0.0) for k in signals)
    if wsum <= 0:
        stat_score = float(np.mean(list(signals.values())))
    else:
        stat_score = sum(
            config.signal_weights.get(k, 0.0) * v for k, v in signals.items()
        ) / wsum
    return _clip01(stat_score), signals


def fuse_confidence(
        config: ConfidenceConfig,
        stat_score: float,
        llm_initial: Optional[float],
        llm_review: Optional[float],
) -> Tuple[float, Dict[str, Any]]:
    """融合层：统计分 + 双阶段 LLM 分（含分歧惩罚）"""
    breakdown: Dict[str, Any] = {"stat_score": round(stat_score, 4)}

    llm_score = None
    if llm_initial is not None or llm_review is not None:
        initial = llm_initial if llm_initial is not None else llm_review
        review = llm_review if llm_review is not None else llm_initial
        disagreement = abs(initial - review)
        penalty = 0.0
        if disagreement > config.disagreement_threshold:
            penalty = min(config.disagreement_penalty, disagreement * 0.5)
        llm_score = config.alpha * initial + (1.0 - config.alpha) * review - penalty
        breakdown["llm_initial"] = round(initial, 4)
        breakdown["llm_review"] = round(review, 4)
        breakdown["disagreement"] = round(disagreement, 4)
        breakdown["disagreement_penalty"] = round(penalty, 4)
        breakdown["llm_score"] = round(_clip01(llm_score), 4)

    if llm_score is None:
        final = stat_score
        breakdown["fusion_mode"] = "stat_only"
    else:
        final = config.lambda_ * stat_score + (1.0 - config.lambda_) * _clip01(llm_score)
        breakdown["fusion_mode"] = "hybrid"
    final = _clip01(final)
    return final, breakdown


class Calibrator(ABC):
    """校准器接口（预留）：未来接入真实执行标签后，将推断置信度映射为校准概率"""

    @abstractmethod
    def transform(self, raw_confidence: float) -> float:
        """将原始推断置信度映射为校准后的概率"""
        raise NotImplementedError

    @abstractmethod
    def calibrate(self, samples: List[Tuple[EfficientPlan, int]]) -> None:
        """用 (plan, 真实执行结果 0/1) 样本拟合校准映射"""
        raise NotImplementedError


class NoopCalibrator(Calibrator):
    """恒等校准器：未接入真实数据时原样返回"""

    def transform(self, raw_confidence: float) -> float:
        return raw_confidence

    def calibrate(self, samples: List[Tuple[EfficientPlan, int]]) -> None:
        pass
