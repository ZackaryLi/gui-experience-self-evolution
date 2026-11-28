# -*- coding: utf-8 -*-
"""带宽式门控（banded gating）判据——意图聚类改进方案 3.3。

职责：对层次聚类产出的簇对做三档分流与 AND 门控裁决，作为"合并约束"
介入聚类结果（不侵入 sklearn 聚类循环内部，见方案 3.3(c) 两步近似）：

  簇中心相似度 ≥ S_high          → 免检档：直接允许合并（进入候选）
  簇中心相似度 ≤ S_low           → 免拆档：互斥，不进入合并流程
  灰度带 (S_low, S_high)         → 启动 AND 门控：
        两簇间所有跨簇样本对逐对检查
        sim_target ≥ T_target 且 sim_core_action ≥ T_action
        全部通过才允许合并（一票否决制）

设计依据（方案 1.3 / 3.3(b)）：错合代价 ≫ 错分代价，AND 逻辑无加权
补偿空间，任一路弱即拒绝；S_high 上界宁高勿低（余弦分布压缩，近邻任务
也可能 0.95）。全部阈值占位默认值，待 clustering_purity_bench.py 的
calibration 报告校准后修正。
"""

import typing as t
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class BandedGatingConfig:
    """带宽式门控全部阈值（占位默认值，待基准校准）"""
    # 灰度带边界：作用于拼接向量（聚类主度量）的簇中心余弦相似度
    s_high: float = 0.93          # 免检合并上界：中心相似度 ≥ 此值直接允许合并候选
    s_low: float = 0.80           # 免拆下界：中心相似度 ≤ 此值直接互斥
    # AND 门控两路阈值（分字段向量余弦相似度）
    t_target: float = 0.80        # target 路：任务边界主信号，阈值从严
    t_action: float = 0.70        # core_action 路：词表小、相似度普遍偏高，阈值从宽
    # 灰度带内每簇对的最大跨簇检查样本对数（防大簇对全量比较爆炸，超限截断判定失败）
    max_gate_pairs_per_cluster: int = 50


@dataclass
class GateVerdict:
    """单个簇对的门控裁决结果"""
    allowed: bool                  # 是否允许合并
    band: str                      # "free_pass" / "free_block" / "gated_pass" / "gated_block" / "gated_error"
    center_sim: float              # 簇中心余弦相似度
    checked_pairs: int = 0         # 实际检查的跨簇样本对数
    failed_pair_index: t.Tuple[int, int] = None  # 首个未通过门控的样本对（簇内相对索引）


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """两两余弦相似度矩阵，输入已 L2 归一化则为点积；兼容未归一化输入。"""
    a = np.atleast_2d(np.asarray(a, dtype=np.float32))
    b = np.atleast_2d(np.asarray(b, dtype=np.float32))
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm == 0] = 1.0
    b_norm[b_norm == 0] = 1.0
    return (a / a_norm) @ (b / b_norm).T


def _center_vector(vectors: np.ndarray) -> np.ndarray:
    """簇中心 = 成员向量均值后 L2 归一化（零向量原样返回，相似度自然为 0）。"""
    center = np.mean(vectors, axis=0)
    norm = np.linalg.norm(center)
    if norm > 0:
        center = center / norm
    return center


def _sample_indices(n: int, limit: int, rng: t.Optional[np.random.Generator] = None) -> t.List[int]:
    """簇成员数超过 limit 时确定性抽样（保持索引稳定：前 limit 个 + 均匀步长补齐）。"""
    if n <= limit:
        return list(range(n))
    if rng is not None:
        return sorted(rng.choice(n, size=limit, replace=False).tolist())
    step = n / limit
    return sorted(min(n - 1, int(i * step)) for i in range(limit))


def gate_cluster_pair(
        config: BandedGatingConfig,
        composite_a: np.ndarray, composite_b: np.ndarray,
        target_a: t.Optional[np.ndarray], target_b: t.Optional[np.ndarray],
        action_a: t.Optional[np.ndarray], action_b: t.Optional[np.ndarray],
) -> GateVerdict:
    """对一对簇执行三档分流与 AND 门控。

    入参为两簇的成员向量矩阵（行=成员）。composite 为主度量（拼接向量），
    target/action 为分字段向量；分字段向量缺失（旧缓存/空值）时灰度带内
    降级为拒绝合并（宁分不合），并标记 gated_error。
    """
    if len(composite_a) == 0 or len(composite_b) == 0:
        return GateVerdict(False, "free_block", 0.0)

    center_a = _center_vector(np.atleast_2d(composite_a))
    center_b = _center_vector(np.atleast_2d(composite_b))
    denom = float(np.linalg.norm(center_a) * np.linalg.norm(center_b))
    center_sim = float(np.dot(center_a, center_b) / denom) if denom > 0 else 0.0

    # 三档分流
    if center_sim >= config.s_high:
        return GateVerdict(True, "free_pass", round(center_sim, 6))
    if center_sim <= config.s_low:
        return GateVerdict(False, "free_block", round(center_sim, 6))

    # 灰度带：AND 门控，跨簇样本对逐对检查
    idx_a = _sample_indices(len(composite_a), config.max_gate_pairs_per_cluster)
    idx_b = _sample_indices(len(composite_b), config.max_gate_pairs_per_cluster)

    has_target = target_a is not None and target_b is not None and len(target_a) and len(target_b)
    has_action = action_a is not None and action_b is not None and len(action_a) and len(action_b)
    if not has_target or not has_action:
        # 分字段向量缺失：无法执行门控，宁分不合
        return GateVerdict(False, "gated_error", round(center_sim, 6))

    tgt_a = np.atleast_2d(np.asarray(target_a, dtype=np.float32))[idx_a]
    tgt_b = np.atleast_2d(np.asarray(target_b, dtype=np.float32))[idx_b]
    act_a = np.atleast_2d(np.asarray(action_a, dtype=np.float32))[idx_a]
    act_b = np.atleast_2d(np.asarray(action_b, dtype=np.float32))[idx_b]

    tgt_sim = _cosine_matrix(tgt_a, tgt_b)
    act_sim = _cosine_matrix(act_a, act_b)

    checked = 0
    for local_i, i in enumerate(idx_a):
        for local_j, j in enumerate(idx_b):
            checked += 1
            if tgt_sim[local_i, local_j] < config.t_target or act_sim[local_i, local_j] < config.t_action:
                return GateVerdict(
                    False, "gated_block", round(center_sim, 6),
                    checked_pairs=checked, failed_pair_index=(i, j),
                )

    return GateVerdict(True, "gated_pass", round(center_sim, 6), checked_pairs=checked)


def merge_gated_clusters(
        config: BandedGatingConfig,
        labels: np.ndarray,
        composite_X: np.ndarray,
        target_X: t.Optional[np.ndarray],
        action_X: t.Optional[np.ndarray],
        cluster_monitor: t.Optional[dict] = None,
) -> t.Tuple[np.ndarray, t.List[dict]]:
    """对过分割聚类结果执行门控驱动的簇对合并（方案 3.3(c) 两步近似）。

    labels: 聚类标签（-1 噪声不参与）；composite_X: 全量拼接向量（行序与 labels 一致）；
    target_X/action_X: 全量分字段向量（可为 None，灰度带内全部按宁分不合处理）。
    返回 (新标签, 裁决记录列表)。

    合并规则：只合并"过分割本应同簇"的对——即两簇中心相似度落在
    [s_low 之上] 且通过门控（free_pass 或 gated_pass）。合并按中心相似度
    从高到低贪心执行，已合并的簇不重复参与（避免链式漂移：
    每次 union 只发生在门控对直接验证过的簇对之间）。
    """
    labels = np.asarray(labels)
    valid_mask = labels != -1
    unique_ids = sorted(set(labels[valid_mask].tolist()))
    if len(unique_ids) < 2:
        return labels, []

    # 簇成员矩阵索引
    members: t.Dict[int, t.List[int]] = {cid: np.where(labels == cid)[0].tolist() for cid in unique_ids}

    comp_mats = {cid: np.atleast_2d(composite_X[members[cid]]) if len(members[cid]) else np.zeros((0, composite_X.shape[1] if composite_X.ndim > 1 else 1), dtype=np.float32) for cid in unique_ids}
    tgt_mats = {cid: (np.atleast_2d(target_X[members[cid]]) if target_X is not None and len(members[cid]) else None) for cid in unique_ids}
    act_mats = {cid: (np.atleast_2d(action_X[members[cid]]) if action_X is not None and len(members[cid]) else None) for cid in unique_ids}

    # 候选簇对：全部两两过 gate（桶内簇数有限，O(k²) 可控）
    verdicts: t.List[tuple] = []
    for x in range(len(unique_ids)):
        for y in range(x + 1, len(unique_ids)):
            cid_a, cid_b = unique_ids[x], unique_ids[y]
            v = gate_cluster_pair(
                config, comp_mats[cid_a], comp_mats[cid_b],
                tgt_mats[cid_a], tgt_mats[cid_b],
                act_mats[cid_a], act_mats[cid_b],
            )
            verdicts.append((cid_a, cid_b, v))

    # 指标落盘（方案第五章：灰度带命中数 / 拦截数 / 通过率 / 免检占比）
    if cluster_monitor is not None:
        free_pass = sum(1 for _, _, v in verdicts if v.band == "free_pass")
        free_block = sum(1 for _, _, v in verdicts if v.band == "free_block")
        gated_pass = sum(1 for _, _, v in verdicts if v.band == "gated_pass")
        gated_block = sum(1 for _, _, v in verdicts if v.band == "gated_block")
        gated_error = sum(1 for _, _, v in verdicts if v.band == "gated_error")
        gated_total = gated_pass + gated_block + gated_error
        cluster_monitor.update({
            "gate_free_pass_pairs": free_pass,
            "gate_free_block_pairs": free_block,
            "gate_gated_pass_pairs": gated_pass,
            "gate_gated_block_pairs": gated_block,
            "gate_gated_error_pairs": gated_error,
            "gate_gated_pass_ratio": round(gated_pass / gated_total, 4) if gated_total else None,
        })

    # 按中心相似度从高到低贪心合并（仅 free_pass / gated_pass 参与）
    allowed = sorted(
        [(v.center_sim, a, b) for a, b, v in verdicts if v.allowed],
        key=lambda x: -x[0],
    )

    # union-find 贪心：新簇 root 只允许通过已验证对合并
    root: t.Dict[int, int] = {cid: cid for cid in unique_ids}

    def find(x: int) -> int:
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    decision_log = []
    for sim, cid_a, cid_b in allowed:
        ra, rb = find(cid_a), find(cid_b)
        if ra == rb:
            decision_log.append({"merge": False, "reason": "already_merged",
                                 "pair": [cid_a, cid_b], "center_sim": sim})
            continue
        root[rb] = ra
        decision_log.append({"merge": True, "reason": "gate_passed",
                             "pair": [cid_a, cid_b], "center_sim": sim})

    # 重排标签：root -> 连续新编号（按成员数从大到小，保持稳定）
    group_members: t.Dict[int, t.List[int]] = defaultdict(list)
    for cid in unique_ids:
        group_members[find(cid)].extend(members[cid])
    sorted_groups = sorted(group_members.items(), key=lambda kv: -len(kv[1]))
    new_label_by_root = {r: i for i, (r, _) in enumerate(sorted_groups)}

    new_labels = labels.copy()
    for r, idx_list in group_members.items():
        for idx in idx_list:
            new_labels[idx] = new_label_by_root[r]

    return new_labels, decision_log
