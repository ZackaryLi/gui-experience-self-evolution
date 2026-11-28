# -*- coding: utf-8 -*-
"""带宽式门控（方案 3.3）的单元测试。

intent_gating.py 只依赖 numpy（无 config.json / aiohttp 依赖），可直接
import 测试，无需测试基础设施 stub。

覆盖：
1. gate_cluster_pair 三档分流：免检 / 免拆 / 灰度带门控（AND 一票否决）；
2. 分字段向量缺失时灰度带内宁分不合（gated_error）；
3. merge_gated_clusters：门控通过簇对合并、链式漂移防护、标签重排、监控计数。
"""

import importlib.util
import unittest
from pathlib import Path

import numpy as np


def _load_gating_module():
    module_path = Path(__file__).resolve().parent / "trajectory_routing_pipeline" / "intent_gating.py"
    spec = importlib.util.spec_from_file_location("trp_gating_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gating = _load_gating_module()
BandedGatingConfig = gating.BandedGatingConfig
gate_cluster_pair = gating.gate_cluster_pair
merge_gated_clusters = gating.merge_gated_clusters


def _unit_vec(angle: float, dim: int = 8) -> np.ndarray:
    """确定性单位向量：angle 控制方向"""
    v = np.zeros(dim, dtype=np.float32)
    v[0] = np.cos(angle)
    v[1] = np.sin(angle)
    return v


class TestGateClusterPair(unittest.TestCase):
    """三档分流与 AND 门控"""

    def setUp(self):
        self.config = BandedGatingConfig(s_high=0.93, s_low=0.80, t_target=0.80, t_action=0.70)

    def test_free_pass_band(self):
        """中心相似度 ≥ S_high：免检直接允许合并"""
        a = np.stack([_unit_vec(0.0), _unit_vec(0.01)])
        b = np.stack([_unit_vec(0.02), _unit_vec(0.03)])
        v = gate_cluster_pair(self.config, a, b, a, b, a, b)
        self.assertTrue(v.allowed)
        self.assertEqual(v.band, "free_pass")
        self.assertGreaterEqual(v.center_sim, 0.93)

    def test_free_block_band(self):
        """中心相似度 ≤ S_low：直接互斥，不进合并流程"""
        a = np.stack([_unit_vec(0.0)])
        b = np.stack([_unit_vec(np.pi / 2)])  # 正交，相似度 0
        v = gate_cluster_pair(self.config, a, b, a, b, a, b)
        self.assertFalse(v.allowed)
        self.assertEqual(v.band, "free_block")

    def test_gated_pass_all_pairs_consistent(self):
        """灰度带内：全部跨簇样本对双路达标 → 允许合并"""
        # 中心相似度约 cos(0.25rad)≈0.97 太高；取约 cos(0.35)≈0.94 仍高于 0.93，
        # 取 0.85 量级：两簇中心角度差约 0.55 rad
        a = np.stack([_unit_vec(0.0), _unit_vec(0.1)])
        b = np.stack([_unit_vec(0.5), _unit_vec(0.6)])
        # 分字段向量完全一致（同任务近邻场景：target/action 都对齐）
        v = gate_cluster_pair(self.config, a, b,
                              np.ones((2, 8), dtype=np.float32), np.ones((2, 8), dtype=np.float32),
                              np.ones((2, 8), dtype=np.float32), np.ones((2, 8), dtype=np.float32))
        # 中心相似度须落在灰度带内
        self.assertLess(v.center_sim, self.config.s_high)
        self.assertGreater(v.center_sim, self.config.s_low)
        self.assertTrue(v.allowed)
        self.assertEqual(v.band, "gated_pass")
        self.assertEqual(v.checked_pairs, 4)

    def test_gated_block_target_veto(self):
        """灰度带内：target 一路不达标 → 一票否决（即使 action 达标）"""
        a = np.stack([_unit_vec(0.0), _unit_vec(0.1)])
        b = np.stack([_unit_vec(0.5), _unit_vec(0.6)])
        # target 路：跨簇对正交（不相似）；action 路：完全一致
        tgt_a = np.stack([_unit_vec(0.0), _unit_vec(0.1)])
        tgt_b = np.stack([_unit_vec(np.pi / 2), _unit_vec(np.pi / 2)])
        act = np.ones((2, 8), dtype=np.float32)
        v = gate_cluster_pair(self.config, a, b, tgt_a, tgt_b, act, act)
        self.assertFalse(v.allowed)
        self.assertEqual(v.band, "gated_block")

    def test_gated_error_missing_field_vectors(self):
        """灰度带内分字段向量缺失（旧缓存条目）：宁分不合，标记 gated_error"""
        a = np.stack([_unit_vec(0.0), _unit_vec(0.1)])
        b = np.stack([_unit_vec(0.5), _unit_vec(0.6)])
        v = gate_cluster_pair(self.config, a, b, None, None, None, None)
        self.assertFalse(v.allowed)
        self.assertEqual(v.band, "gated_error")

    def test_empty_cluster_blocked(self):
        """空簇输入直接拒绝"""
        a = np.zeros((0, 8), dtype=np.float32)
        b = np.stack([_unit_vec(0.0)])
        v = gate_cluster_pair(self.config, a, b, None, None, None, None)
        self.assertFalse(v.allowed)


class TestMergeGatedClusters(unittest.TestCase):
    """簇对合并：贪心 union、标签重排、监控计数"""

    def setUp(self):
        self.config = BandedGatingConfig(s_high=0.93, s_low=0.80, t_target=0.80, t_action=0.70)

    def test_merge_gate_passed_pairs(self):
        """门控通过的簇对被合并，标签连续重排"""
        # 构造 4 条样本：A0/A1 与 B0/B1 拼接向量相近但灰度带内，
        # 分字段向量一致 → 合并为一个簇
        comp = np.stack([_unit_vec(0.0), _unit_vec(0.1), _unit_vec(0.5), _unit_vec(0.6)])
        labels = np.array([0, 0, 1, 1])
        field = np.ones((4, 8), dtype=np.float32)
        monitor = {}
        new_labels, decisions = merge_gated_clusters(
            self.config, labels, comp, field, field, cluster_monitor=monitor)
        self.assertEqual(new_labels.tolist(), [0, 0, 0, 0])
        self.assertTrue(any(d["merge"] for d in decisions))
        self.assertIn("gate_gated_pass_pairs", monitor)

    def test_no_merge_when_field_diverges(self):
        """分字段向量分歧 → 门控拦截，簇保持分裂（宁分不合）"""
        comp = np.stack([_unit_vec(0.0), _unit_vec(0.1), _unit_vec(0.5), _unit_vec(0.6)])
        labels = np.array([0, 0, 1, 1])
        # target 路：前两条一个方向、后两条正交 → 跨簇对必有一路不达标
        target = np.stack([_unit_vec(0.0), _unit_vec(0.1),
                           _unit_vec(np.pi / 2), _unit_vec(np.pi / 2 + 0.1)])
        action = np.ones((4, 8), dtype=np.float32)
        monitor = {}
        new_labels, decisions = merge_gated_clusters(
            self.config, labels, comp, target, action, cluster_monitor=monitor)
        # 保持两簇分裂
        self.assertEqual(len(set(new_labels.tolist())), 2)
        self.assertTrue(any(d["merge"] is False for d in decisions) or not decisions)
        self.assertEqual(monitor.get("gate_gated_block_pairs", 0), 1)

    def test_noise_label_preserved(self):
        """-1 噪声标签不参与门控合并，保持原值"""
        comp = np.stack([_unit_vec(0.0), _unit_vec(0.5), _unit_vec(np.pi / 2)])
        labels = np.array([0, 1, -1])
        monitor = {}
        new_labels, _ = merge_gated_clusters(
            self.config, labels, comp, None, None, cluster_monitor=monitor)
        self.assertEqual(new_labels[2], -1)

    def test_free_pass_outranks_gated_in_greedy(self):
        """免检对（更近）优先于灰度带对合并（贪心按相似度降序）"""
        # 三簇：C0-C1 免检级相近，C0-C2 灰度带；免检对先合
        comp = np.stack([_unit_vec(0.0), _unit_vec(0.01),   # C0
                         _unit_vec(0.02),                    # C1
                         _unit_vec(0.6)])                    # C2
        labels = np.array([0, 0, 1, 2])
        field = np.ones((4, 8), dtype=np.float32)
        new_labels, decisions = merge_gated_clusters(
            self.config, labels, comp, field, field)
        # C0 与 C1 免检合并；C2 是否并入取决于其中心相似度是否过带与门控
        self.assertEqual(new_labels[0], new_labels[1])  # C0 成员同簇
        self.assertTrue(any(d["merge"] and d["reason"] == "gate_passed" for d in decisions))


if __name__ == "__main__":
    unittest.main()
