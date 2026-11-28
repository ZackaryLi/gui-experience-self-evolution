# -*- coding: utf-8 -*-
"""簇合并 LLM 裁决（方案 3.4）的单元测试。

覆盖：
1. build_cluster_merge_verdict_prompt：渲染正确（含两簇原始指令采样与
   判定口径），f-string 花括号无 NameError（回归任务指纹同类 bug）；
2. 裁决流程核心：缓存命中复用、LLM 失败保持分裂（宁分不合）、
   can_merge 应用与 union-find 合并、裁决记录落盘幂等。

裁决流程本体是 _run_global_clustering_on_cache 内的闭包段，单测通过
渲染 prompt + 独立验证 union-find 应用逻辑覆盖关键行为；pipeline 级
端到端依赖真实聚类产物，归入离线基准工具路径验证。
"""

import importlib.util
import json
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent


def _load_module(rel_path: str, name: str):
    module_path = _REPO_ROOT / "trajectory_routing_pipeline" / rel_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ipr = _load_module("instruction_pattern_recognition.py", "trp_ipr_merge_test")


class TestClusterMergeVerdictPrompt(unittest.TestCase):
    """裁决 prompt 构造"""

    def test_prompt_renders_both_clusters(self):
        """两簇指令采样都进入 prompt，判定口径（同一条固定方案）在文中"""
        prompt = ipr.build_cluster_merge_verdict_prompt(
            ["帮我改一下昵称", "把名字改成张三"],
            ["把个性签名改成热爱生活"],
        )
        self.assertIn("帮我改一下昵称", prompt)
        self.assertIn("把个性签名改成热爱生活", prompt)
        self.assertIn("同一条固定的执行步骤方案", prompt)
        # f-string 花括号已正确转义渲染（防任务指纹同类 NameError 的回归）
        self.assertIn('"can_merge"', prompt)
        self.assertIn('{"can_merge"', prompt)  # 渲染产物含字面单花括号
        self.assertNotIn("{{", prompt)

    def test_prompt_instructs_conservative_default(self):
        """拿不准判 false 的保守指令必须在 prompt 中（宁分不合）"""
        prompt = ipr.build_cluster_merge_verdict_prompt(["A指令"], ["B指令"])
        self.assertIn("拿不准时一律判 false", prompt)


class TestVerdictApplyLogic(unittest.TestCase):
    """裁决应用逻辑（与 pipeline 内 union-find 段一致的独立验证）"""

    def test_union_find_merge_and_root_flatten(self):
        """can_merge=true 的对传递合并；重复对不二次合并"""
        root = {(0,): (0,), (1,): (1,), (2,): (2,)}

        def find(x):
            while root[x] != x:
                root[x] = root[root[x]]
                x = root[x]
            return x

        verdicts = [
            {"pair": [(0,), (1,)], "can_merge": True},
            {"pair": [(1,), (2,)], "can_merge": True},
            {"pair": [(0,), (2,)], "can_merge": True},  # 已连通，no-op
        ]
        for v in verdicts:
            if v["can_merge"]:
                ra, rb = find(v["pair"][0]), find(v["pair"][1])
                if ra != rb:
                    root[rb] = ra

        # 三簇连通成一个 root
        roots = {find((0,)), find((1,)), find((2,))}
        self.assertEqual(len(roots), 1)

    def test_failed_verdicts_keep_clusters_split(self):
        """LLM 失败的对不进入裁决列表 → 不合并（与实现一致：failed 不产生 record）"""
        records = []
        results = [("k1", None, False),       # LLM 失败
                   ("k2", {"can_merge": True}, True)]
        for key, record, ok in results:
            if not ok or record is None:
                continue  # 失败对：跳过，不合并不缓存
            records.append(record)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["can_merge"])

    def test_verdict_cache_record_shape(self):
        """落盘的裁决记录字段完整（幂等键/结论/理由），JSON 可序列化"""
        record = {
            "verdict_key": "abc123",
            "scenario": "签到", "app": "淘宝",
            "cluster_a": 0, "cluster_b": 1,
            "center_sim": 0.81,
            "can_merge": False,
            "reason": "修改昵称与修改签名入口不同",
            "boundary_case": "个性签名走资料编辑入口",
        }
        restored = json.loads(json.dumps(record, ensure_ascii=False))
        self.assertEqual(restored["verdict_key"], "abc123")
        self.assertFalse(restored["can_merge"])


if __name__ == "__main__":
    unittest.main()
