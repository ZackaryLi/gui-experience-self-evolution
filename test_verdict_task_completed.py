# -*- coding: utf-8 -*-
"""trajectory_routing_pipeline.verdict.resolve_task_completed 的单元测试。

背景：Verdict 分流曾用 verdict.get('task_completed', "None") == "None" 做合法性校验，
该写法只能拦住"键缺失"；当 LLM 输出 JSON null（解析为 Python None）时会绕过校验，
并因 "None is False" 不成立而落入 success 分支，把完成状态不明的轨迹误判为成功、
污染 IND 经验池。resolve_task_completed 改为显式三态判定，本测试锁住该行为。

说明：trajectory_routing_pipeline/__init__.py 会在 import 时连带加载 pipeline.py
并触发从当前工作目录读取 config.json（仓库无该文件），因此这里通过 importlib
按文件路径直接加载 verdict.py，绕开包级 __init__。
"""

import importlib.util
import unittest
from pathlib import Path


def _load_verdict_module():
    module_path = Path(__file__).resolve().parent / "trajectory_routing_pipeline" / "verdict.py"
    spec = importlib.util.spec_from_file_location("trp_verdict_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verdict_mod = _load_verdict_module()
resolve_task_completed = verdict_mod.resolve_task_completed


class TestResolveTaskCompleted(unittest.TestCase):
    """resolve_task_completed 三态判定行为"""

    # ---------- 正常布尔输出：直接信任 ----------

    def test_true_routes_to_success(self):
        self.assertEqual(resolve_task_completed({"task_completed": True}), (True, None))

    def test_false_routes_to_failure(self):
        self.assertEqual(resolve_task_completed({"task_completed": False}), (False, None))

    # ---------- 核心回归用例：JSON null 不得落入成功 ----------

    def test_json_null_is_rejected_not_success(self):
        """旧实现下 None 会绕过校验并落入 success 分支，这是本次修复的核心"""
        task_completed, reason = resolve_task_completed({"task_completed": None})
        self.assertIsNone(task_completed)
        self.assertIsNotNone(reason)
        # 明确不允许出现"默认成功"的取值
        self.assertNotEqual(task_completed, True)

    def test_missing_key_is_rejected(self):
        task_completed, reason = resolve_task_completed({})
        self.assertIsNone(task_completed)
        self.assertIsNotNone(reason)

    # ---------- completion_status 文本兜底 ----------

    def test_fallback_completed_recovers_true(self):
        """LLM 漏输出布尔字段但 completion_status=completed 时可恢复，避免轨迹报废"""
        self.assertEqual(
            resolve_task_completed({"task_completed": None, "completion_status": "completed"}),
            (True, None),
        )

    def test_fallback_failed_recovers_false(self):
        self.assertEqual(
            resolve_task_completed({"task_completed": None, "completion_status": "failed"}),
            (False, None),
        )

    def test_fallback_status_with_valid_bool_takes_bool(self):
        """布尔字段存在时以布尔为准，completion_status 不参与判定"""
        self.assertEqual(
            resolve_task_completed({"task_completed": False, "completion_status": "completed"}),
            (False, None),
        )

    def test_uncertain_is_rejected_with_explicit_reason(self):
        """uncertain 是合法判定结果但不可用于 IND/OOD 分流，需带明确原因落盘"""
        task_completed, reason = resolve_task_completed(
            {"task_completed": None, "completion_status": "uncertain"}
        )
        self.assertIsNone(task_completed)
        self.assertIn("uncertain", reason)

    # ---------- 非法/脏输出防御 ----------

    def test_non_bool_value_is_rejected(self):
        for bad_value in ("true", "failed", 1, 0, [], [True]):
            with self.subTest(value=bad_value):
                task_completed, reason = resolve_task_completed(
                    {"task_completed": bad_value}
                )
                self.assertIsNone(task_completed, f"非法值 {bad_value!r} 不应产生判定结果")
                self.assertIsNotNone(reason)

    def test_non_dict_verdict_is_rejected(self):
        for bad_verdict in (None, [], "completed", [ {"task_completed": True} ]):
            with self.subTest(verdict=bad_verdict):
                task_completed, reason = resolve_task_completed(bad_verdict)
                self.assertIsNone(task_completed)
                self.assertIsNotNone(reason)
                self.assertIn("结构异常", reason)

    def test_empty_verdict_is_rejected(self):
        task_completed, reason = resolve_task_completed({})
        self.assertIsNone(task_completed)
        self.assertIsNotNone(reason)


if __name__ == "__main__":
    unittest.main()
