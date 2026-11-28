# -*- coding: utf-8 -*-
"""复合任务标记（方案 A）的单元测试。

背景：复合指令（如"打卡后切账号再打卡"）的 target/core_action 与简单指令
完全相同，门控会错误放行合并。提取层新增 is_compound/sub_intents 显式标记，
下游门控据此绕过自动合并（方案 B）。

覆盖：
1. prompt 渲染：复合任务规范与 is_compound 输出格式在文中，few-shot 的
   ex_6（复合示例）/ex_7（多参数单任务反例）存在；
2. _extract_and_embed_items 真实路径：复合字段回写缓存条目、
   非复合时 sub_intents 强制清空、缺字段降级为 false/[]。
"""

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent


def _load_ipr():
    module_path = _REPO_ROOT / "trajectory_routing_pipeline" / "instruction_pattern_recognition.py"
    spec = importlib.util.spec_from_file_location("trp_ipr_compound_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ipr = _load_ipr()

_COMPOINT_INTENT_RESPONSE = json.dumps([
    {"id": "traj_0", "core_action": "执行", "specific_target": "签到任务",
     "attributes": ["今日打卡", "成就中心", "多账号切换"],
     "task_fingerprint": "在成就中心页中循环执行签到任务以完成多账号打卡",
     "is_compound": True,
     "sub_intents": ["执行签到任务", "切换账号", "执行签到任务", "切换回原账号"],
     "core_intent": "执行签到任务"},
    {"id": "traj_1", "core_action": "执行", "specific_target": "签到任务",
     "attributes": ["每日签到"],
     "task_fingerprint": "在会员福利页中执行签到任务以领取奖励",
     "is_compound": False,
     "sub_intents": [],
     "core_intent": "执行签到任务"},
], ensure_ascii=False)


class TestCompoundPrompt(unittest.TestCase):

    def test_prompt_contains_compound_spec(self):
        prompt = ipr.build_batch_intent_recognition_prompt(
            [{"id": "x", "instruction": "测试"}])
        self.assertIn("Compound Detection", prompt)
        self.assertIn('"is_compound"', prompt)
        self.assertIn('"sub_intents"', prompt)
        self.assertIn("该拆就拆", prompt)

    def test_prompt_contains_compound_fewshot(self):
        """few-shot 含复合示例 ex_6 与多参数单任务反例 ex_7"""
        prompt = ipr.build_batch_intent_recognition_prompt(
            [{"id": "x", "instruction": "测试"}])
        self.assertIn("切换回原来的账号", prompt)          # ex_6 复合输入
        self.assertIn('"is_compound": true', prompt)        # ex_6 复合输出
        self.assertIn("生椰拿铁", prompt)                    # ex_7 多参数单任务
        self.assertIn("大杯", prompt)                        # ex_7 attributes


class TestCompoundFieldWriteback(unittest.TestCase):
    """驱动真实 _extract_and_embed_items 路径验证复合字段落盘"""

    @classmethod
    def setUpClass(cls):
        cls._tmp_dir = tempfile.mkdtemp(prefix="trp_compound_test_")
        with open(os.path.join(cls._tmp_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"embedding_api_key": "dummy_key", "chat_api_key": "dummy_key"}, f)

        cls._orig_cwd = os.getcwd()
        cls._orig_sys_path = list(sys.path)
        cls._orig_modules = {k: sys.modules.get(k) for k in ("aiohttp", "trajectory_routing_pipeline")}

        fake_aiohttp = types.ModuleType("aiohttp")
        for attr in ("ClientSession", "TCPConnector", "ClientTimeout", "ClientError", "TimeoutError"):
            setattr(fake_aiohttp, attr, type(attr, (), {}))
        sys.modules["aiohttp"] = fake_aiohttp

        os.chdir(cls._tmp_dir)
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))

        import trajectory_routing_pipeline.pipeline as trp_pipeline
        cls.pipeline_mod = trp_pipeline

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._orig_cwd)
        sys.path[:] = cls._orig_sys_path
        for name, mod in cls._orig_modules.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        for name in [k for k in sys.modules if k.startswith("trajectory_routing_pipeline")]:
            del sys.modules[name]

    def _run_with(self, intent_response):
        p = self.pipeline_mod.TrajectoryRoutingPipeline(
            os.path.join(self._tmp_dir, "compound_out"))

        class _FakeLLM:
            async def ainvoke(self, prompt, count_usage=False):
                if "Core Action" in prompt:
                    return intent_response
                return json.dumps([{"id": "traj_0", "category": "签到"},
                                   {"id": "traj_1", "category": "签到"}], ensure_ascii=False)

        p.llm = _FakeLLM()

        async def _fake_embed(texts, model="bge-m3"):
            return [[0.1] * 8 for _ in texts]

        p._intent_embedding = _fake_embed
        items = [
            {"cache_key": "k0", "app": "淘宝", "instruction": "打卡后切账号再打卡"},
            {"cache_key": "k1", "app": "淘宝", "instruction": "去签到"},
        ]
        asyncio.run(p._extract_and_embed_items(items))
        return items

    def test_compound_fields_persisted(self):
        items = self._run_with(_COMPOINT_INTENT_RESPONSE)
        compound, simple = items[0], items[1]

        self.assertTrue(compound["is_compound"])
        self.assertEqual(compound["sub_intents"],
                         ["执行签到任务", "切换账号", "执行签到任务", "切换回原账号"])
        # 复合条目的主意图仍按第一个子任务口径（与单任务条目可比）
        self.assertEqual(compound["intent"], "执行签到任务")

        self.assertFalse(simple["is_compound"])
        self.assertEqual(simple["sub_intents"], [])

    def test_noncompound_sub_intents_force_cleared(self):
        """LLM 输出矛盾（is_compound=false 但给了 sub_intents）时以布尔为准清空"""
        bad_response = json.dumps([
            {"id": "traj_0", "core_action": "执行", "specific_target": "签到任务",
             "attributes": [], "task_fingerprint": "", "is_compound": False,
             "sub_intents": ["不该出现的子任务"], "core_intent": "执行签到任务"},
            {"id": "traj_1", "core_action": "执行", "specific_target": "签到任务",
             "attributes": [], "task_fingerprint": "", "is_compound": False,
             "sub_intents": [], "core_intent": "执行签到任务"},
        ], ensure_ascii=False)
        items = self._run_with(bad_response)
        self.assertFalse(items[0]["is_compound"])
        self.assertEqual(items[0]["sub_intents"], [])

    def test_missing_compound_fields_default(self):
        """LLM 漏输出复合字段时降级 false/[]（旧 prompt 输出兼容）"""
        old_style_response = json.dumps([
            {"id": "traj_0", "core_action": "执行", "specific_target": "签到任务",
             "attributes": [], "task_fingerprint": "", "core_intent": "执行签到任务"},
            {"id": "traj_1", "core_action": "执行", "specific_target": "签到任务",
             "attributes": [], "task_fingerprint": "", "core_intent": "执行签到任务"},
        ], ensure_ascii=False)
        items = self._run_with(old_style_response)
        for it in items:
            self.assertFalse(it["is_compound"])
            self.assertEqual(it["sub_intents"], [])

    def test_dirty_compound_values_sanitized(self):
        """is_compound 非布尔（字符串）或 sub_intents 非列表时防御性降级"""
        dirty_response = json.dumps([
            {"id": "traj_0", "core_action": "执行", "specific_target": "签到任务",
             "attributes": [], "task_fingerprint": "", "is_compound": "true",
             "sub_intents": "不是列表", "core_intent": "执行签到任务"},
            {"id": "traj_1", "core_action": "执行", "specific_target": "签到任务",
             "attributes": [], "task_fingerprint": "", "is_compound": False,
             "sub_intents": [], "core_intent": "执行签到任务"},
        ], ensure_ascii=False)
        items = self._run_with(dirty_response)
        self.assertFalse(items[0]["is_compound"])   # "true" 字符串不认
        self.assertEqual(items[0]["sub_intents"], [])


if __name__ == "__main__":
    unittest.main()
