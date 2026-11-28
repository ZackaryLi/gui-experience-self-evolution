# -*- coding: utf-8 -*-
"""任务卡式簇命名（方案 3.5）的单元测试。

覆盖：
1. build_intent_summary_prompt：任务卡三字段输出格式渲染、原始指令
   采样块进入 prompt、无 instruction_samples 时兼容旧调用；
2. parse_task_card_response：合法 JSON 解析、markdown 包裹兜底、
   json-repair 修复残缺输出、name 缺失返回 None（触发调用方回退）、
   scope/exclusions 缺失降级为空串。
"""

import importlib.util
import json
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent


def _load_module():
    module_path = _REPO_ROOT / "trajectory_routing_pipeline" / "instruction_pattern_recognition.py"
    spec = importlib.util.spec_from_file_location("trp_ipr_card_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ipr = _load_module()


class TestTaskCardPrompt(unittest.TestCase):

    def test_prompt_contains_three_fields(self):
        prompt = ipr.build_intent_summary_prompt(
            "- 修改昵称\n- 修改名字",
            ["帮我改一下昵称", "把名字改成张三"],
        )
        self.assertIn('"name"', prompt)
        self.assertIn('"scope"', prompt)
        self.assertIn('"exclusions"', prompt)
        self.assertIn("排他边界", prompt)

    def test_instruction_samples_rendered(self):
        """原始指令采样进入 prompt（簇内混任务时的保真依据）"""
        prompt = ipr.build_intent_summary_prompt(
            "- 修改昵称",
            ["帮我改一下昵称", "微信名字改叫张三"],
        )
        self.assertIn("帮我改一下昵称", prompt)
        self.assertIn("微信名字改叫张三", prompt)
        self.assertIn("原始用户指令采样", prompt)

    def test_backward_compatible_without_samples(self):
        """不传 instruction_samples 时（旧调用方式）prompt 仍完整可用"""
        prompt = ipr.build_intent_summary_prompt("- 执行签到任务")
        self.assertIn("执行签到任务", prompt)
        self.assertNotIn("原始用户指令采样", prompt)


class TestParseTaskCard(unittest.TestCase):

    def test_parse_valid_json(self):
        res = ipr.parse_task_card_response(
            '{"name": "修改昵称", "scope": "在个人资料编辑页修改账号名称", "exclusions": "不含修改个性签名"}')
        self.assertEqual(res["name"], "修改昵称")
        self.assertEqual(res["scope"], "在个人资料编辑页修改账号名称")
        self.assertEqual(res["exclusions"], "不含修改个性签名")

    def test_parse_markdown_wrapped(self):
        """markdown 代码块包裹的输出可解析（结构切片兜底）"""
        res = ipr.parse_task_card_response(
            '```json\n{"name": "执行签到任务", "scope": "在会员福利页领取签到奖励", "exclusions": ""}\n```')
        self.assertEqual(res["name"], "执行签到任务")
        self.assertEqual(res["exclusions"], "")

    def test_parse_broken_json_repaired(self):
        """残缺 JSON（尾逗号）由 json-repair 修复后可解析"""
        res = ipr.parse_task_card_response(
            '{"name": "搜索指定商品", "scope": "在电商搜索框检索目标商品", "exclusions": "不含下单购买",}')
        self.assertEqual(res["name"], "搜索指定商品")
        self.assertEqual(res["exclusions"], "不含下单购买")

    def test_missing_name_returns_none(self):
        """name 为空/缺失返回 None——调用方将回退旧纯文本清洗逻辑"""
        self.assertIsNone(ipr.parse_task_card_response('{"scope": "只有范围没有名字"}'))
        self.assertIsNone(ipr.parse_task_card_response('{"name": ""}'))
        self.assertIsNone(ipr.parse_task_card_response("完全不是JSON的输出"))

    def test_missing_optional_fields_degrade_to_empty(self):
        """scope/exclusions 缺失降级为空串（name 是唯一硬性字段）"""
        res = ipr.parse_task_card_response('{"name": "播放音乐"}')
        self.assertEqual(res["name"], "播放音乐")
        self.assertEqual(res["scope"], "")
        self.assertEqual(res["exclusions"], "")

    def test_name_quotes_cleaned(self):
        """name 中的引号被清洗（旧逻辑兼容）"""
        res = ipr.parse_task_card_response(
            json.dumps({"name": '"修改昵称"', "scope": "", "exclusions": ""}, ensure_ascii=False))
        self.assertEqual(res["name"], "修改昵称")


if __name__ == "__main__":
    unittest.main()
