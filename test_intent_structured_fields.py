# -*- coding: utf-8 -*-
"""意图提取结构化字段落盘的单元测试。

背景（INTENT_CLUSTERING_IMPROVEMENT_PLAN.md 3.1 第一点）：
LLM 意图提取实际输出 core_action/target/attributes/core_intent 四个字段，
旧实现只保留汇总短语 core_intent，判别信息（如"修改昵称"与"修改个性签名"
的差异在 target 字段中）在最后一公里被丢弃。本测试锁住三个结构化字段的
全量落盘行为，供后续复合表征 Embedding（方案 3.2）使用。

测试基础设施说明：与 test_verdict_text_and_steps.py 相同——pipeline.py 在
import 时从 cwd 读取 config.json（仓库无该文件），llm.py 依赖的 aiohttp 在
部分环境未安装，因此 stub aiohttp + dummy config 后加载真实模块，结束恢复现场。
"""

import asyncio
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent


class _FakeLLM:
    """假 LLM：按 prompt 类型返回预设的批量 JSON 响应，不访问真实服务"""

    def __init__(self):
        # 意图提取响应：两条指令，一条带完整结构化字段，一条 attributes 非法
        self.intent_response = json.dumps([
            {
                "id": "traj_0",
                "core_action": "修改",
                "target": "昵称",
                "attributes": ["相册最新照片"],
                "core_intent": "修改昵称"
            },
            {
                "id": "traj_1",
                "core_action": "修改",
                "target": "个性签名",
                "attributes": "非法字段-应为列表",
                "core_intent": "修改个性签名"
            }
        ], ensure_ascii=False)
        # 分类响应：仅 id + category，与本测试断言无关但必须可解析
        self.category_response = json.dumps([
            {"id": "traj_0", "category": "应用设置配置"},
            {"id": "traj_1", "category": "应用设置配置"}
        ], ensure_ascii=False)

    async def ainvoke(self, prompt, count_usage=False):
        # 批量意图 prompt 含 "Core Action"（提取规范），分类 prompt 含 "预定义类别列表"
        if "Core Action" in prompt:
            return self.intent_response
        return self.category_response


class TestIntentStructuredFields(unittest.TestCase):
    """_process_batch_classify_and_intent：core_action/target/attributes 全量落盘"""

    @classmethod
    def setUpClass(cls):
        cls._tmp_dir = tempfile.mkdtemp(prefix="trp_intent_test_")
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

    def _run_batch(self):
        """构造 pipeline 实例并注入 FakeLLM 跑一个 batch"""
        p = self.pipeline_mod.TrajectoryRoutingPipeline(
            os.path.join(self._tmp_dir, "intent_test_out")
        )
        p.llm = _FakeLLM()
        batch_data = [
            {"cache_key": "k0", "app": "微信", "instruction": "帮我改一下昵称"},
            {"cache_key": "k1", "app": "微信", "instruction": "把个性签名改成热爱生活"},
        ]
        result = asyncio.run(p._process_batch_classify_and_intent(batch_data))
        return result

    def test_structured_fields_persisted(self):
        """core_action/target/attributes 必须随 core_intent 一起写回轨迹条目"""
        batch = self._run_batch()
        by_instruction = {t["instruction"]: t for t in batch}

        nickname = by_instruction["帮我改一下昵称"]
        self.assertEqual(nickname["intent"], "修改昵称")
        self.assertEqual(nickname["core_action"], "修改")
        self.assertEqual(nickname["target"], "昵称")
        self.assertEqual(nickname["attributes"], ["相册最新照片"])

    def test_fields_survive_jsonl_serialization(self):
        """结构化字段必须可完整序列化进 global_instruction_cache.jsonl
        （写入协程直接 json.dumps 整个 item，字段需为纯 JSON 类型）"""
        batch = self._run_batch()
        for item in batch:
            serialized = json.dumps(item, ensure_ascii=False)
            restored = json.loads(serialized)
            for key in ("core_action", "target", "attributes", "intent"):
                self.assertIn(key, restored)
        nickname = [t for t in batch if t["instruction"] == "帮我改一下昵称"][0]
        self.assertEqual(json.loads(json.dumps(nickname))["attributes"], ["相册最新照片"])

    def test_invalid_attributes_degrades_to_empty_list(self):
        """LLM 输出非法 attributes（非列表）时降级为空列表，不抛异常"""
        batch = self._run_batch()
        signature = [t for t in batch if t["instruction"] == "把个性签名改成热爱生活"][0]
        self.assertEqual(signature["intent"], "修改个性签名")
        self.assertEqual(signature["core_action"], "修改")
        self.assertEqual(signature["target"], "个性签名")
        self.assertEqual(signature["attributes"], [])

    def test_missing_fields_default_to_empty(self):
        """LLM 漏输出结构化字段时降级为空值，core_intent 行为保持不变"""
        fake = _FakeLLM()
        fake.intent_response = json.dumps([
            {"id": "traj_0", "core_intent": "只有汇总短语"}
        ], ensure_ascii=False)
        p = self.pipeline_mod.TrajectoryRoutingPipeline(
            os.path.join(self._tmp_dir, "intent_test_out2")
        )
        p.llm = fake
        batch = [{"cache_key": "k0", "app": "抖音", "instruction": "签到"}]
        result = asyncio.run(p._process_batch_classify_and_intent(batch))
        self.assertEqual(result[0]["intent"], "只有汇总短语")
        self.assertEqual(result[0]["core_action"], "")
        self.assertEqual(result[0]["target"], "")
        self.assertEqual(result[0]["attributes"], [])


if __name__ == "__main__":
    unittest.main()
