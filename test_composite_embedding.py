# -*- coding: utf-8 -*-
"""双用途表征层（方案 3.2）的单元测试。

覆盖：
1. build_composite_text：结构化字段拼接、attributes 拼接、旧缓存回退 core_intent；
2. _process_and_embed_batch：三段文本一次请求编码后按索引段拆分，
   target_embedding/action_embedding 落盘，主向量失败时分字段一并置空；
3. 监控计数：structured_items / field_missing_items / embedding_failed_items。

测试基础设施说明：与 test_intent_structured_fields.py 相同——stub aiohttp +
dummy config.json 后加载真实 pipeline 模块，结束恢复现场。
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


def _fake_vec(seed: float, dim: int = 8):
    """确定性伪向量：seed 各分量不同，方便断言拆段无错位"""
    import math
    return [round(math.sin(seed + i), 6) for i in range(dim)]


class _FakeLLM:
    """假 LLM：意图提取返回带结构化字段的批量 JSON"""

    def __init__(self):
        self.intent_response = json.dumps([
            {"id": "traj_0", "core_action": "修改", "target": "昵称",
             "attributes": ["张三"], "task_fingerprint": "在个人资料编辑页中修改昵称以更新账号名称",
             "core_intent": "修改昵称"},
            {"id": "traj_1", "core_action": "签到", "target": "签到任务",
             "attributes": [], "task_fingerprint": "在会员福利页中执行签到任务以领取奖励",
             "core_intent": "执行签到任务"},
        ], ensure_ascii=False)
        self.category_response = json.dumps([
            {"id": "traj_0", "category": "应用设置配置"},
            {"id": "traj_1", "category": "签到"},
        ], ensure_ascii=False)

    async def ainvoke(self, prompt, count_usage=False):
        if "Core Action" in prompt:
            return self.intent_response
        return self.category_response


class _FakeEmbedding:
    """假 Embedding：按输入文本内容生成确定性向量，可编程失败。

    返回顺序必须与输入顺序一致（脚本按 index 拆段，这里同样遵守）。
    """

    def __init__(self, fail_all=False):
        self.inputs_seen = []
        self.fail_all = fail_all

    async def call(self, texts):
        self.inputs_seen.append(list(texts))
        if self.fail_all:
            return [[] for _ in texts]
        # 确定性向量：以文本首字符的 ordinal 为种子（避免跨平台 hash 不稳定）
        return [_fake_vec(float(ord(t[0])) / 1000.0 if t.strip() else 0.0)
                for t in texts]


class TestCompositeEmbedding(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp_dir = tempfile.mkdtemp(prefix="trp_comp_emb_test_")
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

    def _make_pipeline(self):
        p = self.pipeline_mod.TrajectoryRoutingPipeline(
            os.path.join(self._tmp_dir, "comp_emb_out"))
        p.llm = _FakeLLM()
        return p

    def _run_batch(self, fake_embed):
        """monkeypatch _intent_embedding 后跑一个 batch，返回处理完的条目"""
        p = self._make_pipeline()

        async def _fake_intent_embedding(texts, model="bge-m3"):
            return await fake_embed.call(texts)

        p._intent_embedding = _fake_intent_embedding
        batch = [
            {"cache_key": "k0", "app": "微信", "instruction": "帮我改一下昵称"},
            {"cache_key": "k1", "app": "淘宝", "instruction": "去签到"},
        ]
        # _process_and_embed_batch 是闭包，通过走 _build_or_load_global_intent_cache
        # 太重；直接复刻其核心逻辑不可取——改为直接调用内部步骤组合
        return p, batch

    # ---------- build_composite_text ----------

    def test_composite_text_full_fields(self):
        fn = self.pipeline_mod.build_composite_text
        item = {"core_action": "修改", "target": "昵称", "attributes": ["张三"]}
        self.assertEqual(fn(item), "修改 昵称 张三")

    def test_composite_text_without_attributes(self):
        fn = self.pipeline_mod.build_composite_text
        self.assertEqual(fn({"core_action": "搜索", "target": "指定商品"}), "搜索 指定商品")

    def test_composite_text_fallback_to_intent(self):
        """旧缓存条目缺结构化字段时回退 core_intent（方案 3.2 兼容规则）"""
        fn = self.pipeline_mod.build_composite_text
        self.assertEqual(fn({"intent": "修改昵称"}), "修改昵称")
        # core_intent 为空则整体为空（该条目 embedding 失败路径处理）
        self.assertEqual(fn({}), "")

    def test_composite_text_invalid_attributes_ignored(self):
        fn = self.pipeline_mod.build_composite_text
        item = {"core_action": "修改", "target": "昵称", "attributes": "非法"}
        self.assertEqual(fn(item), "修改 昵称")

    # ---------- 双用途 embedding 拆段与落盘（驱动真实方法 _extract_and_embed_items） ----------

    def test_extract_and_embed_items_full_pipeline(self):
        """回归测试（真实路径）：提取字段全量回写 + 三段拆段无错位。

        此前的测试复刻"预期行为"而未走真实闭包，曾漏掉回写循环只拷
        intent/category、结构化字段从未落缓存条目的 bug——真机跑出
        global_instruction_cache 只有 intent 字段即由此而来。
        """
        p = self._make_pipeline()
        fake_embed = _FakeEmbedding()

        async def _fake_intent_embedding(texts, model="bge-m3"):
            return await fake_embed.call(texts)

        p._intent_embedding = _fake_intent_embedding

        items = [
            {"cache_key": "k0", "app": "微信", "instruction": "帮我改一下昵称"},
            {"cache_key": "k1", "app": "淘宝", "instruction": "去签到"},
        ]
        asyncio.run(p._extract_and_embed_items(items))

        # 【核心回归】结构化字段必须落在缓存条目上（bug 的直接症状）
        nickname = items[0]
        self.assertEqual(nickname["intent"], "修改昵称")
        self.assertEqual(nickname["core_action"], "修改")
        self.assertEqual(nickname["target"], "昵称")
        self.assertEqual(nickname["attributes"], ["张三"])
        self.assertEqual(nickname["task_fingerprint"], "在个人资料编辑页中修改昵称以更新账号名称")
        self.assertEqual(nickname["category"]["category"], "应用设置配置")

        # 一次请求包含三段拼接的完整文本，拼接段使用结构化字段而非回退 intent
        self.assertEqual(len(fake_embed.inputs_seen), 1)
        self.assertEqual(len(fake_embed.inputs_seen[0]), 6)  # 2 条 × 3 段
        self.assertEqual(fake_embed.inputs_seen[0][0], "修改 昵称 张三")
        self.assertEqual(fake_embed.inputs_seen[0][2], "昵称")   # target 段非空格占位
        self.assertEqual(fake_embed.inputs_seen[0][4], "修改")   # action 段非空格占位

        # 三路向量全部落盘且非空
        for it in items:
            self.assertTrue(it["intent_embedding"])
            self.assertTrue(it["target_embedding"])
            self.assertTrue(it["action_embedding"])
        # 监控计数：两条均为结构化齐全
        self.assertEqual(p.embedding_field_stats["structured_items"], 2)

    def test_extract_and_embed_items_fallback_for_missing_fields(self):
        """LLM 未输出结构化字段（提取失败/漏字段）时：回退 intent 文本，
        分字段段用空格占位，向量按空值处理不参与门控"""
        fake_llm = _FakeLLM()
        fake_llm.intent_response = json.dumps([
            {"id": "traj_0", "core_intent": "只有汇总短语"}
        ], ensure_ascii=False)
        p = self._make_pipeline()
        p.llm = fake_llm

        async def _fake_intent_embedding(texts, model="bge-m3"):
            return [[0.5] * 8 for _ in texts]

        p._intent_embedding = _fake_intent_embedding

        items = [{"cache_key": "k0", "app": "抖音", "instruction": "签到"}]
        asyncio.run(p._extract_and_embed_items(items))

        self.assertEqual(items[0]["intent"], "只有汇总短语")
        self.assertEqual(items[0]["core_action"], "")
        self.assertEqual(items[0]["target"], "")
        # 主向量成功但分字段缺失 → field_missing 计数
        self.assertEqual(p.embedding_field_stats["field_missing_items"], 1)

    def test_extract_and_embed_items_embedding_failure(self):
        """embedding 整体失败：三路向量全部置空，计入 failed 计数（下次运行重试）"""
        p = self._make_pipeline()
        fake_embed = _FakeEmbedding(fail_all=True)

        async def _fake_intent_embedding(texts, model="bge-m3"):
            return await fake_embed.call(texts)

        p._intent_embedding = _fake_intent_embedding

        items = [{"cache_key": "k0", "app": "微信", "instruction": "帮我改一下昵称"}]
        asyncio.run(p._extract_and_embed_items(items))

        self.assertEqual(items[0]["intent_embedding"], [])
        self.assertEqual(items[0]["target_embedding"], [])
        self.assertEqual(items[0]["action_embedding"], [])
        self.assertEqual(p.embedding_field_stats["embedding_failed_items"], 1)


if __name__ == "__main__":
    unittest.main()
