# -*- coding: utf-8 -*-
"""trajectory_routing_pipeline 的 Verdict 输入构建与步骤解析口径测试。

覆盖三项修复：
1. verdict.get_sorted_steps / pipeline.calc_valid_step 步骤解析口径统一：
   旧实现 calc_valid_step 自行排序，遇到 stepX 脏键时 ValueError 被静默吞掉，
   排序退化为字典序（step10 排在 step2 前），与 verdict 侧看到序列不一致。
2. verdict.build_trajectory_text 超长轨迹改为"头部摘要 + 完整尾部"：
   旧实现固定保留前 max_steps 步、丢弃尾部，恰好截掉对完成度判定最关键的
   收尾步骤（是否点了最终确认按钮、停在哪个页面）。
3. pipeline.TrajectoryRoutingPipeline.log 在 Windows GBK 控制台下不丢日志：
   旧实现 print 与文件写入同处一个 try，emoji 触发 UnicodeEncodeError 后
   整条日志完全不落盘。

测试基础设施说明：
- trajectory_routing_pipeline/__init__.py 与 pipeline.py 会在 import 时从当前
  工作目录读取 config.json（仓库无该文件），且 llm.py 依赖的 aiohttp 在部分
  环境未安装。因此这里在临时目录放置 dummy config.json、stub 掉 aiohttp 后
  再 import 真实 pipeline 模块，结束恢复现场，不影响同进程的其他测试。
"""

import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent


def _build_traj(num_steps: int, action_fmt: str = "action_{i}") -> dict:
    """构造 num_steps 步的标准化轨迹，action 各不相同且均为有效动作"""
    traj = {"app": "测试App", "instruction": "测试指令"}
    for i in range(1, num_steps + 1):
        traj[f"step{i}"] = {"action": action_fmt.format(i=i), "thought": "", "manager_action": ""}
    return traj


class _VerdictModuleTestCase(unittest.TestCase):
    """基类：按文件路径加载 verdict.py，绕开包级 __init__ 的 config.json 依赖"""

    @classmethod
    def setUpClass(cls):
        module_path = _REPO_ROOT / "trajectory_routing_pipeline" / "verdict.py"
        spec = importlib.util.spec_from_file_location("trp_verdict_text_under_test", module_path)
        cls.verdict_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.verdict_mod)


class TestGetSortedSteps(_VerdictModuleTestCase):
    """get_sorted_steps：数字序解析，脏键排除"""

    def test_numeric_order_not_lexicographic(self):
        """step10 必须排在 step9 之后，而不是字典序的 step1 与 step2 之间"""
        traj = _build_traj(12)
        steps = self.verdict_mod.get_sorted_steps(traj)
        self.assertEqual([s["step_num"] for s in steps], list(range(1, 13)))

    def test_dirty_step_key_is_excluded(self):
        """stepX 等非数字编号脏键不得进入步骤序列（与旧实现的静默排序失败区分开）"""
        traj = _build_traj(3)
        traj["stepX"] = {"action": "脏数据动作", "thought": "", "manager_action": ""}
        steps = self.verdict_mod.get_sorted_steps(traj)
        self.assertEqual([s["step_num"] for s in steps], [1, 2, 3])

    def test_non_step_keys_ignored(self):
        traj = _build_traj(2)
        traj["instruction"] = "x"
        traj["step_notes"] = {"action": "无关"}
        steps = self.verdict_mod.get_sorted_steps(traj)
        self.assertEqual(len(steps), 2)

    def test_step_data_preserved(self):
        traj = _build_traj(1)
        steps = self.verdict_mod.get_sorted_steps(traj)
        self.assertEqual(steps[0]["action"], "action_1")


class TestBuildTrajectoryText(_VerdictModuleTestCase):
    """build_trajectory_text：超长轨迹保留头部 + 完整尾部"""

    def setUp(self):
        self.fn = self.verdict_mod.build_trajectory_text

    def test_short_trajectory_not_truncated(self):
        """不超限时全部步骤保留、无省略标记"""
        text = self.fn(_build_traj(10), max_steps=80)
        self.assertIn("[step10]", text)
        self.assertNotIn("中间省略", text)
        self.assertIn("action_10", text)

    def test_long_trajectory_keeps_head_and_full_tail(self):
        """超限时尾部必须完整保留——完成度判定依赖最终状态"""
        text = self.fn(_build_traj(100), max_steps=80)
        # 头部保留
        self.assertIn("[step1]", text)
        self.assertIn("[step20]", text)
        # 尾部完整保留（最后 60 步 = max_steps 80 - head_keep 20）
        self.assertIn("[step41]", text)
        self.assertIn("[step100]", text)
        # 中间被省略
        self.assertNotIn("[step21]", text)
        self.assertNotIn("[step40]", text)
        # 省略标记存在且数量正确（100 - 20 头部 - 60 尾部 = 20）
        self.assertIn("中间省略 20 步", text)

    def test_boundary_exactly_max_steps(self):
        """恰好等于 max_steps 时不截断"""
        text = self.fn(_build_traj(80), max_steps=80)
        self.assertIn("[step80]", text)
        self.assertNotIn("中间省略", text)

    def test_omitted_step_count_matches(self):
        """省略步数 = 总数 - 头部 - 尾部"""
        text = self.fn(_build_traj(90), max_steps=80, head_keep_steps=20)
        self.assertIn("中间省略 10 步", text)  # 90 - 20 头部 - 60 尾部 = 10
        self.assertIn("[step90]", text)

    def test_custom_head_keep(self):
        """head_keep_steps 可调：头部为 5 时尾部保留 75 步"""
        text = self.fn(_build_traj(100), max_steps=80, head_keep_steps=5)
        self.assertIn("[step5]", text)
        self.assertNotIn("[step6]", text)
        self.assertIn("[step26]", text)
        self.assertIn("[step100]", text)


class TestCalcValidStepAndLog(unittest.TestCase):
    """pipeline 侧：calc_valid_step 口径统一 + log() Windows GBK 兜底。

    通过 dummy config.json + stub aiohttp 加载真实 pipeline 模块，
    测试结束恢复 sys.modules / sys.path / cwd，避免污染同进程其他测试。
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp_dir = tempfile.mkdtemp(prefix="trp_test_")
        # pipeline.py 在 import 时从 cwd 读取 config.json
        with open(os.path.join(cls._tmp_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"embedding_api_key": "dummy_key", "chat_api_key": "dummy_key"}, f)

        cls._orig_cwd = os.getcwd()
        cls._orig_sys_path = list(sys.path)
        cls._orig_modules = {k: sys.modules.get(k) for k in ("aiohttp", "trajectory_routing_pipeline")}

        # stub 未安装的 aiohttp：llm.py 的类型标注（如 -> aiohttp.ClientSession）在
        # 类定义时求值，因此模块对象上需先挂占位属性，运行期不会被真正调用
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
        # 清理包内所有已加载子模块，防止 dummy config 残留到其他测试
        for name in [k for k in sys.modules if k.startswith("trajectory_routing_pipeline")]:
            del sys.modules[name]

    # ---------- calc_valid_step：步骤解析口径统一 ----------

    def test_calc_valid_step_numeric_order_with_dirty_key(self):
        """脏键 stepX 存在时仍按数字序统计，且脏键不计入有效步。

        场景构造：step10 的动作与 step2 完全相同。数字序下 step10 紧跟 step2
        之后，会被连续重复规则过滤 → 2 个有效步。旧实现遇到 stepX 排序抛
        ValueError 被吞，退化为字典序迭代（stepX 会被计入），结果为 3。
        """
        traj = {
            "step1": {"action": "click([1,1])", "thought": "", "manager_action": ""},
            "step2": {"action": "click([2,2])", "thought": "", "manager_action": ""},
            "step10": {"action": "click([2,2])", "thought": "", "manager_action": ""},
            "stepX": {"action": "click([9,9])", "thought": "", "manager_action": ""},
        }
        self.assertEqual(self.pipeline_mod.TrajectoryRoutingPipeline.calc_valid_step(traj), 2)

    def test_calc_valid_step_basic_counting(self):
        traj = _build_traj(7)
        self.assertEqual(self.pipeline_mod.TrajectoryRoutingPipeline.calc_valid_step(traj), 7)

    def test_calc_valid_step_filters_invalid_actions(self):
        traj = {
            "step1": {"action": "do_nothing()", "thought": "", "manager_action": ""},
            "step2": {"action": "等待页面加载", "thought": "", "manager_action": ""},
            "step3": {"action": "", "thought": "", "manager_action": ""},
            "step4": {"action": "", "thought": "", "manager_action": "manager 指令"},
            "step5": {"action": "click([5,5])", "thought": "", "manager_action": ""},
        }
        self.assertEqual(self.pipeline_mod.TrajectoryRoutingPipeline.calc_valid_step(traj), 1)

    # ---------- log()：Windows GBK 控制台兜底 ----------

    class _FakeGbkStdout:
        """模拟 Windows GBK 控制台：emoji 字符 write 时抛 UnicodeEncodeError"""

        encoding = "gbk"
        errors = "strict"

        def __init__(self):
            self.buf = []

        def write(self, s):
            s.encode(self.encoding, errors=self.errors)  # emoji 在此抛 UnicodeEncodeError
            self.buf.append(s)
            return len(s)

        def flush(self):
            pass

    def test_log_file_write_survives_console_encoding_error(self):
        """控制台 print 抛 UnicodeEncodeError 时，日志文件仍必须完整落盘"""
        pipeline_cls = self.pipeline_mod.TrajectoryRoutingPipeline
        save_path = Path(self._tmp_dir) / "log_test_out"
        p = pipeline_cls(save_path)

        fake_stdout = self._FakeGbkStdout()
        orig_stdout = sys.stdout
        sys.stdout = fake_stdout
        try:
            p.log("✅ 带emoji的测试日志消息")
        finally:
            sys.stdout = orig_stdout

        # 文件写入不受控制台失败影响，emoji 完整保留
        log_file = save_path / "run.log"
        self.assertTrue(log_file.exists())
        content = log_file.read_text(encoding="utf-8")
        self.assertIn("✅ 带emoji的测试日志消息", content)

        # 控制台降级输出也发生（替换字符版本），且未抛异常到调用方
        self.assertTrue(len(fake_stdout.buf) > 0)
        self.assertNotIn("✅", "".join(fake_stdout.buf))

    def test_log_normal_console_still_prints(self):
        """正常控制台下行为不变：打印 + 落盘"""
        pipeline_cls = self.pipeline_mod.TrajectoryRoutingPipeline
        save_path = Path(self._tmp_dir) / "log_test_out2"
        p = pipeline_cls(save_path)

        captured = io.StringIO()
        orig_stdout = sys.stdout
        sys.stdout = captured
        try:
            p.log("普通消息")
        finally:
            sys.stdout = orig_stdout

        self.assertIn("普通消息", captured.getvalue())
        self.assertIn("普通消息", (save_path / "run.log").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
