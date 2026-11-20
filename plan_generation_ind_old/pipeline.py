import os
import random
import re
import json
import copy
import asyncio
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from pathlib import Path

from openai import OpenAI
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from .log import log
from .llm import LLMAgent

from .prompt_v2 import (
    build_candidate_golden_path_prompt,
    build_golden_path_prompt,
    build_redundant_prompt,
    build_patch_plan_prompt,
    build_from_scratch_plan_prompt,
    build_plan_diff_prompt,
    build_baseline_plan_selection_prompt,
    build_confidence_review_prompt
)

from .schemas import (
    TrajectoryRecord,
    TrajectoryStep,
    IntentGroup,
    GoldenPathPattern,
    RedundantPattern,
    PlanPhase,
    EfficientPlan
)

from .confidence import (
    ConfidenceConfig,
    compute_stat_score,
    fuse_confidence,
    Calibrator,
    NoopCalibrator,
)

from esi.trajectory_routing_pipeline.pipeline import TrajectoryRoutingPipeline

MAX_LLM_CONCURRENCY = 6
MLOPS_MODEL = "qwen38-27b-vl"
MAX_EMBEDDING_CONCURRENCY = 20
MAX_INPUT_SIZE = 25

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
    EMBEDDING_TOKEN = config["embedding_api_key"]
    CHAT_TOKEN = config["chat_api_key"]


def sanitize_filename(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:180] if len(name) > 180 else name

def semantic_diversity_sampling(
        trajectories: List[Dict[str, Any]],
        target_samples: int,
        instruction_key: str = "instruction",
        random_state: int = 42,
        batch_size: int = 64,
        embedding_token: str = "",
) -> List[Dict[str, Any]]:
    # ---------------- 阶段 1: 前置鲁棒性校验 ----------------
    if not trajectories or target_samples <= 0:
        return []

    total_count = len(trajectories)
    # 如果现有样本量本身就不多于目标数，直接全部返回
    if total_count <= target_samples:
        return list(trajectories)

    # ---------------- 阶段 2: 数据清洗与索引对齐 ----------------
    valid_indices = []
    texts_to_embed = []
    empty_indices = []

    for idx, item in enumerate(trajectories):
        inst = item.get(instruction_key)
        # 校验字段是否存在且非空字符串
        if isinstance(inst, str) and inst.strip():
            valid_indices.append(idx)
            texts_to_embed.append(inst.strip())
        else:
            empty_indices.append(idx)

    # 脏数据兜底：若所有文本均无效，直接随机采样
    if not valid_indices:
        chosen_indices = np.random.RandomState(random_state).choice(
            total_count, size=target_samples, replace=False
        )
        return [trajectories[i] for i in chosen_indices]

    # 若有效文本总量小于等于目标采样数，优先保留所有有效数据，剩余名额补充空数据
    if len(valid_indices) <= target_samples:
        needed_more = target_samples - len(valid_indices)
        additional_indices = empty_indices[:needed_more]
        selected_indices = valid_indices + additional_indices
        return [trajectories[i] for i in selected_indices]

    # ---------------- 阶段 3: 获取 Embedding 向量 ----------------
    # 请确保全局已定义 EMBEDDING_API_KEY，或在此处作为参数传入
    client = OpenAI(
        api_key=embedding_token,
        base_url="http://mlops.huawei.com/mlops-service/api/v2/agentService/v1"
    )

    all_embeddings = []

    # 增加 tqdm 进度条与分批处理
    for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="Generating Embeddings"):
        batch_texts = texts_to_embed[i: i + batch_size]

        response = client.embeddings.create(
            input=batch_texts,
            model="bge-m3"
        )

        # 提取真正的 embedding 向量列表，并追加到总列表中
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    # 将嵌套列表转为 numpy 数组，供聚类算法使用
    embeddings = np.array(all_embeddings, dtype=np.float32)

    if len(embeddings) != len(texts_to_embed):
        raise ValueError("Embedding 函数返回的向量数量与输入文本数量不一致")

    # ---------------- 阶段 4: 聚类与中心代表点提取 ----------------
    # 确定实际聚类簇数（不得大于唯一有效样本数）
    k = min(target_samples, len(valid_indices))

    # 使用 MiniBatchKMeans，兼顾速度与内存消耗
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        batch_size=min(1024, len(embeddings)),
        n_init=3
    )
    kmeans.fit(embeddings)
    cluster_centers = kmeans.cluster_centers_

    # 为每个聚类中心找到距离最近的样本点索引（基于欧氏距离最近邻）
    closest_indices, _ = pairwise_distances_argmin_min(cluster_centers, embeddings)

    # 去重处理（避免极少数情况下两个中心最近点为同一个样本）
    selected_internal_indices = list(dict.fromkeys(closest_indices))

    # 如果去重后不足目标数，从剩余点中按距离补齐
    if len(selected_internal_indices) < target_samples:
        remaining_indices = list(set(range(len(embeddings))) - set(selected_internal_indices))
        # 补齐所需数量
        shortage = target_samples - len(selected_internal_indices)
        selected_internal_indices.extend(remaining_indices[:shortage])

    # ---------------- 阶段 5: 还原到原轨迹数据 ----------------
    final_trajectories = [
        trajectories[valid_indices[internal_idx]]
        for internal_idx in selected_internal_indices
    ]

    return final_trajectories


def random_sampling(
        trajectories: List[TrajectoryRecord],
        target_samples: int,
        random_state: int = 42
) -> List[TrajectoryRecord]:

    # 1. 前置校验：如果输入为空或目标数为 0，直接返回空列表
    if not trajectories or target_samples <= 0:
        return []

    # 2. 数量校验：如果总数据量不足或刚好等于目标数，直接返回原始数据的副本
    if len(trajectories) <= target_samples:
        return trajectories

    # 3. 设定随机种子（如果需要固定结果的话）
    if random_state is not None:
        random.seed(random_state)

    # 4. 执行无放回随机采样 (random.sample)
    return random.sample(trajectories, target_samples)

class PlanGenerationPipeline:
    def __init__(
            self,
            output_dir: str,
            max_retries: int = 4,
            confidence_config: Optional[ConfidenceConfig] = None,
            calibrator: Optional[Calibrator] = None
    ):
        self.out_dir = Path(output_dir)
        self.work_dir = Path("")
        self.log_dir = self.out_dir / "logs"
        self.max_retries = max_retries
        self.max_input_size = MAX_INPUT_SIZE

        self.llm = LLMAgent(
            api_token=CHAT_TOKEN,
            model=MLOPS_MODEL,
            api_url="http://mlops.huawei.com/mlops-service/api/v2/agentService/v1/chat/completions",
            max_concurrent=MAX_LLM_CONCURRENCY
        )

        self.sampling_method = random_sampling

        # 置信度计算配置与校准器
        self.confidence_config = confidence_config or ConfidenceConfig()
        self.calibrator = calibrator or NoopCalibrator()

    def update_work_dir(self, task_cls: str, app: str):
        self.work_dir = self.out_dir / f"{task_cls}-{app}"
        self.log_dir = self.work_dir / "logs"

    # =========================
    # 工具函数 (Static / Class Methods)
    # =========================
    @staticmethod
    def load_json(file_path: str) -> Any:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            with open(file_path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f.readlines()]

    async def _file_writer_task(self, queue: asyncio.Queue):
        while True:
            item = await queue.get()
            # 接收到停止信号，退出循环
            if item is None:
                queue.task_done()
                break

            data, file_name, log_file = item
            try:
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                with open(file_name, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                log(f"生成经验保存错误: {e}", self.work_dir / log_file)
            finally:
                queue.task_done()

    @classmethod
    def dataclass_to_dict(cls, obj) -> Dict[str, Any]:
        if hasattr(obj, '__dataclass_fields__'):
            result = {}
            for field_name in obj.__dataclass_fields__:
                value = getattr(obj, field_name)
                if isinstance(value, list):
                    result[field_name] = [
                        cls.dataclass_to_dict(v) if hasattr(v, '__dataclass_fields__') else v
                        for v in value
                    ]
                elif hasattr(value, '__dataclass_fields__'):
                    result[field_name] = cls.dataclass_to_dict(value)
                else:
                    result[field_name] = value
            return result
        return obj

    @staticmethod
    def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        text = text.strip()
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
        return None

    # =========================
    # 1. 数据加载与解析 (CPU 密集型，保持同步)
    # =========================
    def parse_trajectory(self, raw_data: Dict[str, Any], trajectory_id: str) -> TrajectoryRecord:
        steps = []
        for key, value in raw_data.items():
            if not key.startswith("step"):
                continue
            try:
                step_num = int(key.replace("step", ""))
            except ValueError:
                continue

            if isinstance(value, dict):
                step = TrajectoryStep(
                    step_num=step_num,
                    manager_action=value.get("manager_action", ""),
                    thought=value.get("thought", ""),
                    action=value.get("action", ""),
                    role=value.get("role", "")
                )
                steps.append(step)

        steps.sort(key=lambda x: x.step_num)

        return TrajectoryRecord(
            trajectory_id=trajectory_id,
            task_cls=sanitize_filename(raw_data.get("category", {}).get("category")),
            app=raw_data.get("app", ""),
            instruction=raw_data.get("instruction", ""),
            intent_name=sanitize_filename(raw_data.get("intent_cluster_name", "") or raw_data.get("cluster_name", "")),
            steps=steps,
            task_status="success",
            raw_data=raw_data,
            raw_plan=raw_data.get("plan", ""),
        )

    def group_by_intent(self, trajectories: List[TrajectoryRecord]) -> List[IntentGroup]:
        intent_groups: Dict[str, List[TrajectoryRecord]] = defaultdict(list)
        for traj in trajectories:
            task = traj.task_cls if traj.task_cls else "None"
            app = traj.app if traj.app else "None"
            if traj.intent_name:
                intent_groups[
                    f"{task.replace('-', '')}-{app.replace('-', '')}-{traj.intent_name.replace('-', '')}"
                ].append(traj)

        groups = []
        for intent_name, trajs in intent_groups.items():
            intent = intent_name.split("-")[2]
            app = intent_name.split("-")[1]
            task_cls = intent_name.split("-")[0]
            groups.append(IntentGroup(task_cls=task_cls, app=app, intent_name=intent, trajectories=trajs))
        return groups

    def filter_intent_groups(
            self,
            groups: List[IntentGroup],
            min_trajectories: int = 3,
            sample_ratio: float = 0.85
    ) -> List[IntentGroup]:

        processed_groups = []

        for group in groups:
            if len(group.trajectories) <= min_trajectories:
                continue
            if group.intent_name == "未成簇意图_噪声":
                continue

            new_group = copy.deepcopy(group)

            if len(new_group.trajectories) > self.max_input_size:
                target_size = int(self.max_input_size * sample_ratio)
                new_group.trajectories = self.sampling_method(new_group.trajectories, target_size)

            processed_groups.append(new_group)

        return processed_groups

    # =========================
    # 3. 核心算法: 异步 LLM 调用
    # =========================
    async def identify_candidate_golden_path_patterns(
            self, group: IntentGroup, count_usage: bool = False
    ) -> Dict[str, Any]|Tuple[Dict[str, Any], ...]:
        task_cls = group.task_cls
        app = group.app
        intent = group.intent_name
        trajectories = group.trajectories
        token_usage = {}
        traj_idx_mapping = {f"轨迹_{idx}": traj for idx, traj in enumerate(trajectories, start=1)}
        retries = 0
        prompt = build_candidate_golden_path_prompt(intent, trajectories)

        while retries < self.max_retries:
            try:
                raw_output = await self.llm.ainvoke(prompt, count_usage=count_usage)
                if isinstance(raw_output, tuple):
                    raw_output, token_usage = raw_output
                parsed = self.extract_json_from_text(raw_output)

                results = []
                for item in parsed.get("golden_paths", []):
                    pattern = GoldenPathPattern(
                        golden_path=item.get("golden_path", []),
                        relevant_trajectories=[
                            traj_idx_mapping.get(traj_id) for traj_id in
                            item.get("relevant_redundant_trajectory_idx", [])
                            if traj_idx_mapping.get(traj_id)
                        ]
                    )
                    results.append(pattern)
                output_res = {
                    "optimal_path_type_analysis": parsed.get("optimal_path_type_analysis", ""),
                    "representative_optimal_trajectory_idx": [traj_idx_mapping.get(traj_id) for traj_id in
                                                              parsed.get("representative_optimal_trajectory_idx", [])],
                    "golden_paths": results
                }
                return output_res if not count_usage else (output_res, token_usage)

            except Exception as e:
                retries += 1
                await asyncio.sleep(5)
                log(f"      -> [{intent}] Golden Path生成错误，重试中: {e}", self.log_dir/f"{task_cls}-{app}-{intent}.log")
        return {} if not count_usage else ({}, token_usage)

    async def identify_golden_path_patterns(
            self, group: IntentGroup, candidate_golden_paths: List[List[str]], count_usage: bool = False
    ) -> Dict[str, Any] | None:
        task_cls = group.task_cls
        app = group.app
        intent = group.intent_name
        trajectories = group.trajectories
        token_usage = {}
        retries = 0
        prompt = build_golden_path_prompt(intent, trajectories, candidate_golden_paths)

        while retries < self.max_retries:
            try:
                raw_output = await self.llm.ainvoke(prompt, count_usage=count_usage)
                if isinstance(raw_output, tuple):
                    raw_output, token_usage = raw_output
                parsed = self.extract_json_from_text(raw_output)
                golden_pattern = GoldenPathPattern(
                    golden_path=parsed.get("optimal_path", []),
                    relevant_trajectories=trajectories
                )
                output_res = {
                    "golden_pattern": golden_pattern,
                    "golden_path_analysis": parsed.get("golden_path_analysis", ""),
                }
                return output_res if not count_usage else (output_res, token_usage)
            except Exception as e:
                retries += 1
                await asyncio.sleep(5)
                log(f"      -> [{intent}] Golden Path生成错误，重试中: {e}", self.log_dir/f"{task_cls}-{app}-{intent}.log")

        return None if not count_usage else (None, token_usage)

    async def _select_baseline_plan(
            self,
            group: IntentGroup,
            count_usage: bool = False
    ):
        task_cls = group.task_cls
        app = group.app
        intent = group.intent_name
        trajectories = group.trajectories
        token_usage = {}
        plans = [traj.raw_plan for traj in trajectories if traj.raw_plan]
        # 如果这个group下所有轨迹的plan都无效，则不选择baseline plan
        if not (True in [TrajectoryRoutingPipeline.plan_effectiveness_detection(plan) for plan in plans]):
            return None if not count_usage else (None, token_usage)

        retries = 0
        prompt = build_baseline_plan_selection_prompt(intent, trajectories)

        while retries < self.max_retries:
            try:
                raw_output = await self.llm.ainvoke(prompt, count_usage=count_usage)
                if isinstance(raw_output, tuple):
                    raw_output, token_usage = raw_output
                parsed = self.extract_json_from_text(raw_output)
                return parsed if not count_usage else (None, token_usage)
            except Exception as e:
                retries += 1
                await asyncio.sleep(5)
                log(f"      -> [{intent}] Baseline Plan生成错误，重试中: {e}", self.log_dir/f"{task_cls}-{app}-{intent}.log")
        return None if not count_usage else (None, token_usage)

    async def identify_redundant_patterns(
            self,
            group: IntentGroup,
            golden_path: List[str],
            redundant_trajectories: List[TrajectoryRecord],
            count_usage: bool = False
    ) -> Tuple[List[RedundantPattern], str]|Tuple[List[RedundantPattern], str, Dict[str, int]]:
        task_cls = group.task_cls
        app = group.app
        intent = group.intent_name

        token_usage = {}

        traj_idx_mapping = {f"轨迹_{idx}": traj for idx, traj in enumerate(redundant_trajectories, start=1)}
        retries = 0
        prompt = build_redundant_prompt(intent, golden_path, redundant_trajectories)

        while retries < self.max_retries:
            try:
                raw_output = await self.llm.ainvoke(prompt, count_usage=count_usage)
                if isinstance(raw_output, tuple):
                    raw_output, token_usage = raw_output
                parsed = self.extract_json_from_text(raw_output)
                patterns = []
                for traj in parsed.get("trajectory_diagnostics", []):
                    t_obj = traj_idx_mapping.get(traj.get("trajectory_idx"))
                    if not t_obj: continue
                    patterns.append(RedundantPattern(
                        trajectory_id=t_obj.trajectory_id,
                        instruction=t_obj.instruction,
                        redundancy_analysis=traj.get("redundancy_analysis", ""),
                        redundancy_types=traj.get("redundancy_types", []),
                        redundant_steps=traj.get("redundant_steps", []),
                        detailed_analysis=traj.get("detailed_analysis", ""),
                        plan_defect_analysis=traj.get("plan_defect_analysis", "")
                    ))
                summary = parsed.get("path_vulnerability_summary", "")
                return (patterns, summary) if not count_usage else (patterns, summary, token_usage)
            except Exception as e:
                retries += 1
                await asyncio.sleep(5)
                log(f"      -> [{intent}] Redundancy 分析错误，重试中: {e}", self.log_dir/f"{task_cls}-{app}-{intent}.log")
        return ([], "") if not count_usage else ([], "", token_usage)

    async def analyze_intent_group(self, group: IntentGroup, count_usage: bool = False) -> Dict[str, Any]:

        intent = group.intent_name
        trajs = group.trajectories

        token_usage = {} if not count_usage else {"total_tokens": [], "completion_tokens": []}

        step_counts = [t.step_count for t in trajs]
        avg_steps = sum(step_counts) / len(step_counts) if step_counts else 0
        min_steps = min(step_counts) if step_counts else 0
        max_steps = max(step_counts) if step_counts else 0

        # 并发获取候选 Golden Path 和 Baseline Plan
        golden_task = asyncio.create_task(self.identify_candidate_golden_path_patterns(group, count_usage))
        baseline_task = asyncio.create_task(self._select_baseline_plan(group, count_usage))

        golden_patterns, baseline_plan = await asyncio.gather(golden_task, baseline_task)
        if isinstance(golden_patterns, tuple):
            golden_patterns, golden_pattern_usage = golden_patterns
            token_usage["total_tokens"] += golden_pattern_usage.get("total_tokens", 0)
            token_usage["completion_tokens_tokens"] += golden_pattern_usage.get("completion_tokens", 0)
        if isinstance(baseline_plan, tuple):
            baseline_plan, baseline_plan_usage = baseline_plan
            token_usage["total_tokens"] += baseline_plan_usage.get("total_tokens", 0)
            token_usage["completion_tokens_tokens"] += baseline_plan_usage.get("completion_tokens", 0)

        golden_path_patterns = [pattern.golden_path for pattern in golden_patterns.get("golden_paths", [])]
        golden_path_pattern_obj = {}
        if len(golden_path_patterns) > 1:
            golden_path_pattern_obj = await self.identify_golden_path_patterns(group, golden_path_patterns, count_usage)
            if isinstance(golden_path_pattern_obj, tuple):
                golden_path_pattern_obj, golden_path_obj_usage = golden_path_pattern_obj
                token_usage["total_tokens"] += golden_path_obj_usage.get("total_tokens", 0)
                token_usage["completion_tokens_tokens"] += golden_path_obj_usage.get("completion_tokens", 0)

            golden_path_pattern = golden_path_pattern_obj.get(
                "golden_pattern",
                GoldenPathPattern(golden_path=[], relevant_trajectories=[])
            ).golden_path if golden_path_pattern_obj.get("golden_pattern") else golden_path_patterns[0]
        elif len(golden_path_patterns) == 1:
            golden_path_pattern = golden_path_patterns[0]
        else:
            golden_path_pattern = []

        redundant_results = await self.identify_redundant_patterns(group, golden_path_pattern, trajs, count_usage)
        if len(redundant_results) == 3:
            redundant_patterns, vuln_summary, redundant_token_usage = redundant_results
            token_usage["total_tokens"] += redundant_token_usage.get("total_tokens", 0)
            token_usage["completion_tokens_tokens"] += redundant_token_usage.get("completion_tokens", 0)
        elif len(redundant_results) == 2:
            redundant_patterns, vuln_summary = redundant_results
        else:
            redundant_patterns, vuln_summary = [], ""

        redundant_analysis = {
            "golden_path_pattern": golden_path_pattern,
            "baseline_plan": baseline_plan.get("optimal_intent_plan", "") if baseline_plan else "",
            "vulnerability_summary": vuln_summary,
            "redundant_patterns": [pattern.to_json() for pattern in redundant_patterns]
        }

        golden_path = {
            "golden_path": redundant_analysis["golden_path_pattern"],
            "relevant_trajectory": {
                t.trajectory_id: {
                    "action_list": t.get_actions(),
                    "raw_plan": t.get_raw_plan(),
                }
                for t in trajs
            },
        }

        formatted_redundant_patterns = [
            {
                "trajectory_id": pattern.get("trajectory_id"),
                "instruction": pattern.get("instruction"),
                "redundancy_types": pattern.get("redundancy_types"),
                "redundant_steps": pattern.get("redundant_steps"),
                "analysis": pattern.get("detailed_analysis"),
                "plan_defect_analysis": pattern.get("plan_defect_analysis")
            } for pattern in redundant_analysis["redundant_patterns"]
        ]

        reduction_potential = (avg_steps - min_steps) / avg_steps if len(
            trajs) >= 2 and 0 < min_steps < avg_steps else 0.0

        return_content = {
            "intent_name": intent,
            "total_trajectories": len(trajs),
            "step_stats": {
                "avg": round(avg_steps, 2),
                "min": min_steps,
                "max": max_steps,
                "distribution": sorted(step_counts)
            },
            "golden_path_pattern": golden_path,

            # stage 1: 获取候选最优路径的过程性输出
            "candidate_golden_paths": ["->".join(path.golden_path) for path in golden_patterns.get("golden_paths", [])],
            "candidate_golden_path_selection_reason": golden_patterns.get("optimal_path_type_analysis", ""),
            "candidate_trajectories_of_golden_path": [
                {
                    "sessionid": item.trajectory_id,
                    "instruction": item.instruction,
                    "raw_plan": item.raw_plan
                } for item in golden_patterns.get("representative_optimal_trajectory_idx", [])
            ],

            # stage 2: 选择baseline_plan的过程性输出(包含意图分析，最优plan的推理过程，选出的最优baseline)
            "baseline_plan_content": baseline_plan,

            # stage 3: 选择golden_path的过程性输出(选择golden_path的理由)
            "golden_path_selection_reason": golden_path_pattern_obj.get(
                "golden_path_analysis", ""
            ),

            "redundant_patterns": formatted_redundant_patterns,
            "step_reduction_potential": reduction_potential,
            "redundant_analysis": redundant_analysis
        }

        return return_content if not count_usage else (return_content, token_usage)

    # =========================
    # 4. Plan 生成
    # =========================
    async def generate_optimal_path_plan(
            self,
            group: IntentGroup,
            trajectories_mapping: Dict[str, TrajectoryRecord],
            golden_path: List[str], vulnerability_summary: str,
            baseline_plan: str | None = None,
            redundant_patterns: List[Dict[str, Any]] | None = None,
            count_usage: bool = False
    ) -> Optional[EfficientPlan]:

        task_cls = group.task_cls
        app = group.app
        intent = group.intent_name

        token_usage = {}

        redundant_patterns_objs = [RedundantPattern.from_json(pattern) for pattern in (redundant_patterns or [])]
        traj_redundancy_tuples = [
            (
                trajectories_mapping[rp.trajectory_id], rp.redundancy_types, rp.redundant_steps,
                rp.detailed_analysis, rp.plan_defect_analysis
            )
            for rp in redundant_patterns_objs if rp.trajectory_id in trajectories_mapping
        ]

        retries = 0
        if baseline_plan:
            log(f"      -> [{intent}] 检测到plan基线，改进中", self.log_dir/f"{task_cls}-{app}-{intent}.log")
            prompt = build_patch_plan_prompt(intent, golden_path, baseline_plan, traj_redundancy_tuples)
            mode = "patched"
        else:
            log(f"      -> [{intent}] 无历史plan基线，从零生成", self.log_dir/f"{task_cls}-{app}-{intent}.log")
            prompt = build_from_scratch_plan_prompt(intent, golden_path, traj_redundancy_tuples)
            mode = "from_scratch"

        while retries < self.max_retries:
            try:
                raw_output = await self.llm.ainvoke(prompt, count_usage=count_usage)
                if isinstance(raw_output, tuple):
                    raw_output, token_usage = raw_output
                parsed = self.extract_json_from_text(raw_output)

                phases = [
                    PlanPhase(
                        phase_idx=p.get("phase_idx", idx), phase_goal=p.get("phase_goal", ""),
                        mapped_actions=p.get("mapped_actions", []),
                        expected_visual_state=p.get("expected_visual_state", ""),
                        attention_warning=p.get("attention_warning", "None")
                    ) for idx, p in enumerate(parsed.get("execution_plan", []), start=1)
                ]

                plan = EfficientPlan(
                    intent=intent, app=app, summary=parsed.get("plan_overview", ""),
                    execution_plan=phases, generation_mode=mode, diff_review=None
                )

                if mode == "patched":
                    diff_prompt = build_plan_diff_prompt(
                        baseline_plan=baseline_plan,
                        new_plan_json_str=json.dumps(self.dataclass_to_dict(plan), ensure_ascii=False),
                        diagnostics_summary=vulnerability_summary
                    )
                    diff_res = self.extract_json_from_text(await self.llm.ainvoke(diff_prompt, count_usage=count_usage))
                    if diff_res:
                        plan.diff_review = diff_res

                # 置信度计算（信号层 + 独立评审 + 融合），失败时降级不阻断 plan 产出
                plan = await self._evaluate_plan_confidence(
                    plan=plan,
                    group=group,
                    golden_path=golden_path,
                    vulnerability_summary=vulnerability_summary,
                    has_baseline=bool(baseline_plan),
                    redundant_patterns_objs=redundant_patterns_objs,
                    plan_gen_parsed=parsed,
                    count_usage=count_usage
                )

                return plan if not count_usage else (plan, token_usage)
            except Exception as e:
                retries += 1
                await asyncio.sleep(5)
                log(f"      -> [{intent}] Plan 生成错误，重试中: {e}", self.log_dir/f"{task_cls}-{app}-{intent}.log")
        return None if not count_usage else (None, token_usage)

    async def _evaluate_plan_confidence(
            self,
            plan: EfficientPlan,
            group: IntentGroup,
            golden_path: List[str],
            vulnerability_summary: str,
            has_baseline: bool,
            redundant_patterns_objs: List[RedundantPattern],
            plan_gen_parsed: Dict[str, Any],
            count_usage: bool = False
    ) -> EfficientPlan:
        """为 plan 计算置信度：统计信号层 + 双阶段 LLM 评估 + 融合，失败时降级为统计分或 None"""
        intent = group.intent_name
        task_cls = group.task_cls
        app = group.app

        token_usage = {}

        try:
            # ---- 阶段 A: 生成时自评 ----
            llm_initial = None
            try:
                raw_initial = plan_gen_parsed.get("confidence_self_assessment")
                if raw_initial is not None:
                    llm_initial = min(1.0, max(0.0, float(raw_initial)))
            except (TypeError, ValueError):
                llm_initial = None

            # ---- 统计信号层 ----
            step_counts = [t.step_count for t in group.trajectories]
            avg_steps = sum(step_counts) / len(step_counts) if step_counts else 0
            min_steps = min(step_counts) if step_counts else 0
            reduction_potential = (
                (avg_steps - min_steps) / avg_steps
                if len(group.trajectories) >= 2 and 0 < min_steps < avg_steps else 0.0
            )
            diff_improved = None
            if plan.diff_review:
                di = plan.diff_review.get("is_improved")
                if isinstance(di, bool):
                    diff_improved = di
                elif isinstance(di, str):
                    diff_improved = di.strip().lower() in ("true", "1", "yes", "是", "pass")

            stat_score, signal_scores = compute_stat_score(
                config=self.confidence_config,
                total_trajectories=len(group.trajectories),
                step_counts=step_counts,
                reduction_potential=reduction_potential,
                redundant_patterns=redundant_patterns_objs,
                has_baseline=has_baseline,
                diff_improved=diff_improved,
                trajectories=group.trajectories,
            )

            # ---- 阶段 B: 独立评审 ----
            llm_review = None
            review_reason = ""
            critical_flaws: List[str] = []
            if self.confidence_config.lambda_ < 1.0:
                # stat_signal_summary = "\n".join(
                #     f"- {k}: {v:.3f}" for k, v in signal_scores.items()
                # )
                # redundancy_summary = "\n".join(
                #     f"//轨迹_{idx}: 冗余类型={rp.redundancy_types} | 缺陷诊断={rp.plan_defect_analysis}"
                #     for idx, rp in enumerate(redundant_patterns_objs, start=1)
                # )
                review_prompt = build_confidence_review_prompt(
                    task_cls=task_cls,
                    app=app,
                    intent=intent,
                    instructions=[traj.instruction for traj in group.trajectories],
                    plan_content="->".join(["->".join(item.mapped_actions) for item in plan.execution_plan]),
                )
                review_raw = await self.llm.ainvoke(review_prompt, count_usage=count_usage)
                if isinstance(review_raw, tuple):
                    review_raw, token_usage = review_raw
                review_parsed = self.extract_json_from_text(review_raw) if review_raw else None
                if review_parsed:
                    try:
                        rv = review_parsed.get("final_prm_confidence")
                        if rv is not None:
                            llm_review = min(1.0, max(0.0, float(rv)))
                    except (TypeError, ValueError):
                        llm_review = None
                    review_reason = review_parsed.get("overall_analysis", "")
                    flaws = [step.get("step_content") for step in review_parsed.get("step_analysis", "") if step.get("score", 1.0) < 1.0]
                    if isinstance(flaws, list):
                        critical_flaws = [str(f) for f in flaws]

            # ---- 融合 ----
            final_confidence, breakdown = fuse_confidence(
                self.confidence_config, stat_score, llm_initial, llm_review
            )
            breakdown["signal_scores"] = {k: round(v, 4) for k, v in signal_scores.items()}
            if critical_flaws:
                breakdown["critical_flaws"] = critical_flaws

            final_confidence = self.calibrator.transform(final_confidence)

            plan.confidence = round(float(final_confidence), 4)
            plan.confidence_reason = review_reason
            plan.confidence_breakdown = breakdown
            plan.filtered = plan.confidence < self.confidence_config.threshold

            log(
                f"      -> [{intent}] 置信度: {plan.confidence} "
                f"(stat={stat_score:.3f}, initial={llm_initial}, review={llm_review}, "
                f"filtered={plan.filtered})",
                self.log_dir / f"{task_cls}-{app}-{intent}.log"
            )
        except Exception as e:
            log(
                f"      -> [{intent}] 置信度计算异常，降级为默认值: {e}",
                self.log_dir / f"{task_cls}-{app}-{intent}.log"
            )
        return plan

    async def generate_plans(
            self,
            group: IntentGroup,
            analysis_result: Dict[str, Any],
            count_usage: bool = False,
    ) -> List[EfficientPlan] | None:
        trajectories_mapping = {t.trajectory_id: t for t in group.trajectories}
        # app_name = group.trajectories[0].app if group.trajectories else "Unknown"
        redundant_analysis = analysis_result.get("redundant_analysis", {})

        return await self.generate_optimal_path_plan(
            group=group,
            trajectories_mapping=trajectories_mapping,
            vulnerability_summary=redundant_analysis.get("vulnerability_summary", ""),
            golden_path=redundant_analysis.get("golden_path_pattern", []),
            baseline_plan=redundant_analysis.get("baseline_plan", ""),
            redundant_patterns=redundant_analysis.get("redundant_patterns", []),
            count_usage=count_usage
        )

    # =========================
    # 单组处理逻辑 (并发调用)
    # =========================
    async def generate_plan_by_intent(
            self,
            group: IntentGroup,
            log_file: str,
            queue: asyncio.Queue,
            count_usage: bool = False,
    ) -> Tuple[Dict[str, Any], str] | Exception:
        """处理单一意图组的完整流程，返回结果和生成的文件名供索引记录"""
        task_cls = group.task_cls
        app = group.app
        intent = group.intent_name
        token_usage_record = []
        group_info = f"{group.task_cls}-{group.app}-{group.intent_name}"
        log(f"\n[启动并发任务] 处理意图: {group.intent_name}", self.log_dir/f"{task_cls}-{app}-{intent}.log", append=False)

        group_result = {
            "task_cls": group.task_cls, "app": group.app, "intent_name": group.intent_name,
            "instructions": list(set(t.instruction for t in group.trajectories)),
            "total_trajectories": len(group.trajectories),
            "timestamp": datetime.now().isoformat()
        }

        # 3.1 分析簇并生成诊断
        try:
            analysis = await self.analyze_intent_group(group, count_usage)
            if isinstance(analysis, tuple):
                analysis, token_usage_intent_analysis = analysis
                token_usage_intent_analysis.update({"count_part": "intent group analysis"})
                token_usage_record.append(token_usage_intent_analysis)
        except Exception as e:
            log(f"意图组处理失败 (intent Group 分析) | 意图: {group_info} | 错误详情: {e}",
                self.log_dir / f"{task_cls}-{app}-{group.intent_name}.log")
            log(f"意图组处理失败 (intent Group 分析) | 意图: {group_info} | 错误详情: {e}", self.work_dir / log_file)
            return e

        # 输出步骤统计量
        group_result["step_statistics"] = analysis.get("step_stats", {})

        # 输出候选的最优路径 pattern
        group_result["golden_path_pattern"] = analysis.get("golden_path_pattern", [])
        # 从候选中选择这一条 golden path 的 reason
        group_result["golden_path_selection_reason"] = analysis.get("golden_path_selection_reason", {})

        # 候选的golden_paths以及选择这些候选路径的reason
        group_result["candidate_golden_paths"] = analysis.get("candidate_golden_paths", [])
        group_result["candidate_golden_path_selection_reason"] = analysis.get("candidate_golden_path_selection_reason", "")
        # 候选的golden_path对应的轨迹
        group_result["candidate_trajectories_of_golden_path"] = analysis.get("candidate_trajectories_of_golden_path", [])
        # 选择baseline plan的具体信息
        group_result["baseline_plan_content"] = analysis.get("baseline_plan_content", {})

        group_result["redundant_patterns"] = analysis.get("redundant_analysis", {}).get("redundant_patterns", [])
        group_result["clusters_analysis"] = {
            "baseline_plan": analysis.get("redundant_analysis", {}).get("baseline_plan"),
            "vulnerability_summary": analysis.get("redundant_analysis", {}).get("vulnerability_summary")
        }

        # 整理低效路径和动作列表...
        full_trajectory_actions = {}
        inefficient_paths_detail = []
        avg_steps = group_result["step_statistics"].get("avg", 0)

        for t in group.trajectories:
            full_trajectory_actions[t.trajectory_id] = {
                "step_count": t.step_count, "actions": t.get_actions(), "manager_actions": t.get_manager_actions()
            }
            if t.step_count > avg_steps * 1.3:
                inefficient_paths_detail.append({
                    "trajectory_id": t.trajectory_id, "raw_plan": t.get_raw_plan(), "step_count": t.step_count,
                    "avg_steps": avg_steps, "extra_steps": t.step_count - avg_steps, "actions": t.get_actions()
                })

        group_result["full_trajectory_actions"] = full_trajectory_actions
        group_result["inefficient_paths_detail"] = inefficient_paths_detail

        # 3.2 生成 Plan
        try:
            plan = await self.generate_plans(group, analysis, count_usage)
            if isinstance(plan, tuple):
                plan, token_usage_plan_gen = plan
                token_usage_plan_gen.update({"count_part": "plan generation"})
                token_usage_record.append(token_usage_plan_gen)
        except Exception as e:
            log(f"意图组处理失败 (Plan 生成) | 意图: {group_info} | 错误详情: {e}",
                self.log_dir / f"{task_cls}-{app}-{group.intent_name}.log")
            log(f"意图组处理失败 (Plan 生成) | 意图: {group_info} | 错误详情: {e}", self.work_dir / log_file)
            return e
        group_result["generated_plans"] = self.dataclass_to_dict(plan)

        # 落地文件
        safe_filename = sanitize_filename(f"{group.task_cls}-{group.app}-{group.intent_name}")
        intent_filename = f"{safe_filename}.json"
        intent_file = os.path.join(self.work_dir / "intents", intent_filename)

        # 使用多线程写入避免阻塞事件循环
        await queue.put((group_result, intent_file, log_file))

        log(f"[任务完成] 保存结果到: {intent_filename}", self.log_dir/f"{task_cls}-{app}-{intent}.log")

        log(f"意图组处理成功 | 意图: {group_info}", self.work_dir / log_file)

        return (group_result, intent_filename) if not count_usage else (group_result, intent_file, token_usage_record)

    # =========================
    # 5. 主Pipeline
    # =========================
    async def run_pipeline(
            self,
            input_data: List[Dict[str, Any]],
            task_cls: str,
            app: str,
            target_intents: Optional[List[str]] = None,
            count_usage: bool = False,
    ):

        for data in input_data:
            data.pop("intent_embedding", None)

        pipeline_start_time = datetime.now()
        intents_dir = self.work_dir / "intents"

        # Step 1: 解析轨迹
        trajectories = [
            self.parse_trajectory(
                item,
                f"traj_{item.get('sessionid', idx)}"
            ) for idx, item in enumerate(input_data) if all(
                [
                    item.get("category", {}).get('category') == task_cls,
                    item.get("app") == app,
                    item.get("intent_cluster_name") in target_intents if target_intents else True
                ]
            )
        ]

        # Step 2: 分组与初筛
        intent_groups = self.group_by_intent(trajectories)
        intent_groups = self.filter_intent_groups(intent_groups)

        log_file = f"{task_cls}-{app}-total.log"

        if target_intents:
            log_file = f"{task_cls}-{app}-partial.log"

            # 预处理 target_intents 以匹配格式
            processed_targets = set(intent.replace('-', '') for intent in target_intents)
            found_intents = set(g.intent_name for g in intent_groups)
            missed_intents = processed_targets - found_intents

            if missed_intents:
                log(f"[{task_cls}-{app}] 存在未命中的目标意图: {missed_intents}", self.work_dir / log_file)

            # 过滤 intent_groups
            intent_groups = [
                g for g in intent_groups if g.intent_name in processed_targets
            ]

        if not intent_groups:
            # log(f"检测到不存在意图簇：{task_cls}-{app}", self.work_dir / log_file)
            return []

        os.makedirs(intents_dir, exist_ok=True)

        # =========================
        # 检测本地已生成意图并跳过
        # =========================
        to_process_groups: List[IntentGroup] = []
        all_intent_results: List[Dict[str, Any]] = []
        successful_filenames: List[str] = []

        for group in intent_groups:
            safe_filename = sanitize_filename(f"{group.task_cls}-{group.app}-{group.intent_name}")
            intent_filename = f"{safe_filename}.json"
            target_path = intents_dir / intent_filename

            # 1. 检查文件是否存在且非空
            if target_path.exists() and target_path.stat().st_size > 0:
                try:
                    # 尝试读取，确认不是半写入或损坏的非法 json
                    cached_data = self.load_json(str(target_path))
                    all_intent_results.append(cached_data)
                    successful_filenames.append(intent_filename)
                    log(f"[跳过生成] 检测到本地已存在有效意图结果: {intent_filename}", self.work_dir / log_file)
                    continue
                except Exception as e:
                    log(f"[检测异常] 读取本地已存在文件失败，重新生成: {intent_filename}, 错误: {e}", self.work_dir / log_file)

            to_process_groups.append(group)

        log(f"意图统计: 总计 {len(intent_groups)} 个 | 本地复用 {len(successful_filenames)} 个 | 待生成 {len(to_process_groups)} 个", self.work_dir / log_file)

        try:
            write_queue = asyncio.Queue()
            writer_task = asyncio.create_task(self._file_writer_task(write_queue))
            # 若全部已经生成，直接跳到写索引和返回阶段
            if to_process_groups:

                log(f"\n开始并发处理剩余 {len(to_process_groups)} 个意图组...", self.work_dir / log_file)
                tasks = [
                    self.generate_plan_by_intent(
                        group,
                        log_file,
                        count_usage=count_usage,
                        queue=write_queue,
                    ) for group in to_process_groups
                ]

                results = await tqdm_asyncio.gather(
                    *tasks,
                    desc="Processing intent groups"
                )

                for res in results:
                    if not isinstance(res, Exception):
                        if len(res) == 2:
                            group_result, filename = res
                        elif len(res) == 3:
                            group_result, filename, token_usage_record = res
                        else:
                            group_result, filename = [], ""
                        all_intent_results.append(group_result)
                        successful_filenames.append(filename)


            # 统一更新索引（合并历史与新生成的意图索引）
            if successful_filenames:
                index_file = intents_dir / "intents_index.json"
                index_data = {"intents": []}
                if os.path.exists(index_file):
                    try:
                        with open(index_file, "r", encoding="utf-8") as f:
                            index_data = json.load(f)
                    except Exception:
                        pass

                existing_intents = set(index_data.get("intents", []))
                existing_intents.update(successful_filenames)
                index_data["intents"] = list(existing_intents)

                with open(index_file, "w", encoding="utf-8") as f:
                    json.dump(index_data, f, ensure_ascii=False, indent=4)

            pipeline_duration = (datetime.now() - pipeline_start_time).total_seconds()
            log("\n" + "=" * 80, self.work_dir / log_file)
            log(f"Pipeline执行完成 | 耗时: {pipeline_duration:.1f}秒", self.work_dir / log_file)
            log("=" * 80, self.work_dir / log_file)

            await write_queue.put(None)

            await writer_task
        finally:
            if self.llm is not None:
                await self.llm.close()
                self.llm = None


    def run(self, data_file: str, task_cls: str, app_name: str,
                           intent_name: Optional[List[str]] = None):
        data = self.load_json(data_file)
        filtered_data = [d for d in data if d.get("cluster_effectiveness", {}).get("is_match", True)]
        self.update_work_dir(task_cls, app_name)
        if not isinstance(data, list):
            raise ValueError(f"输入数据应该是list格式，实际: {type(data)}")
        asyncio.run(
            self.run_pipeline(
                input_data=filtered_data,
                task_cls=task_cls,
                app=app_name,
                target_intents=intent_name
            )
        )

    async def llm_gen_instructions(
            self,
            app: str,
            intent: str,
            instructions: List[str],
            generated_num: int,
            count_usage: bool = False
    ) -> List[str]:
        '''
        根据 Intent 和 Instructions 清洗现有的重复指令，并生成更多样性的新指令。

        Args:
            intent: 具体的意图描述
            instructions: 现有的 instruction 列表（可能包含高度相似的冗余数据）
            generated_num: 期望最终达到的 instruction 总数

        Returns:
            List[str]: 经过清洗并补充完整后的最终 instruction 列表（总数为 generated_num）
        '''

        # 将现有的 instructions 格式化为字符串
        existing_inst_str = "\n".join([f"- {inst}" for inst in instructions])

        # 设计 Prompt：要求 LLM 同时处理“去重”和“补充生成”
        prompt = f"""
    你是一个专业的指令生成与优化助手。请根据下面提供的具体意图（Intent）和现有的指令示例，清理高度相似的指令，并生成具有更高多样性的新指令。

    【app】
    {app}

    【意图描述】
    {intent}

    【现有指令示例】（可能包含高度相似的重复项）
    {existing_inst_str}

    【任务要求】
    1. 去重与筛选：审查【现有指令示例】，剔除其中表达高度相似或重复的指令，仅保留具备多样性的优质指令。
    2. 补充生成：在保留的指令基础上，补充生成全新的指令，使得最终输出的指令总数严格等于 {generated_num} 个。所有指令必须符合上述【意图描述】以及【app】场景。
    3. 全局多样性：所有指令（包含保留的旧指令和新生成的指令）在表达方式、句式结构、语气或具体场景上都应保持多样性，绝对避免相互之间高度重复。
    4. 输出格式：请直接输出最终组合好的 {generated_num} 个指令，每个指令占一行。
    5. 严格限制：不要输出任何开场白、解释性文字、序号（如"1."、"2."）或项目符号（如"-"、"*"），纯文本输出即可。
    """

        # 调用 LLM API (此处假设 call_llm 是已定义的方法)
        response = await self.llm.ainvoke(prompt, count_usage=count_usage)

        if not response:
            print("[错误] LLM 未返回任何结果。")
            # 如果调用失败，采取降级策略：至少去重现有的并返回
            return list(set(instructions))[:generated_num]

        # 解析和清洗 LLM 返回的结果
        final_instructions = []

        # 按行分割
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 使用正则表达式清洗可能存在的序号或项目符号 (例如 "1. ", "- ", "* ", "1、")
            cleaned_line = re.sub(r'^(\d+[\.\)、]\s*|[-*]\s*)', '', line).strip()

            # 简单代码层面的二次去重过滤，防止大模型抽风输出一模一样的行
            if cleaned_line and cleaned_line not in final_instructions:
                final_instructions.append(cleaned_line)

        # 截取所需数量（以防 LLM 生成了过多或过少的结果），如果不够则原样返回已解析的
        return final_instructions[:generated_num]

    async def _output_generated_plan_async(self, count_usage: bool = False):
        """真正的异步执行逻辑，全程共享同一个事件循环和 TCP 连接池"""
        # 注意：如果在你的环境中初始化 LLMAgent 是懒加载的，这里不需要重新实例化。
        # 如果你之前没有在 LLMAgent 类中实现自适应循环，可以在这里加一句确保初始化：

        from datetime import date

        today = date.today()
        date = today.strftime("%Y-%m-%d")

        if self.llm is None:
            self.llm = LLMAgent(
                api_token=CHAT_TOKEN,
                model=MLOPS_MODEL,
                api_url="http://mlops.huawei.com/mlops-service/api/v2/agentService/v1/chat/completions",
                max_concurrent=MAX_LLM_CONCURRENCY
            )

        scenario_list = [entry.name for entry in os.scandir(self.out_dir) if entry.is_dir()]
        output_plans = []

        try:
            for scenario in scenario_list:
                path = self.out_dir / scenario / "intents" / "intents_index.json"
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        intents = json.load(f)
                else:
                    continue

                plans = intents['intents']
                for p in tqdm(plans, desc="⚙️ 正在生成指令及转换格式", unit="plan"):
                    file_path = self.out_dir / scenario / "intents" / f"{p}"

                    if not os.path.exists(file_path):
                        tqdm.write(f"⚠️ 文件未找到，跳过: {file_path}")
                        continue

                    with open(file_path, 'r', encoding='utf-8') as f:
                        p_json = json.load(f)

                    original_instructions = p_json.get('instructions', [])

                    # 【关键修改 1】使用 await 替代 asyncio.run()
                    try:
                        if not p_json:
                            continue
                        if not p_json.get("generated_plans", {}).get("app"):
                            continue
                        if not p_json.get("generated_plans", {}).get("intent"):
                            continue
                        new_instructions = await self.llm_gen_instructions(
                            app=p_json.get("generated_plans", {}).get("app"),
                            intent=p_json.get("generated_plans", {}).get("intent"),
                            instructions=original_instructions,
                            generated_num=7,
                            count_usage=count_usage
                        )
                    except Exception:
                        continue

                    # 组合所有指令 (原有的 + 新生成的)
                    final_instructions = original_instructions + new_instructions

                    out = {
                        "_id": "",
                        "date": date.replace("-", ""),
                        "task": p_json['generated_plans']['intent'],
                        "description": p_json['generated_plans']['summary'],
                        "app_name": [p_json['generated_plans']['app']],
                        "bundle_name": [],
                        "instructions": final_instructions,
                        "difficulty": "",
                        "content": "->".join(
                            ["->".join(item['mapped_actions']) for item in
                             p_json['generated_plans']['execution_plan']]),
                        "task_id": "",
                        "confidence": p_json["generated_plans"]["confidence"],
                        "confidence_details": p_json["generated_plans"]["confidence_breakdown"],
                        "confidence_reason": p_json["generated_plans"]["confidence_reason"],
                    }
                    output_plans.append(out)

            # 写入结果文件
            with open(self.out_dir / f"generated_plans_{date}.json", 'w', encoding='utf-8') as f:
                for plan in output_plans:
                    f.write(json.dumps(plan, ensure_ascii=False) + '\n')

        finally:
            # 【关键修改 2】在所有任务跑完后，统一且安全地关闭 aiohttp 连接池
            if self.llm is not None:
                await self.llm.close()
                self.llm = None

    def output_generated_plan(self, count_usage: bool = False):
        """同步入口：对外暴露的接口不变，内部启动一次全局事件循环"""
        asyncio.run(self._output_generated_plan_async(count_usage))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Success Experience Generation Pipeline (Async)")
    parser.add_argument("--input_file", type=str, required=True, help="输入数据文件（JSON格式）")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--task_cls", type=str, required=True, help="任务类别")
    parser.add_argument("--app_name", type=str, required=True, help="应用名称")
    parser.add_argument("--intent", type=str, help="指定意图名称（可选）")
    args = parser.parse_args()

    pipeline = PlanGenerationPipeline(output_dir=args.output_dir, max_retries=4)

    # 启动异步事件循环
    result = pipeline.run(
        data_file=args.input_file,
        task_cls=args.task_cls,
        app_name=args.app_name,
        intent_name=args.intent
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))