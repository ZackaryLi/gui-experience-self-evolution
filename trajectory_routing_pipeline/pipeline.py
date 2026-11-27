import json
import re
import os
import uuid
import typing as t
import asyncio
import datetime
import concurrent.futures
from collections import defaultdict
from pathlib import Path
import hashlib
import time

import numpy as np
import openai

from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from openai import AsyncOpenAI

from .plan_recall import query_plan
from .llm import extract_json, LLMAgent
from .instruction_pattern_recognition import (
    build_intent_summary_prompt,
    build_batch_scenario_classify_prompt, build_batch_intent_recognition_prompt
)
from .verdict import build_trajectory_text, build_batch_verdict_prompt, resolve_task_completed

LOG_FILE = "run.log"
MAX_LLM_CONCURRENCY = 6
MLOPS_MODEL = "qwen38-27b-vl"
MAX_EMBEDDING_CONCURRENCY = 10

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
    EMBEDDING_TOKEN = config["embedding_api_key"]
    CHAT_TOKEN = config["chat_api_key"]

def get_normalized_key(app: str, instruction: str) -> str:
    """
    对 App 和 Instruction 进行严格的清洗与归一化，生成唯一 Key。
    过滤标点符号、全半角差异及英文大小写，避免相同语义的指令被拆分。
    """
    clean_app = str(app).strip().lower() if app else "unknown_app"

    if not instruction:
        clean_instr = ""
    else:
        clean_instr = instruction.strip()
        # 全角转半角
        clean_instr = ''.join([chr(ord(c) - 65248) if 65281 <= ord(c) <= 65374 else c for c in clean_instr])
        # 转小写
        clean_instr = clean_instr.lower()
        # 移除所有标点符号、特殊字符，只保留中文、英文、数字
        clean_instr = re.sub(r'[^\w\u4e00-\u9fa5]', '', clean_instr)

    raw_key = f"{clean_app}#{clean_instr}"
    return hashlib.md5(raw_key.encode('utf-8')).hexdigest()

def _process_cluster_worker_backup(local_X):
    """
    分离出的 CPU 密集型任务：聚类与降维
    """
    clustering = DBSCAN(eps=0.08, min_samples=2, metric='cosine')
    labels = clustering.fit_predict(local_X)
    unique_labels = set(labels)

    # 噪声回捞逻辑
    if -1 in unique_labels and len(unique_labels) > 1:
        valid_c_ids = [c for c in unique_labels if c != -1]
        centroids = np.array([np.mean(local_X[labels == c_id], axis=0) for c_id in valid_c_ids])

        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids = np.divide(centroids, norms, out=np.zeros_like(centroids), where=norms != 0)

        noise_indices = np.where(labels == -1)[0]
        noise_vectors = local_X[noise_indices]

        n_norms = np.linalg.norm(noise_vectors, axis=1, keepdims=True)
        noise_vectors = np.divide(noise_vectors, n_norms, out=np.zeros_like(noise_vectors), where=n_norms != 0)

        sim_matrix = np.dot(noise_vectors, centroids.T)

        if sim_matrix.size > 0:
            max_sim_indices = np.argmax(sim_matrix, axis=1)
            max_sim_values = np.max(sim_matrix, axis=1)

            for i, n_idx in enumerate(noise_indices):
                if max_sim_values[i] >= 0.80:
                    labels[n_idx] = valid_c_ids[max_sim_indices[i]]

    # 降维并加入规模保护机制
    n_samples = len(local_X)
    if n_samples >= 4:
        try:
            # 【修复关键】将 TSNE 的退化阈值从 5000 降到 2000
            # 当单簇数据量超过 2000 时，强行使用 PCA，防止单一任务耗尽内存
            if n_samples > 2000:
                coords2d = PCA(n_components=2).fit_transform(local_X)
            else:
                reducer = TSNE(n_components=2, perplexity=min(30, n_samples - 1), random_state=42)
                coords2d = reducer.fit_transform(local_X)
        except Exception:
            coords2d = PCA(n_components=2).fit_transform(local_X)
    else:
        coords2d = PCA(n_components=2).fit_transform(local_X)

    return labels, coords2d


def _process_cluster_worker(local_X: np.ndarray, need_coords: bool = False):
    """
    重构后的 CPU 密集型任务：向量去重保护 + 层次聚类 + 按需安全降维
    彻底解决全等向量导致的 0xC00000FD 线程栈溢出崩溃与 PCA/TSNE 除零警告
    """
    n_samples = len(local_X)

    # 1. 极小样本保护（消除 n_samples < 2 时 PCA 报错）
    if n_samples == 0:
        return np.array([], dtype=int), np.empty((0, 2), dtype=np.float32)
    if n_samples == 1:
        return np.zeros(1, dtype=int), np.zeros((1, 2), dtype=np.float32)

    # 2. 样本级 L2 归一化（确保 cosine 距离数值绝对稳定）
    norms = np.linalg.norm(local_X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed_X = local_X / norms

    # 3. 【核心防崩机制】对完全相同的重复向量先折叠去重，再聚类
    # 避免大量距离为 0 的全等向量导致层次聚类递归树退化引发 C++ 栈溢出 (0xC00000FD)
    rounded_X = np.round(normed_X, decimals=5)
    _, unique_indices, inverse_indices = np.unique(
        rounded_X, axis=0, return_index=True, return_inverse=True
    )
    unique_X = normed_X[unique_indices]
    n_unique = len(unique_X)

    if n_unique == 1:
        # 所有样本的向量完全一致，直接全部归为第 0 簇
        labels = np.zeros(n_samples, dtype=int)
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='cosine',
            linkage='complete',
            distance_threshold=0.22
        )
        unique_labels = clustering.fit_predict(unique_X)
        # 将唯一向量的聚类标签映射回全量样本
        labels = unique_labels[inverse_indices]

    # 4. 按需降维逻辑（全局缓存聚类不需要坐标时直接跳过，提速 5~10 倍并规避警告）
    if not need_coords:
        return labels, None

    # 如果需要可视化坐标，先检查方差是否退化（全等向量或极小方差直接返回零坐标，避免 divide by zero）
    if n_unique < 2 or np.var(normed_X) < 1e-8:
        coords2d = np.zeros((n_samples, 2), dtype=np.float32)
    elif n_samples > 2000 or n_samples < 4 or n_unique < 4:
        try:
            coords2d = PCA(n_components=2).fit_transform(normed_X)
            coords2d = np.nan_to_num(coords2d, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            coords2d = np.zeros((n_samples, 2), dtype=np.float32)
    else:
        perplexity = min(30, n_unique - 1)
        try:
            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                metric='cosine',
                init='random',  # 避免默认 init='pca' 在低秩矩阵上触发除零警告
                random_state=42
            )
            coords2d = reducer.fit_transform(normed_X)
            coords2d = np.nan_to_num(coords2d, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            coords2d = np.zeros((n_samples, 2), dtype=np.float32)

    return labels, coords2d


class TrajectoryRoutingPipeline:
    def __init__(
            self,
            save_path,
            log_file=LOG_FILE,
            max_embedding_concurrency=MAX_EMBEDDING_CONCURRENCY,
            count_usage: bool = False,
    ):
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.log_file = self.save_path / log_file

        self.min_validate_step = 5

        self.success_trajectories_path = self.save_path / "success_trajectories.jsonl"
        self.failure_trajectories_path = self.save_path / "failure_trajectories.jsonl"
        self.ineffective_plan_recall_trajectories_path = self.save_path / "ineffective_plan_recall_trajectories.jsonl"
        self.rejection_trajectories_path = self.save_path / "rejected_trajectories.jsonl"
        self.parse_failed_trajectories_path = self.save_path / "parse_failed_trajectories.jsonl"

        self.routed_trajectories_path = self.save_path / "routed_trajectories.jsonl"
        self.visualization_cache_path = self.save_path / "clustering_visualization.json"

        self.filtered_routed_trajectories_path = self.save_path / "filtered_routed_trajectories.jsonl"

        # 新增路径
        self.global_instruction_cache_path = self.save_path / "global_instruction_cache.jsonl"
        self.global_intent_cache_path = self.save_path / "global_intent_cache.jsonl"

        # 意图级再分流产物（基于 filtered_routed_trajectories.jsonl，按意图聚合判定 IND/OOD）
        self.intent_ind_trajectories_path = self.save_path / "intent_ind_trajectories.jsonl"
        self.intent_ood_trajectories_path = self.save_path / "intent_ood_trajectories.jsonl"

        self.embedding_client = AsyncOpenAI(
            api_key=EMBEDDING_TOKEN,
            base_url="http://mlops.huawei.com/mlops-service/api/v2/agentService/v1"
        )
        self.llm = LLMAgent(
            api_token=CHAT_TOKEN,
            model=MLOPS_MODEL,
            api_url="http://mlops.huawei.com/mlops-service/api/v2/agentService/v1/chat/completions",
            max_concurrent=MAX_LLM_CONCURRENCY
        )
        self.embedding_sem = asyncio.Semaphore(max_embedding_concurrency)

        self.count_usage = count_usage
        self.perf_stats = {
            # 1. Token 消耗统计
            "token_usage": {
                "stage1_classify_and_intent": {"total_tokens": 0, "completion_tokens": 0, "prompt_tokens": 0},
                "stage1_cluster_naming": {"total_tokens": 0, "completion_tokens": 0, "prompt_tokens": 0},
                "stage2_verdict": {"total_tokens": 0, "completion_tokens": 0, "prompt_tokens": 0},
                "all_stages_total": {"total_tokens": 0, "completion_tokens": 0, "prompt_tokens": 0},
            },
            # 2. 各阶段实际墙钟耗时 (Wall-Clock Seconds)
            "stage_wall_time_sec": {
                "stage1_1_load_and_scan_cache": 0.0,  # 扫描文件与加载历史缓存
                "stage1_1_llm_and_embed_pipeline": 0.0,  # 增量指令 LLM提取 + Embedding + 落盘总耗时
                "stage1_2_local_clustering_cpu": 0.0,  # 局部层次聚类 (线程池 CPU)
                "stage1_2_cluster_llm_naming": 0.0,  # 聚类簇 LLM 命名总结
                "stage1_2_synonym_cluster_merge": 0.0,  # 同义簇 Embedding 合并
                "stage1_2_total_clustering": 0.0,  # Stage 1.2 聚类全流程总耗时
                "stage2_0_preprocess_and_normalize": 0.0,  # 轨迹标准化、拒答过滤与挂载聚类标签
                "stage2_1_plan_recall": 0.0,  # Plan Recall 经验召回判定
                "stage2_2_trajectory_verdict": 0.0,  # 轨迹 Verdict 批量判定与落盘
                "stage2_total": 0.0,  # Stage 2 全流程总耗时
                "stage3_post_routing_and_split": 0.0,  # 收尾合并与 IND/OOD 意图级再分流
                "pipeline_total_wall_time": 0.0,  # 整个 Pipeline 端到端总耗时
            },
            # 3. 并发内部子操作累计耗时 (Cumulative Coroutine Seconds，用于分析并发瓶颈)
            "cumulative_subtask_time_sec": {
                "stage1_llm_classify_and_intent_cum": 0.0,  # 所有 Batch 花在分类+意图 LLM 上的累计时间
                "stage1_intent_embedding_cum": 0.0,  # 所有 Batch 花在 Embedding 上的累计时间
                "stage2_verdict_llm_cum": 0.0,  # 所有 Batch 花在 Verdict LLM 上的累计时间
            }
        }

    def _add_usage(self, stage_key: str, usage_dict: dict):
        """安全累加单个批次的 token 消耗（自动推导 prompt_tokens）"""
        if not self.count_usage or not usage_dict or not isinstance(usage_dict, dict):
            return
        t_tok = usage_dict.get("total_tokens") or 0
        c_tok = usage_dict.get("completion_tokens") or 0
        p_tok = max(0, t_tok - c_tok)  # 原版 llm.py 未返回 prompt_tokens，直接由差值精准算出

        target = self.perf_stats["token_usage"][stage_key]
        total = self.perf_stats["token_usage"]["all_stages_total"]

        target["total_tokens"] += t_tok
        target["completion_tokens"] += c_tok
        target["prompt_tokens"] += p_tok

        total["total_tokens"] += t_tok
        total["completion_tokens"] += c_tok
        total["prompt_tokens"] += p_tok

    def _set_wall_time(self, key: str, seconds: float):
        """记录某个阶段的墙钟耗时"""
        self.perf_stats["stage_wall_time_sec"][key] = round(seconds, 3)

    def _flush_perf_report(self):
        """实时将当前最新的性能与 Token 统计刷入磁盘文件"""
        if not self.count_usage:
            return
        try:
            report_file = self.save_path / "perf_and_token_report.json"
            report_file.parent.mkdir(parents=True, exist_ok=True)
            report_str = json.dumps(self.perf_stats, ensure_ascii=False, indent=2)
            with open(report_file, "w", encoding="utf-8") as f_rep:
                f_rep.write(report_str)
        except Exception as e:
            self.log(f"[实时写入性能报告失败] {e}")

    def _add_cum_time(self, key: str, seconds: float):
        """累加并发子任务的耗时"""
        self.perf_stats["cumulative_subtask_time_sec"][key] = round(
            self.perf_stats["cumulative_subtask_time_sec"][key] + seconds, 3
        )


    def log(self, msg: str):
        if self.log_file:
            try:
                # 获取当前时间，格式化为: YYYY-MM-DD HH:MM:SS
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # 组合成带有时间戳的日志格式
                log_msg = f"[{current_time}] {msg}\n"
                print(log_msg)

                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(log_msg)
            except Exception as e:
                print(f"[日志写入失败] {e}")

    @staticmethod
    def load_json_data(file_path: str):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return [json.loads(line) for line in f.readlines()]
            except Exception:
                return []

    @staticmethod
    def calc_valid_step(trajectory: t.Dict[str, t.Any]) -> int:
        """
        计算轨迹的有效执行步数。
        有效步数定义：manager_action为空，且action不为空，不包含do_nothing()，
        不以'等待'或'重新'开头。
        并且过滤连续的重复动作（字符串完全相同，或提取的 click 动作完全相同）。
        """
        valid_step_count = 0
        last_valid_act = None  # 用于记录上一次的有效动作，过滤连续重复操作

        # 预编译正则，用于提取类似 click([449, 203]) 的指令
        # 匹配 "click([" 开头，"])" 结尾的最短内容
        click_pattern = re.compile(r'click\(\[.*?\]\)')

        # 1. 筛选出所有以 "step" 开头的键 (例如 "step1", "step2")
        step_keys = [k for k in trajectory.keys() if isinstance(k, str) and k.startswith("step")]

        # 2. 按照步骤编号进行排序，确保按照先后顺序遍历
        try:
            step_keys.sort(key=lambda x: int(x.replace("step", "")))
        except ValueError:
            pass

            # 3. 按顺序遍历步骤，统计有效步数
        for key in step_keys:
            step_data = trajectory.get(key)
            if not isinstance(step_data, dict):
                continue

            manager_act = step_data.get("manager_action", "")
            act = str(step_data.get("action", "")).strip()

            # 判断条件：
            # 1. manager_action 为空
            # 2. action 不为空
            # 3. action 不包含 do_nothing()
            # 4. action 不以 "等待" 开头
            # 5. action 不以 "重新" 开头
            if (not manager_act and
                    act and
                    "do_nothing()" not in act and
                    not act.startswith("等待") and
                    not act.startswith("重新")):

                is_duplicate = False

                # 6. 检测连续两次相同的动作
                if act == last_valid_act:
                    # 规则 A: 字符串完全相同
                    is_duplicate = True
                elif last_valid_act is not None:
                    # 规则 B: 字符串不同，但包含的 click 动作坐标相同
                    curr_click_match = click_pattern.search(act)
                    last_click_match = click_pattern.search(last_valid_act)

                    if curr_click_match and last_click_match:
                        # 如果两者都包含 click，且内容一致，也视作重复
                        if curr_click_match.group() == last_click_match.group():
                            is_duplicate = True

                # 如果不是重复的动作，才计入有效步数
                if not is_duplicate:
                    valid_step_count += 1
                    last_valid_act = act  # 计入有效步数后，更新 last_valid_act

        return valid_step_count

    def trajectory_normalize(self, raw_traj: t.Union[str, t.Dict[str, t.Any]]):
        """
        修正了原版的 normalize，确保返回标准化的字典结构，
        并增加提取有效执行步数。
        有效步数定义：manager_action为空，且action不为空，不包含do_nothing()，且不以'等待'开头。
        """
        session_id = raw_traj.get("sessionid", "") if isinstance(raw_traj, dict) else ""
        if isinstance(raw_traj, str):
            try:
                trajectory = json.loads(raw_traj)
                session_id = trajectory.get("sessionid", "")
            except json.decoder.JSONDecodeError:
                self.log(f"Traj:{session_id} 标准化失败\n Normalization：json提取失败")
                return raw_traj, False
        elif isinstance(raw_traj, dict):
            trajectory = raw_traj
        else:
            self.log(f"Traj:{session_id} 标准化失败\n Normalization：Trajectory 格式错误（必须为dict或str）")
            return raw_traj, False

        # 提取 lastHistory, lastAction, actionPurpose
        session_id = trajectory.get("sessionid", "")
        query = trajectory.get("instruction", "")
        plan = trajectory.get("plan", "")
        rule_id = trajectory.get("ruleId", "")
        intent_scene = trajectory.get("intentScene", "")
        app = trajectory.get("targetAppName", "")
        trajectory_str = trajectory.get("lastHistory", "")
        last_action = trajectory.get("lastAction", "")
        action_purpose = trajectory.get("actionPurpose", "")
        last_model_input = trajectory.get("lastModelInput", "")

        if not trajectory or not last_model_input or not last_action:
            self.log(
                f"Traj:{session_id} 标准化失败\n Trajectory Normalization 轨迹中不存在必要字段(lastHistory/lastModelInput/lastAction)")
            return raw_traj, False

        part_1 = {
            "sessionid": session_id,
            "instruction": query,
            "plan": plan,
            "rule_id": rule_id,
            "app": app,
            "intent_scene": intent_scene
        }

        cleaned_str = re.sub(r'<\|vision_start\|>.*?<\|vision_end\|>', '', trajectory_str, flags=re.DOTALL)

        cleaned_str = cleaned_str.strip()
        if cleaned_str.startswith('{') and cleaned_str.endswith('}'):
            cleaned_str = cleaned_str[1:-1].strip()

        steps = []
        step_counter = 1

        pattern = re.compile(
            r'\[(Thought|[^\[\]]*(?:执行动作|手动操作|动作|执行操作)):\s*(.*?)\]'
            r'(?=\s*(?:\[|</?br>|<\\br>|页面\d+:|$|<\|vision))',
            re.DOTALL
        )

        current_thought = None

        for match in pattern.finditer(cleaned_str):
            block_type = match.group(1).strip()
            content = match.group(2).strip()

            if block_type == "Thought":
                if current_thought is not None:
                    steps.append({
                        "step_id": step_counter,
                        "thought": current_thought,
                        "action": "plan"
                    })
                    step_counter += 1

                current_thought = content

            else:
                step_data = {
                    "step_id": step_counter,
                    "thought": current_thought
                }

                # 动态键名分流逻辑
                if "ManagerAgent" in block_type:
                    step_data["manager_action"] = content
                else:
                    step_data["action"] = content

                steps.append(step_data)
                step_counter += 1
                current_thought = None

        if current_thought is not None:
            steps.append({
                "step_id": step_counter,
                "thought": current_thought,
                "action": "plan"
            })
            step_counter += 1

        if last_action is not None or action_purpose is not None:
            steps.append({
                "step_id": step_counter,
                "thought": action_purpose,
                "action": last_action
            })

        part_2 = {
            f"step{idx}": {
                "manager_action": step.get("manager_action", ""),
                "thought": step.get("thought", ""),
                "action": step.get("action", "")
            } for idx, step in enumerate(steps, start=1)
        }

        if not part_2:
            return raw_traj, False

        return {**part_1, **part_2}, True

    def _normalize_trajectory_with_validation(
            self,
            raw_traj: t.Union[str, t.Dict[str, t.Any]]
    ) -> tuple:

        try:
            # 1. 轨迹标准化
            traj, success_status = self.trajectory_normalize(raw_traj)
            if not traj:
                traj = {"parse_failed_reason": "轨迹数据异常，解析无结果"}
                return "parse_failed", traj

            if not success_status:
                traj.update({"parse_failed_reason": "轨迹数据异常，解析无结果"})
                return "parse_failed", traj

            # 2. 拒答检测
            valid_step = self.calc_valid_step(traj)
            if valid_step <= self.min_validate_step:

                if "lastAction" not in raw_traj:
                    rejection_reason = "轨迹异常/不完整"
                    rejection_type = "轨迹不完整"
                else:
                    last_action_output = raw_traj.get("lastAction", "").strip()

                    # 【修复1】：用更鲁棒的正则，取代脆弱的 split("(")[0]，无视前面的标号、空格或前缀
                    if re.search(r'\b(preCheckDone|done)\s*\(', last_action_output, re.IGNORECASE):

                        # ==================== 1. OOD（Agent 自身能力与动作空间超纲） ====================
                        ood_patterns = [
                            # 【修复2】：将 .*? 替换为 [^。？！]*?，防止跨越句号产生误杀
                            r"不支持[^。？！]*?(指令|功能|手势|线下操作)",
                            r"当前能力不支持",
                            r"(不能|不会)(由我)?(完全)?代为?(打|玩|操作|执行|发布)",
                            r"(属于|是|涉及|不支持)[^。？！]*?(物理|线下)操作",
                            r"(属于|是|需要)[^。]*?系统(级)?[^。？！]*?(操作|功能|多窗|多任务|窗口管理|高风险操作)",
                            r"无法(自动|直接)?(执行|调用|触发|完成|控制)[^。？！]*?系统(分屏|截图|多任务|操作|关机)",
                            r"(需要|建议|请)[您你]?(在|到)?[^。？！]*?(手机)?系统设置(中|里)[^。？！]*?(开启|切换|完成|修改|操作)",
                            r"无法在[^。？！]*?内生成并保存音频",
                            r"无法[^。？！]*?直接提取视频中说话的文字"
                        ]

                        # ==================== 2. 环境/业务/外部约束（失败但非 OOD） ====================
                        env_patterns = [
                            r"未安装[^。？！]*?应用",
                            # 【修复1】：将变长断言拆分为 3 个独立的固定长度断言，完美绕过 Python 限制
                            r"(?<!未出现)(?<!没有)(?<!无)(启动|页面|应用|状态)?卡(住|死)",
                            r"停留在(启动页|黑色页面)",
                            r"多次等待仍停留",
                            r"(需要|请)[^。？！]*?(手机号|扫码|验证码)?登录",
                            # 【修复2】：将 | 替换为正则字符集 []，彻底兼容所有 Python 版本
                            r"(?<![未不])涉及[^。？！]*?(账号隐私|账号授权|付款|支付|财产|资金|生物识别)",
                            r"(未开放|已下架|临时取消|不再有效|门店已打烊|不可用|已拉黑)",
                            r"未到[^。？！]*?指定(的)?时间",
                            r"没有看到已打开的(照片|图片)",
                            r"(尚未|暂未)[^。？！]*?(提交|发送|完成)",
                            r"(不再|不能|无法)继续(执行|操作|完成)",
                            r"(任务已?(停止|终止|取消)|(停止|退出|取消)任务)",
                            r"(没有|未发现|未出现|未看到|未找到|未进入)[^。？！]*?(入口|界面|开关|设置项|功能|设备|文件)(?![，。]*?(因此|说明|所以|这说明))",
                            r"(均)?未(能)?生效|未进入全屏模式|未成功切换",
                            r"无法(在[^。？！]*?内)?(直接|自动)?(点击|点开|查看|设置|缩小|保存|退出|关闭|进入|发送|获取)",
                            r"([您你]?(需要|请|需|接下来|之后|后续|可以)[您你]?)[^。？！]*?(手动|自行|自己去|上传|补充|填写)",
                            r"(?<!不)建议[您你]?[^。？！]*?(手动|自行(?!使用)|自己去|您自己|签到开放时间|第三方|其他工具)"
                        ]

                        # 【修复6】：宽容提取括弧里的内容，防止末尾多余空格导致 match 失败
                        match = re.search(r'(?:preCheckDone|done)\s*\((.*)\)', last_action_output, re.DOTALL)
                        if match:
                            content = match.group(1).strip('"\'')
                        else:
                            content = last_action_output

                        # 判定优先级
                        if any(re.search(p, content) for p in ood_patterns):
                            rejection_reason = f"因自身能力不支持(OOD)导致任务结束，轨迹步数小于分析阈值({self.min_validate_step})"
                            rejection_type = "自身能力不支持"
                        elif any(re.search(p, content) for p in env_patterns):
                            rejection_reason = f"因执行环境受限导致任务结束，轨迹步数小于分析阈值({self.min_validate_step})"
                            rejection_type = "执行环境受限"
                        else:
                            rejection_reason = f"因任务过于简单(真正完成)，轨迹步数小于分析阈值({self.min_validate_step})"
                            rejection_type = "任务过于简单"
                    else:
                        rejection_reason = "轨迹异常/不完整"
                        rejection_type = "轨迹不完整"

                traj.update({"rejection_reason": rejection_reason})
                traj.update({"rejection_type": rejection_type})
                return "rejection_case", traj

            # 3. 全部校验通过
            return "success_parsing", traj

        except Exception as e:
            # 捕获由于脏数据导致的 JSONDecodeError, KeyError 等
            self.log(f"轨迹后台处理发生异常: {e}")
            return "process_error", {"error": str(e), "raw": raw_traj}

    async def _build_or_load_global_intent_cache(
            self,
            input_file_paths: t.List[str | Path]
    ) -> t.Dict[str, dict]:
        """
        第一阶段核心：流式扫描所有文件，收集唯一 (App, Instruction)，
        比对本地 global_instruction_cache.jsonl，对 Miss 的指令以 Batch 流水线方式
        并发执行 (LLM提取 -> Embedding向量化 -> 队列异步安全落盘)。
        """
        t_scan_start = time.perf_counter()
        cache_file = self.global_instruction_cache_path
        global_cache = {}
        processed_cache_keys = set()  # 用于充当 writer 协程中的 processed_session_ids

        # 1. 加载本地已有的全局缓存
        if cache_file.exists():
            self.log(f"正在加载全局意图缓存: {cache_file}")
            with open(cache_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in tqdm(lines, desc="加载历史缓存", unit="line"):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    key = item.get("cache_key")
                    if key:
                        global_cache[key] = item
                        processed_cache_keys.add(key)  # 记录已存在的 key
                except json.JSONDecodeError:
                    continue
            self.log(f"成功加载历史缓存条目数: {len(global_cache)}")

        # 2. 流式扫描所有输入文件，收集本次涉及的所有唯一 Key
        unique_tasks_map = {}

        for path in tqdm(input_file_paths, desc="扫描输入文件", unit="file"):
            data = self.load_json_data(path)
            for raw_traj in data:
                app = str(raw_traj.get("targetAppName", "") or raw_traj.get("app", "")).strip()
                instruction = str(raw_traj.get("instruction", "")).strip()
                if not instruction or not app:
                    continue

                key = get_normalized_key(app, instruction)
                if key not in unique_tasks_map and key not in processed_cache_keys:
                    unique_tasks_map[key] = {
                        "cache_key": key,
                        "app": app,
                        "instruction": instruction
                    }

        self.log(f"发现历史未命中、需要新增计算的唯一指令数: {len(unique_tasks_map)}")
        self._set_wall_time("stage1_1_load_and_scan_cache", time.perf_counter() - t_scan_start)  # <--- 【记录扫描耗时】

        # 3. 启动队列与后台写入协程，并发执行 (LLM + Embedding)
        if unique_tasks_map:
            t_pipe_start = time.perf_counter()  # <--- 【新增计时点 2：增量流水线开始】
            new_items_list = list(unique_tasks_map.values())
            batch_size = 15

            intent_write_queue = asyncio.Queue()
            intent_writer_task = asyncio.create_task(
                self._intent_clustering_writer_task(
                    queue=intent_write_queue,
                    processed_session_ids=processed_cache_keys,
                    global_cache=global_cache
                )
            )

            async def _process_and_embed_batch(sub_items: t.List[dict]):
                batch_inputs = [
                    {"id": f"item_{idx}", "instruction": item["instruction"]}
                    for idx, item in enumerate(sub_items)
                ]

                # 3.1 批量提取分类与意图（带计时与 Token 统计）
                t_llm_0 = time.perf_counter()
                res = await self._process_batch_classify_and_intent(batch_inputs, count_usage=self.count_usage)
                self._add_cum_time("stage1_llm_classify_and_intent_cum", time.perf_counter() - t_llm_0)

                if isinstance(res, tuple):
                    batch_inputs, batch_usage = res
                    self._add_usage("stage1_classify_and_intent", batch_usage)

                intents_to_embed = []
                for idx, item in enumerate(sub_items):
                    extracted = batch_inputs[idx]
                    intent_text = extracted.get("intent", "")
                    category_info = extracted.get("category", {})

                    item["intent"] = intent_text
                    item["category"] = category_info
                    intents_to_embed.append(intent_text)

                # 3.2 立即对该 Batch 进行 Embedding（带计时）
                t_emb_0 = time.perf_counter()
                embeddings = await self._intent_embedding(intents_to_embed)
                self._add_cum_time("stage1_intent_embedding_cum", time.perf_counter() - t_emb_0)

                for idx, item in enumerate(sub_items):
                    if embeddings and idx < len(embeddings) and embeddings[idx]:
                        item["intent_embedding"] = embeddings[idx]
                    else:
                        item["intent_embedding"] = []

                # 3.3 将处理完毕的整个 Batch 放入队列
                await intent_write_queue.put(sub_items)

            tasks = [
                _process_and_embed_batch(new_items_list[i: i + batch_size])
                for i in range(0, len(new_items_list), batch_size)
            ]

            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="LLM提取 & Embedding 流水线",
                             unit="batch"):
                await coro

            await intent_write_queue.put(None)
            await intent_writer_task

            self._set_wall_time("stage1_1_llm_and_embed_pipeline", time.perf_counter() - t_pipe_start)  # <--- 【记录流水线耗时】
            self.log(f"增量计算与队列落盘完成，当前全局缓存总条目数: {len(global_cache)}")

        return global_cache

    async def _run_global_clustering_on_cache(
            self,
            global_cache: t.Dict[str, dict],
            dpp_k: int = 30,
            force_recluster: bool = False
    ):
        """
        第三步核心：针对全局缓存中的所有唯一指令进行整体聚类重算、LLM 命名总结及同义合并，
        并将脱敏（不含 embedding）的轻量结果写入 global_intent_cache.jsonl 中。
        支持缓存短路机制：若本地已有完整覆盖当前指令集的聚类结果，则直接加载跳过重算。
        """
        t_cluster_all_start = time.perf_counter()
        items_list = list(global_cache.values())
        n_samples = len(items_list)
        if n_samples == 0:
            self.log("全局缓存为空，跳过聚类。")
            return global_cache

        # =====================================================================
        # 0. 聚类结果缓存校验与短路加载机制
        # =====================================================================
        if not force_recluster and self.global_intent_cache_path.exists():
            try:
                cached_intent_map = {}
                with open(self.global_intent_cache_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            k = obj.get("cache_key")
                            if k and "cluster_label" in obj and obj.get("intent_cluster_name"):
                                cached_intent_map[k] = obj
                        except json.JSONDecodeError:
                            continue

                current_keys = set(global_cache.keys())
                if current_keys and current_keys.issubset(cached_intent_map.keys()):
                    self.log(
                        f"✅ 检测到本地全局意图聚类缓存完整命中 ({len(current_keys)} 条唯一指令)，"
                        f"直接加载 {self.global_intent_cache_path.name} 并跳过聚类重算！"
                    )
                    for k, item in global_cache.items():
                        item.pop("intent_embedding", None)
                        cached_item = cached_intent_map[k]
                        item["cluster_label"] = int(cached_item["cluster_label"])
                        item["intent_cluster_name"] = cached_item["intent_cluster_name"]
                        if "category" not in item and "category" in cached_item:
                            item["category"] = cached_item["category"]
                        if "intent" not in item and "intent" in cached_item:
                            item["intent"] = cached_item["intent"]

                    self._set_wall_time("stage1_2_total_clustering", time.perf_counter() - t_cluster_all_start)
                    return global_cache
                else:
                    missing_cnt = len(current_keys - cached_intent_map.keys())
                    self.log(f"🔄 检测到 {missing_cnt} 条新增/未聚类的唯一指令，触发全局聚类重算...")
            except Exception as e:
                self.log(f"[全局聚类缓存加载异常] 将回退至全量重算: {e}")

        self.log("开始执行全局唯一的指令聚类重算...")

        # 1. 确定向量维度并构建全局归一化特征矩阵
        vector_dim = 1024
        for item in items_list:
            emb = item.get('intent_embedding')
            if emb and isinstance(emb, list) and len(emb) > 0:
                vector_dim = len(emb)
                break

        global_X = np.zeros((n_samples, vector_dim), dtype=np.float32)
        vector_success_map = np.zeros(n_samples, dtype=bool)

        for i, item in enumerate(items_list):
            emb = item.pop('intent_embedding', None)
            if emb and isinstance(emb, list) and len(emb) == vector_dim:
                vec = np.asarray(emb, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 1e-9 and not np.isnan(norm):
                    global_X[i] = vec / norm
                    vector_success_map[i] = True
                else:
                    vector_success_map[i] = False
            else:
                vector_success_map[i] = False

        # 2. 按场景 (category) 和 app 分桶
        grouped_data = defaultdict(lambda: defaultdict(list))
        for i, item in enumerate(items_list):
            cat_obj = item.get('category', {})
            scenario = cat_obj.get('category', 'unknown_scenario') if isinstance(cat_obj, dict) else 'unknown_scenario'
            app = str(item.get('app', '')).strip() or 'unknown_app'
            grouped_data[scenario][app].append({
                'global_index': i,
                'item': item,
                'success': bool(vector_success_map[i])
            })

        # 3. 局部层次聚类（【耗时统计】子计时 1：CPU 层次聚类开始）
        t_cpu_start = time.perf_counter()
        loop = asyncio.get_running_loop()
        cluster_futures = []
        cluster_meta = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as thread_pool:
            for scenario, apps in grouped_data.items():
                for app, items in apps.items():
                    if not items:
                        continue

                    valid_items = [it for it in items if it['success']]
                    invalid_items = [it for it in items if not it['success']]

                    for it in invalid_items:
                        it['item']['cluster_label'] = -1

                    if len(valid_items) == 1:
                        valid_items[0]['item']['cluster_label'] = 0
                    elif len(valid_items) >= 2:
                        group_indices = [it['global_index'] for it in valid_items]
                        local_X = global_X[group_indices]

                        fut = loop.run_in_executor(
                            thread_pool, _process_cluster_worker, local_X, False
                        )
                        cluster_futures.append(fut)
                        cluster_meta.append(valid_items)

            if cluster_futures:
                self.log(f"正在并发执行 {len(cluster_futures)} 个 (场景-App) 分桶的层次聚类...")
                cluster_results = await asyncio.gather(*cluster_futures)
                for valid_items, (labels, _) in zip(cluster_meta, cluster_results):
                    for it, label in zip(valid_items, labels):
                        it['item']['cluster_label'] = int(label)

        self._set_wall_time("stage1_2_local_clustering_cpu", time.perf_counter() - t_cpu_start)

        # 4. LLM 聚类簇命名总结（【耗时统计】子计时 2：LLM 命名开始）
        t_naming_start = time.perf_counter()
        task_args = []
        summary_dict = defaultdict(lambda: defaultdict(dict))

        for scenario, apps in grouped_data.items():
            for app, items in apps.items():
                cluster_groups = defaultdict(list)
                for it in items:
                    c_id = it['item'].get('cluster_label', -1)
                    if c_id != -1 and it['success']:
                        cluster_groups[c_id].append(it)

                for c_id, group_items in cluster_groups.items():
                    if len(group_items) == 1:
                        single_intent = (group_items[0]['item'].get('intent') or "").strip()
                        summary_dict[scenario][app][c_id] = single_intent if single_intent else f"类别 {c_id}"
                        continue

                    g_indices = [it['global_index'] for it in group_items]
                    cluster_matrix = global_X[g_indices]

                    sampled_local_indices = self._dpp_sample(cluster_matrix, k=dpp_k)
                    sampled_intents = []
                    for idx in sampled_local_indices:
                        it = group_items[idx]
                        txt = (it['item'].get('intent') or "").strip()
                        if txt:
                            sampled_intents.append(txt)

                    unique_sampled = list(dict.fromkeys(sampled_intents))
                    if not unique_sampled:
                        summary_dict[scenario][app][c_id] = f"类别 {c_id}"
                        continue
                    if len(unique_sampled) == 1:
                        summary_dict[scenario][app][c_id] = unique_sampled[0]
                        continue

                    intents_str = "\n".join([f"- {txt}" for txt in unique_sampled])
                    prompt = build_intent_summary_prompt(intents_str)
                    task_args.append(((scenario, app, c_id), prompt))

        if task_args:
            self.log(f"共生成 {len(task_args)} 个多样本簇 LLM 命名任务，开始并发执行...")

            async def run_and_update(meta, prompt_text, pbar):
                try:
                    # 【Token 统计】传入 count_usage 并安全解包累加
                    res = await self.llm.ainvoke(
                        prompt=prompt_text,
                        temperature=0.2,
                        count_usage=self.count_usage
                    )
                    if isinstance(res, tuple):
                        res, usage = res
                        self._add_usage("stage1_cluster_naming", usage)
                except Exception as e:
                    res = e
                pbar.update(1)
                return res

            with tqdm(total=len(task_args), desc="Global LLM 命名") as pbar:
                llm_tasks = [run_and_update(meta, prompt, pbar) for meta, prompt in task_args]
                results = await asyncio.gather(*llm_tasks)

            for meta, res in zip([t[0] for t in task_args], results):
                scenario, app, c_id = meta
                if isinstance(res, Exception) or not res or not isinstance(res, str):
                    summary_dict[scenario][app][c_id] = f"类别 {c_id}"
                else:
                    clean_name = res.strip().replace('"', '').replace("'", "").splitlines()[0].strip()
                    summary_dict[scenario][app][c_id] = clean_name if clean_name else f"类别 {c_id}"

        self._set_wall_time("stage1_2_cluster_llm_naming", time.perf_counter() - t_naming_start)

        # 5. 后置同义簇合并（【耗时统计】子计时 3：同义簇合并开始）
        t_merge_start = time.perf_counter()
        global_cluster_id_map = defaultdict(lambda: defaultdict(dict))
        embed_batch_size = 64

        for scenario, apps in summary_dict.items():
            for app, c_name_map in apps.items():
                c_ids = list(c_name_map.keys())
                for cid in c_ids:
                    global_cluster_id_map[scenario][app][cid] = cid

                if len(c_ids) < 2:
                    continue

                names = [c_name_map[cid] for cid in c_ids]
                try:
                    raw_embeddings_list = []
                    for start_idx in range(0, len(names), embed_batch_size):
                        batch_names = names[start_idx: start_idx + embed_batch_size]
                        batch_embs = await self._intent_embedding(batch_names)
                        raw_embeddings_list.extend(batch_embs)

                    if (
                            len(raw_embeddings_list) != len(names)
                            or any(not emb or len(emb) != vector_dim for emb in raw_embeddings_list)
                    ):
                        self.log(f"[同义簇合并跳过] {scenario}-{app} 部分名称 Embedding 失败或维度不一致")
                        continue

                    name_embeddings = np.asarray(raw_embeddings_list, dtype=np.float32)
                    name_norms = np.linalg.norm(name_embeddings, axis=1, keepdims=True)
                    name_norms[name_norms == 0] = 1.0
                    normed_name_embs = name_embeddings / name_norms
                    sim_matrix = normed_name_embs @ normed_name_embs.T
                except Exception as e:
                    self.log(f"[同义簇合并异常] {scenario}-{app}: {e}")
                    continue

                cluster_sizes = defaultdict(int)
                for it in grouped_data[scenario][app]:
                    c_lbl = it['item'].get('cluster_label', -1)
                    if c_lbl != -1:
                        cluster_sizes[c_lbl] += 1

                sorted_idx = sorted(range(len(c_ids)), key=lambda idx: cluster_sizes[c_ids[idx]], reverse=True)

                processed = set()
                for i in sorted_idx:
                    if i in processed:
                        continue
                    main_c_id = c_ids[i]
                    main_name = c_name_map[main_c_id]
                    processed.add(i)

                    sim_indices = np.where(sim_matrix[i] >= 0.85)[0]
                    for j in sim_indices:
                        if j not in processed:
                            sub_c_id = c_ids[j]
                            c_name_map[sub_c_id] = main_name
                            global_cluster_id_map[scenario][app][sub_c_id] = main_c_id
                            processed.add(j)

        self._set_wall_time("stage1_2_synonym_cluster_merge", time.perf_counter() - t_merge_start)

        # 6. 将最终的 cluster_label 与 intent_cluster_name 回写到各个 item 中
        for scenario, apps in grouped_data.items():
            for app, items in apps.items():
                for it in items:
                    item = it['item']
                    old_c_id = item.get('cluster_label', -1)
                    if old_c_id == -1:
                        item['intent_cluster_name'] = "未成簇意图/噪声"
                    else:
                        new_c_id = global_cluster_id_map[scenario][app].get(old_c_id, old_c_id)
                        item['cluster_label'] = int(new_c_id)
                        item['intent_cluster_name'] = summary_dict[scenario][app].get(new_c_id, f"类别 {new_c_id}")

        # 7. 将不含 intent_embedding 的轻量级结果写入 global_intent_cache.jsonl
        self.log(f"正在写入轻量级全局意图聚类结果文件: {self.global_intent_cache_path}")
        with open(self.global_intent_cache_path, "w", encoding="utf-8") as f_out:
            for item in items_list:
                clean_item = {k: v for k, v in item.items() if k != "intent_embedding"}
                f_out.write(json.dumps(clean_item, ensure_ascii=False) + "\n")

        self._set_wall_time("stage1_2_total_clustering",
                            time.perf_counter() - t_cluster_all_start)  # <--- 【耗时统计】Stage 1.2 总计时结束
        self.log("全局指令聚类重算与缓存刷新完成！")
        return global_cache

    async def _run_trajectory_verdict(self, input_file_paths: t.List[str | Path], global_cache: t.Dict[str, dict]):
        """
        第二阶段核心：流式读取原始轨迹，【全量无差别】查表挂载第一阶段的聚类意图标签，
        然后执行轨迹校验、经验召回（含独立断点缓存）及 Verdict 判定，并流式落盘。
        """
        t_stage2_start = time.perf_counter()  # <--- 【耗时统计】Stage 2 总计时开始
        self.log(f"开始执行第二阶段，输出目录为: {self.save_path.resolve()}")

        # 惰性初始化 plan_recall 缓存路径
        plan_recall_cache_path = getattr(
            self, "plan_recall_cache_path", self.save_path / "plan_recall_cache.jsonl"
        )

        # =========================================================================
        # 1. 恢复断点与去重集合
        # =========================================================================
        processed_session_ids = set()
        paths_to_check = [
            self.success_trajectories_path, self.failure_trajectories_path,
            self.ineffective_plan_recall_trajectories_path,
            self.rejection_trajectories_path, self.parse_failed_trajectories_path
        ]
        session_id_pattern = re.compile(r'"sessionid"\s*:\s*"([^"]+)"')
        for path in paths_to_check:
            if path and path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        match = session_id_pattern.search(line)
                        if match:
                            processed_session_ids.add(match.group(1))

        if processed_session_ids:
            self.log(f"⚠️ 检测到历史已处理轨迹数: {len(processed_session_ids)} 条（将在本次运行中自动跳过）")

        # =========================================================================
        # 2. 启动后台异步写入队列 (Consumer)
        # =========================================================================
        write_queue = asyncio.Queue()
        writer_task = asyncio.create_task(self._stream_file_writer_task(write_queue, processed_session_ids))

        # =========================================================================
        # 3. 逐个文件读取，【先注入聚类结果】，再进行合法性分流
        # =========================================================================
        t_prep_start = time.perf_counter()  # <--- 【耗时统计】子计时 1：预处理开始
        valid_candidates = []
        status_counter = {"parse_failed": 0, "rejection_case": 0, "valid": 0, "skipped": 0}

        for path_str in input_file_paths:
            self.log(f"第二阶段正在扫描并挂载文件: {path_str}")
            data = self.load_json_data(path_str)
            if not data:
                self.log(f"⚠️ 文件为空或读取失败: {path_str}")
                continue

            for raw_traj in tqdm(data, desc="预处理与挂载聚类标签", unit="traj"):
                if isinstance(raw_traj, str):
                    try:
                        raw_traj_dict = json.loads(raw_traj)
                    except Exception:
                        raw_traj_dict = {}
                elif isinstance(raw_traj, dict):
                    raw_traj_dict = raw_traj
                else:
                    raw_traj_dict = {}

                session_id = raw_traj_dict.get("sessionid", "")
                if session_id and session_id in processed_session_ids:
                    status_counter["skipped"] += 1
                    continue

                # 3.1 执行轨迹合法性与拒答校验
                status, traj = self._normalize_trajectory_with_validation(raw_traj)
                if not isinstance(traj, dict):
                    traj = {"raw": str(traj), "parse_failed_reason": "返回格式非字典"}

                app = str(
                    traj.get("app") or raw_traj_dict.get("targetAppName") or raw_traj_dict.get("app") or "").strip()
                instruction = str(traj.get("instruction") or raw_traj_dict.get("instruction") or "").strip()
                if "sessionid" not in traj and session_id:
                    traj["sessionid"] = session_id
                if "app" not in traj and app:
                    traj["app"] = app
                if "instruction" not in traj and instruction:
                    traj["instruction"] = instruction

                # 3.2 全量注入第一阶段的聚类结果
                if app and instruction:
                    cache_key = get_normalized_key(app, instruction)
                    cached_info = global_cache.get(cache_key, {})
                else:
                    cached_info = {}

                traj['category'] = cached_info.get("category", {"category": "unknown_scenario"})
                traj['intent'] = cached_info.get("intent", instruction)
                traj['cluster_label'] = cached_info.get("cluster_label", -1)
                traj['intent_cluster_name'] = cached_info.get("intent_cluster_name", "未成簇意图/噪声")

                # 3.3 分流
                if status == "rejection_case":
                    status_counter["rejection_case"] += 1
                    await write_queue.put(("rejection_case", traj))
                elif status == "success_parsing":
                    status_counter["valid"] += 1
                    valid_candidates.append(traj)
                else:
                    status_counter["parse_failed"] += 1
                    await write_queue.put(("parse_failed", traj))

            await asyncio.sleep(0.1)

        self._set_wall_time("stage2_0_preprocess_and_normalize",
                            time.perf_counter() - t_prep_start)  # <--- 【耗时统计】记录预处理耗时
        self.log(
            f"📊 预处理统计 -> 跳过已存在: {status_counter['skipped']} | 解析失败: {status_counter['parse_failed']} | 拒答/短轨迹: {status_counter['rejection_case']} | 有效待评估: {status_counter['valid']}")

        # =========================================================================
        # 4. 异步批处理：经验召回判定 (Plan Recall)
        # =========================================================================
        t_recall_start = time.perf_counter()
        verdict_candidates = []
        if valid_candidates:
            recall_cache_map: t.Dict[str, dict] = {}
            if plan_recall_cache_path.exists():
                try:
                    with open(plan_recall_cache_path, "r", encoding="utf-8") as f_rc:
                        for line in f_rc:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                                sid = obj.get("sessionid")
                                eff = obj.get("plan_recall_effectiveness")
                                if sid and isinstance(eff, dict) and "judgement" in eff:
                                    recall_cache_map[sid] = eff
                            except json.JSONDecodeError:
                                continue
                    if recall_cache_map:
                        self.log(f"✅ 成功加载历史 Plan Recall 缓存条目数: {len(recall_cache_map)}")
                except Exception as e:
                    self.log(f"[Plan Recall 缓存加载异常] {e}")

            recall_batch_size = 15
            recall_cache_Hit_count = 0

            with open(plan_recall_cache_path, "a", encoding="utf-8") as f_recall_out:

                async def _trajectory_recall_check(batch_data: list):
                    nonlocal recall_cache_Hit_count
                    valid_for_verdict = []
                    new_cache_lines = []

                    for traj in batch_data:
                        sid = traj.get("sessionid", "")

                        if sid and sid in recall_cache_map:
                            recall_cache_Hit_count += 1
                            traj['plan_recall_effectiveness'] = recall_cache_map[sid]
                        else:
                            try:
                                # judgement, reason = await self.plan_recall_effectiveness_judge(traj)
                                judgement, reason = True, ""
                            except Exception as e:
                                judgement, reason = True, str(e)

                            eff_res = {"judgement": judgement, "judge_reason": str(reason)}
                            traj['plan_recall_effectiveness'] = eff_res

                            if sid and "embedding 失效" not in str(reason):
                                recall_cache_map[sid] = eff_res
                                new_cache_lines.append(
                                    json.dumps(
                                        {"sessionid": sid, "plan_recall_effectiveness": eff_res},
                                        ensure_ascii=False
                                    ) + "\n"
                                )

                        if traj['plan_recall_effectiveness']['judgement'] is False:
                            await write_queue.put(("ineffective_recall", traj))
                        else:
                            valid_for_verdict.append(traj)

                    if new_cache_lines:
                        try:
                            f_recall_out.writelines(new_cache_lines)
                            f_recall_out.flush()
                        except Exception as e:
                            self.log(f"[写入 Plan Recall 缓存失败] {e}")

                    return valid_for_verdict

                recall_tasks = [
                    _trajectory_recall_check(valid_candidates[i: i + recall_batch_size])
                    for i in range(0, len(valid_candidates), recall_batch_size)
                ]

                for task in tqdm(asyncio.as_completed(recall_tasks), total=len(recall_tasks),
                                 desc="Phase 2.1: Plan Recall",
                                 unit="batch"):
                    result_batch = await task
                    verdict_candidates.extend(result_batch)

            if recall_cache_Hit_count > 0:
                self.log(f"⚡ Phase 2.1 缓存命中跳过: {recall_cache_Hit_count} 条轨迹 (免除重复检索与 Embedding)")

        self._set_wall_time("stage2_1_plan_recall",
                            time.perf_counter() - t_recall_start)  # <--- 【耗时统计】记录 Plan Recall 耗时

        # =========================================================================
        # 5. 异步批处理：Verdict 判定 (双层执行正确性)
        # =========================================================================
        t_verdict_start = time.perf_counter()  # <--- 【耗时统计】子计时 3：Verdict 判定开始
        if verdict_candidates:
            verdict_batch_size = 5

            async def _trajectory_verdict_process(batch_data: list):
                t_v_0 = time.perf_counter()
                # 显式传入 count_usage=self.count_usage 并解包统计
                res = await self._process_batch_verdict(batch_data, count_usage=self.count_usage)
                self._add_cum_time("stage2_verdict_llm_cum", time.perf_counter() - t_v_0)

                if isinstance(res, tuple):
                    batch_data, batch_usage = res
                    self._add_usage("stage2_verdict", batch_usage)  # <--- 【Token 统计】累加 Verdict 阶段 Token
                else:
                    batch_data = res

                for traj in batch_data:
                    session_id = traj.get("sessionid", "unknown")
                    verdict = traj.get('verdict', {})

                    # 显式三态判定：task_completed 只信任真正的布尔值，无法确定时按
                    # 解析失败处理，严禁默认当作成功（详见 resolve_task_completed 注释）
                    task_completed, invalid_reason = resolve_task_completed(verdict)

                    if invalid_reason is not None:
                        traj["verdict_invalid_reason"] = invalid_reason
                        await write_queue.put(("parse_failed", traj))
                        continue

                    if task_completed is False:
                        failure_traj = await self._process_ood_trajectory(traj)
                        await write_queue.put(("failure", failure_traj))
                    else:
                        success_traj = await self._process_ind_trajectory(traj)
                        if "sessionid" not in success_traj and session_id:
                            success_traj["sessionid"] = session_id
                        await write_queue.put(("success", success_traj))

            verdict_tasks = [
                _trajectory_verdict_process(verdict_candidates[i: i + verdict_batch_size])
                for i in range(0, len(verdict_candidates), verdict_batch_size)
            ]

            for task in tqdm(asyncio.as_completed(verdict_tasks), total=len(verdict_tasks), desc="Phase 2.2: Verdict",
                             unit="batch"):
                await task

        # =========================================================================
        # 6. 收尾：等待队列清空并安全关闭写入协程
        # =========================================================================
        await write_queue.put(None)
        await writer_task
        self._set_wall_time("stage2_2_trajectory_verdict",
                            time.perf_counter() - t_verdict_start)  # <--- 【耗时统计】记录 Verdict 墙钟耗时
        self._set_wall_time("stage2_total", time.perf_counter() - t_stage2_start)  # <--- 【耗时统计】Stage 2 总计时结束
        self.log("第二阶段 Verdict 判定及全量结果落盘完成！")


    async def _process_batch_classify_and_intent(self, batch_data: t.List[dict], count_usage: bool=False):
        """
        批量处理 Category 和 Intent (因为 instruction 通常很短，Batch Size 可以设大点，如 10-20)
        """
        token_usage = {
            "total_tokens": 0,
            "completion_tokens": 0,
        } if count_usage else None

        inputs = []
        short_id_to_traj = {}  # 建立 短ID -> 原始traj对象 的内存映射

        # 1. 构建带有短 ID 的输入结构
        for idx, traj in enumerate(batch_data):
            short_id = f"traj_{idx}"
            short_id_to_traj[short_id] = traj  # 记录引用

            inputs.append({
                "id": short_id,
                "instruction": traj.get("instruction", "")
            })

        prompt_category = build_batch_scenario_classify_prompt(inputs)
        prompt_intent = build_batch_intent_recognition_prompt(inputs)

        # 2. 并发执行这两项批处理任务
        res_category, res_intent = await asyncio.gather(
            self.llm.ainvoke(prompt_category, count_usage=count_usage),
            self.llm.ainvoke(prompt_intent, count_usage=count_usage)
        )

        if isinstance(res_category, tuple):
            res_category, category_usage = res_category
            if token_usage is not None and isinstance(category_usage, dict):
                token_usage["total_tokens"] += (category_usage.get("total_tokens") or 0)
                token_usage["completion_tokens"] += (category_usage.get("completion_tokens") or 0)

        if isinstance(res_intent, tuple):
            res_intent, intent_usage = res_intent
            if token_usage is not None and isinstance(intent_usage, dict):
                token_usage["total_tokens"] += (intent_usage.get("total_tokens") or 0)
                token_usage["completion_tokens"] += (intent_usage.get("completion_tokens") or 0)

        # 3. 解析返回的 JSON List，直接利用短 ID 映射写回原数据
        # 处理分类结果
        parsed_category_result = extract_json(res_category)
        if isinstance(parsed_category_result, dict):
            # 排除兜底返回的空字典 {}
            if not parsed_category_result:
                parsed_category_result = []
            else:
                parsed_category_result = [parsed_category_result]
        parsed_intent_result = extract_json(res_intent)
        if isinstance(parsed_intent_result, dict):
            # 排除兜底返回的空字典 {}
            if not parsed_intent_result:
                parsed_intent_result = []
            else:
                parsed_intent_result = [parsed_intent_result]

        for item in parsed_category_result:
            if not isinstance(item, dict):
                continue
            short_id = item.get('id')
            if short_id in short_id_to_traj:
                short_id_to_traj[short_id]['category'] = item

        for item in parsed_intent_result:
            if not isinstance(item, dict):
                continue
            short_id = item.get('id')
            if short_id in short_id_to_traj:
                short_id_to_traj[short_id]['intent'] = item.get("core_intent", "")

        if count_usage:
            return batch_data, token_usage
        return batch_data

    async def _process_batch_verdict(self, batch_data: t.List[dict], count_usage: bool = False):
        """
        批量处理 Verdict (因为包含完整的轨迹历史，Token数巨大，Batch Size 建议 2-5)
        """
        token_usage = {
            "total_tokens": 0,
            "completion_tokens": 0,
        } if count_usage else None

        inputs = []
        short_id_to_traj = {}

        # 1. 构建带有短 ID 的输入结构
        for idx, traj in enumerate(batch_data):
            short_id = f"traj_{idx}"
            short_id_to_traj[short_id] = traj

            inputs.append({
                "id": short_id,
                "trajectory_text": build_trajectory_text(traj, max_steps=80)
            })

        prompt_verdict = build_batch_verdict_prompt(inputs)

        # 显式接收 (content, usage) 二元组
        res_verdict = await self.llm.ainvoke(prompt_verdict, count_usage=count_usage)

        if isinstance(res_verdict, tuple):
            res_verdict, verdict_usage = res_verdict
            if token_usage is not None and isinstance(verdict_usage, dict):
                token_usage["total_tokens"] += (verdict_usage.get("total_tokens") or 0)
                token_usage["completion_tokens"] += (verdict_usage.get("completion_tokens") or 0)

        # 2. 解析返回的 JSON List，直接利用短 ID 映射写回原数据
        parsed_result = extract_json(res_verdict)

        if isinstance(parsed_result, dict):
            if not parsed_result:
                parsed_result = []
            else:
                parsed_result = [parsed_result]

        for item in parsed_result:
            if not isinstance(item, dict):
                continue
            short_id = item.get('id')
            if short_id in short_id_to_traj:
                short_id_to_traj[short_id]['verdict'] = item

        if count_usage:
            return batch_data, token_usage
        return batch_data

    @staticmethod
    def plan_effectiveness_detection(
            plan: str,
    ):
        if not plan:
            return False
        return not any(kw in plan for kw in (
            "不适用",
            "不直接适用",
            "不要进入",
            "执行时不要",
            "不完全匹配",
            "不匹配",
            "无直接匹配",
            "无直接对应关系",
            "无直接关系",
            "不直接相关",
            "无关",
            "不用于",
            "不应用于",
            "不直接复用",
            "不指导"
        ))

    async def plan_recall_effectiveness_judge(
            self,
            traj: t.Dict[str, t.Any],
            similarity_threshold: float = 0.95
    ):
        intent = traj.get('intent', '')
        app = traj.get('app', '')
        plan = traj.get('plan', '')

        if plan:
            plan_effectiveness = self.plan_effectiveness_detection(plan)
            if plan_effectiveness:
                return True, "检测到经验库存在可用 plan"

        if not app:
            return False, "app 字段不存在，数据异常"

        if not intent:
            return False, "intent 字段不存在，pipeline异常"

        query = f"{app} {intent}"

        try:
            # 【改动点】同步调用改为异步 await 调用，受控于 plan_query_sem 信号量
            recalled_plan = await query_plan(query)
        except Exception as e:
            return False, f"query_plan 远程检索失败: {e}"

        hits = recalled_plan.get('hits', [])

        # ---------------------------------------------------------
        # 第一步：严格精准匹配 app，并收集符合条件的 task
        # ---------------------------------------------------------
        candidate_tasks = []
        for hit in hits:
            hit_app_names = hit.get('app_name', [])
            task_text = hit.get('task', '')

            if not task_text:
                continue

            app_matched = False
            for name_str in hit_app_names:
                # 兼容形如 '浏览器,QQ浏览器,抖音' 的多APP合并字符串
                parsed_apps = [a.strip() for a in name_str.split(',')]
                if app in parsed_apps:
                    app_matched = True
                    break

            # 如果 APP 匹配成功，将 task 收集起来用于后续统一向量化
            if app_matched:
                candidate_tasks.append(task_text)

        # 如果没有一个 hit 匹配当前 app，直接判定为召回有效
        if not candidate_tasks:
            return True, "轨迹不携带plan，且库中不存在有效经验"

        # ---------------------------------------------------------
        # 第二步：批量调用 Embedding 接口
        # ---------------------------------------------------------
        # 将 query 放在索引 0 的位置，后面跟着所有候选的 task，只发起一次网络请求
        inputs_for_embedding = [query] + candidate_tasks

        retries = 3
        embeddings = None
        for attempt in range(retries):
            try:
                # 拿到锁才能发请求
                async with self.embedding_sem:
                    response = await self.embedding_client.embeddings.create(
                        input=inputs_for_embedding,
                        model="bge-m3"
                    )
                # 离开 async with，立刻释放锁
                embeddings = [data.embedding for data in response.data]
                break  # 成功则跳出重试循环
            except Exception as e:
                if attempt == retries - 1:
                    self.log(f"Embedding API call failed after {retries} retries: {e}")
                    return False, "embedding 失效，无法判断召回是否有效"
                # 在锁外部进行指数退避休眠
                await asyncio.sleep(2 ** attempt)

        if not embeddings:
            return False, "embedding 失效，无法判断召回是否有效"

        # 取出 query 向量与其他 task 向量
        query_embedding = np.array(embeddings[0])
        task_embeddings = [np.array(emb) for emb in embeddings[1:]]

        # ---------------------------------------------------------
        # 第三步：利用 Numpy 计算余弦相似度并阈值判定
        # ---------------------------------------------------------
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return False, "query norm 异常"

        for task_emb in task_embeddings:
            task_norm = np.linalg.norm(task_emb)
            if task_norm == 0:
                continue

            # Numpy 计算余弦相似度公式: (A·B) / (|A|*|B|)
            sim_score = np.dot(query_embedding, task_emb) / (query_norm * task_norm)

            # 只要有一个符合条件的 task 相似度达标，即判定为找到了对应的 plan，但是实际情况没召回，输出 False 表示召回无效
            if sim_score >= similarity_threshold:
                return False, "轨迹未携带 plan，但库中有可用plan，召回无效"

        return True, "轨迹未携带 plan，且库中无可用plan，召回有效"

    async def _intent_embedding(
            self,
            texts: t.List[str],
            model: str = "bge-m3"  # 替换为华为 MLOps 实际使用的 Embedding 模型名
    ) -> t.List[t.List[float]]:
        """
        对传入的文本列表进行批量 Embedding，包含并发控制与指数退避重试。
        """
        if not texts:
            return []

        # 1. 预处理：防止出现纯空字符串导致 API 报错 400 (Bad Request)
        # 如果意图为空，用一个空格代替以防整个 Batch 崩溃
        safe_texts = [text if text and text.strip() else " " for text in texts]

        max_retries = 3

        # 2. 获取并发锁（控制全局最大并发数）
        await self.embedding_sem.acquire()
        try:
            for attempt in range(max_retries):
                try:
                    # 3. 发起调用。OpenAI 兼容接口天然支持传入 list 进行批量 embedding
                    response = await self.embedding_client.embeddings.create(
                        model=model,
                        input=safe_texts
                    )

                    # 4. 解析结果：由于部分 API 可能不保证乱序返回，推荐根据 index 显式排序保证一一对应
                    sorted_data = sorted(response.data, key=lambda x: x.index)
                    embeddings = [item.embedding for item in sorted_data]

                    return embeddings

                except openai.RateLimitError as e:
                    self.log(f"[Embedding 触发限流 429] 准备重试 ({attempt + 1}/{max_retries})")

                except (openai.APIConnectionError, openai.InternalServerError) as e:
                    self.log(f"[Embedding 网络/服务端错误] {str(e)}, 准备重试 ({attempt + 1}/{max_retries})")

                except openai.BadRequestError as e:
                    self.log(f"[Embedding 客户端请求错误 400] 终止重试: {str(e)}")
                    break  # 比如输入文本超出了模型支持的最大 Token 长度，重试无用直接截断

                except Exception as e:
                    self.log(f"[Embedding 未知异常] {str(e)}")
                    break

                # 5. 指数退避休眠 (在获取锁的情况下等待，防止大量并发持续冲击 API)
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

            # 重试用尽或遇到不可恢复错误，返回与输入长度对应的空数组以防止上层索引越界报错
            self.log(f"[Embedding 失败] 最终未能获取到 Embedding，返回空值占位。")
            return [[] for _ in texts]

        finally:
            # 6. 释放信号量
            self.embedding_sem.release()

    async def _process_ind_trajectory(self, traj: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        """
        IND 后处理流程
        """


        return traj

    async def _process_ood_trajectory(self, traj: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        """
        OOD 后处理流程
        """

        return traj

    def _ood_analysis_flow(self, traj: t.Dict[str, t.Any]):
        """
        OOD 分析流程
        """
        pass  # TODO: OOD 具体分析逻辑

    async def _intent_clustering_writer_task(
            self,
            queue: asyncio.Queue,
            processed_session_ids: set,
            global_cache: t.Dict[str, dict] = None
    ):
        """
        第一阶段专属的意图缓存写入协程。
        负责从队列中获取处理完（LLM提取 + Embedding）的唯一指令数据（支持单条或Batch列表），
        单线程安全地追加写入 global_instruction_cache.jsonl，并同步更新内存缓存字典。
        """
        # 在循环外打开文件，避免每次写入都重复执行 open/close 系统调用，极大提升 I/O 性能
        with open(self.global_instruction_cache_path, "a", encoding="utf-8") as f_out:
            while True:
                item = await queue.get()
                if item is None:
                    queue.task_done()
                    break

                # 兼容处理：允许生产者一次性塞入一个 Batch (list)，或者塞入单条数据 (dict)
                items_to_write = item if isinstance(item, list) else [item]

                try:
                    for single_item in items_to_write:
                        # 优先取 cache_key，兼容 sessionid
                        unique_id = single_item.get("cache_key") or single_item.get("sessionid", "")

                        # 二次去重校验（在单线程消费者中校验，绝对安全）
                        if unique_id and unique_id in processed_session_ids:
                            continue

                        # 写入文件
                        f_out.write(json.dumps(single_item, ensure_ascii=False) + "\n")

                        # 更新去重集合与内存中的全局缓存字典
                        if unique_id:
                            processed_session_ids.add(unique_id)
                            if global_cache is not None:
                                global_cache[unique_id] = single_item

                    # 每处理完队列中的一个元素（如一个 Batch），立即刷入磁盘防止断电丢失
                    f_out.flush()

                except Exception as e:
                    self.log(f"写入意图缓存文件失败: {str(e)}")
                finally:
                    queue.task_done()

    async def _stream_file_writer_task(self, queue: asyncio.Queue, processed_session_ids: set):
        """
        加固版：专属的文件写入协程。
        """
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break

            data_type, data = item
            if not isinstance(data, dict):
                queue.task_done()
                continue

            session_id = data.get("sessionid", "")

            if session_id and session_id in processed_session_ids:
                queue.task_done()
                continue

            try:
                if data_type == "success":
                    path = self.success_trajectories_path
                elif data_type == "failure":
                    path = self.failure_trajectories_path
                elif data_type == "ineffective_recall":
                    path = self.ineffective_plan_recall_trajectories_path
                elif data_type == "rejection_case":
                    path = self.rejection_trajectories_path
                elif data_type == "parse_failed":
                    path = self.parse_failed_trajectories_path
                else:
                    path = None

                if path:
                    # 确保父目录存在
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "a", encoding="utf-8") as f_out:
                        f_out.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
                        f_out.flush()

                    if session_id:
                        processed_session_ids.add(session_id)
            except Exception as e:
                self.log(f"❌ [严重错误] 写入文件失败 (类型: {data_type}, ID: {session_id}): {str(e)}")
                self.log(f"写入文件失败 (ID: {session_id}): {str(e)}")
            finally:
                queue.task_done()

    # 贪心 DPP 多样性采样静态方法 (内聚设计，方便调用)
    @staticmethod
    def _dpp_sample(features_matrix: np.ndarray, k: int) -> t.List[int]:
        """
        增强版贪心 DPP 多样性采样：加入重心归一化、增量投影残差更新与二次正交化
        """
        n_items = features_matrix.shape[0]
        if n_items <= k:
            return list(range(n_items))

        # 1. 特征向量 L2 归一化
        norms = np.linalg.norm(features_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed_features = features_matrix / norms

        selected = []
        basis = []

        # 2. 计算簇重心并进行 L2 归一化，选择最靠近重心的样本作为锚点
        centroid = np.mean(normed_features, axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 1e-9:
            centroid = centroid / c_norm

        first_idx = int(np.argmax(normed_features @ centroid))
        selected.append(first_idx)

        first_vec = normed_features[first_idx].copy()
        basis.append(first_vec / (np.linalg.norm(first_vec) + 1e-9))

        # 3. 增量维护正交残差模长平方（避免每轮重复遍历全部 basis）
        ortho_norms_sq = np.ones(n_items, dtype=np.float32)
        ortho_norms_sq[first_idx] = -1.0

        for _ in range(1, k):
            last_b = basis[-1]
            ortho_norms_sq -= (normed_features @ last_b) ** 2
            ortho_norms_sq[selected] = -1.0

            next_idx = int(np.argmax(ortho_norms_sq))
            if ortho_norms_sq[next_idx] < 1e-5:
                break

            selected.append(next_idx)

            # 4. 二次 Gram-Schmidt 正交化（消除浮点舍入导致的非正交漂移）
            new_vec = normed_features[next_idx].copy()
            for _ in range(2):
                for b in basis:
                    new_vec -= (new_vec @ b) * b

            vec_norm = np.linalg.norm(new_vec)
            if vec_norm < 1e-6:
                break
            basis.append(new_vec / vec_norm)

        return selected

    async def _split_routed_by_intent(self):
        """对 filtered_routed_trajectories.jsonl 按意图进行再分流：
        意图下所有轨迹 completion_status 均为 failed → 该意图下全部数据为 OOD；
        否则（全部成功或部分成功）→ 该意图下全部数据为 IND（失败轨迹也归入 IND）。
        输出 intent_ind_trajectories.jsonl / intent_ood_trajectories.jsonl。
        """
        if not self.filtered_routed_trajectories_path.exists():
            self.log("filtered_routed_trajectories.jsonl 不存在，跳过意图级再分流")
            return

        # 1. 读取全部有效簇记录
        records = []
        with open(self.filtered_routed_trajectories_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # 仅保留聚类有效（is_match=true）的轨迹，与后端 _cluster_effective 对齐
                ce = rec.get("cluster_effectiveness")
                is_match = (ce.get("is_match") is True) if isinstance(ce, dict) else (ce is True)
                if is_match:
                    records.append(rec)

        if not records:
            self.log("filtered_routed_trajectories.jsonl 无有效簇记录，意图级再分流无输出")
            return

        # 2. 按 (category, app, intent) 聚合 verdict 状态
        intent_status = defaultdict(lambda: {"total": 0, "statuses": set()})
        for rec in records:
            cat_obj = rec.get("category")
            cat = cat_obj.get("category", "") if isinstance(cat_obj, dict) else (cat_obj if isinstance(cat_obj, str) else "")
            app = rec.get("app", "") or ""
            intent = rec.get("intent_cluster_name") or rec.get("cluster_name") or rec.get("intent") or "未识别意图"
            key = (cat, app, intent)
            intent_status[key]["total"] += 1
            v = rec.get("verdict") or {}
            status = (v.get("completion_status") or "").strip() or "unknown"
            intent_status[key]["statuses"].add(status)

        # 3. 逐条分流写入（同一意图的所有轨迹进入同一文件）
        with open(self.intent_ind_trajectories_path, "w", encoding="utf-8") as f_ind, \
                open(self.intent_ood_trajectories_path, "w", encoding="utf-8") as f_ood:
            ind_count = ood_count = 0
            for rec in records:
                cat_obj = rec.get("category")
                cat = cat_obj.get("category", "") if isinstance(cat_obj, dict) else (cat_obj if isinstance(cat_obj, str) else "")
                app = rec.get("app", "") or ""
                intent = rec.get("intent_cluster_name") or rec.get("cluster_name") or rec.get("intent") or "未识别意图"
                key = (cat, app, intent)
                is_ood = intent_status[key]["statuses"] == {"failed"}
                target = f_ood if is_ood else f_ind
                target.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if is_ood:
                    ood_count += 1
                else:
                    ind_count += 1

        self.log(f"意图级再分流完成：IND {ind_count} 条 / OOD {ood_count} 条")
        self.log(f"意图级再分流完成：IND {ind_count} 条 / OOD {ood_count} 条")

    def run(self, input_file_path: str | Path | t.List[str|Path]):
        """
        两阶段解耦管线总入口
        """
        if isinstance(input_file_path, str):
            input_file_path = [input_file_path]

        async def _run_all():
            start_time = time.perf_counter()  # Pipeline 总计时开始
            try:
                # =========================================================================
                # 【阶段一.一】构建或加载全局意图缓存
                # =========================================================================
                global_cache = await self._build_or_load_global_intent_cache(input_file_path)
                self._flush_perf_report()  # 💡 模块一跑完，立刻实时落盘一次！

                # =========================================================================
                # 【阶段一.二】全局指令聚类重算与缓存刷新
                # =========================================================================
                global_cache = await self._run_global_clustering_on_cache(global_cache)
                self._flush_perf_report()  # 💡 模块二跑完，立刻实时落盘一次！

                # =========================================================================
                # 【阶段二】逐文件流式读取轨迹并执行 Verdict 判定
                # =========================================================================
                await self._run_trajectory_verdict(input_file_path, global_cache)
                self._flush_perf_report()  # 💡 模块三跑完，立刻实时落盘一次！

                # =========================================================================
                # 【阶段三：后续收尾】意图级再分流逻辑
                # =========================================================================
                t_post_start = time.perf_counter()
                success_trajectories = []
                failure_trajectories = []

                if self.success_trajectories_path.exists():
                    with open(self.success_trajectories_path, "r", encoding="utf-8") as f:
                        success_trajectories = [json.loads(line) for line in f.readlines() if line.strip()]
                if self.failure_trajectories_path.exists():
                    with open(self.failure_trajectories_path, "r", encoding="utf-8") as f:
                        failure_trajectories = [json.loads(line) for line in f.readlines() if line.strip()]

                routed_trajectories = success_trajectories + failure_trajectories
                if routed_trajectories:
                    with open(self.filtered_routed_trajectories_path, "w", encoding="utf-8") as f:
                        for d in routed_trajectories:
                            d.update({
                                "cluster_effectiveness": {
                                    "instruction_intent": "",
                                    "cluster_intent": "",
                                    "reasoning": "",
                                    "is_match": True
                                }
                            })
                            f.write(json.dumps(d, ensure_ascii=False) + "\n")

                    await self._split_routed_by_intent()

                self._set_wall_time("stage3_post_routing_and_split", time.perf_counter() - t_post_start)

                # =========================================================================
                # 【收尾汇总】
                # =========================================================================
                end_time = time.perf_counter()
                total_elapsed = end_time - start_time
                self._set_wall_time("pipeline_total_wall_time", total_elapsed)

                # 最后再完整刷入并打印最终报告
                self._flush_perf_report()

                report_str = json.dumps(self.perf_stats, ensure_ascii=False, indent=2)
                self.log(f"Pipeline 性能与 Token 统计报告:\n{report_str}")
                self.log(f"\n📊 ========== Pipeline 性能与 Token 统计报告 ==========\n{report_str}")
                self.log(f"========== 全量管线处理完成，总耗时: {total_elapsed:.2f} 秒 ==========")

            finally:
                if hasattr(self, "llm") and self.llm:
                    await self.llm.close()
                if hasattr(self, "embedding_client") and self.embedding_client:
                    await self.embedding_client.close()
                await asyncio.sleep(0.25)

        asyncio.run(_run_all())
