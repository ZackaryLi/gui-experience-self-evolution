# -*- coding: utf-8 -*-
"""意图聚类质量基准离线工具（INTENT_CLUSTERING_IMPROVEMENT_PLAN.md 第五章）。

不参与线上 pipeline，手动触发的离线测量工具，三个子命令：

  purity      纯度基准：对 global_intent_cache.jsonl 的每个簇采样成员对，
              调 LLM 判"是否同一任务"，输出簇内纯度与全局混簇率基线数字。

  calibration 灰度带校准：对难例对基准集（人工标注的 JSONL）逐对算
              拼接向量/target 向量/action 向量的余弦相似度，输出
              "同任务对/不同任务对"的分布与重叠区（= 灰度带），及
              T_target / T_action 建议初值。

  pairs       难例对清单生成：从缓存里挑相似度落在边界带的指令对，
              产出待人工标注的候选清单（标注完成后作为 calibration 输入）。

环境前提（与 pipeline 相同）：当前工作目录需有 config.json
（含 chat_api_key / embedding_api_key）。

用法示例：
  python clustering_purity_bench.py purity  --cache global_intent_cache.jsonl \
      --max-pairs-per-cluster 5 --out purity_report.json
  python clustering_purity_bench.py calibration --pairs labeled_pairs.jsonl \
      --out calibration_report.json
  python clustering_purity_bench.py pairs --cache global_intent_cache.jsonl \
      --band 0.70,0.90 --limit 60 --out pair_candidates.jsonl
"""

import argparse
import asyncio
import json
import random
import sys
import time
import typing as t
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

CONFIG_PATH = "config.json"
EMBEDDING_MODEL = "bge-m3"
EMBEDDING_CONCURRENCY = 10

# 纯度判定的 LLM prompt：与方案 3.4 同口径——"能否用同一条固定执行方案"
PURITY_JUDGE_PROMPT = """你是手机 GUI 任务分析专家。判断以下两条用户指令是否在执行**同一个任务**。

判定标准：这两条指令能否用**同一条固定的执行步骤方案**完成？
- 如果执行时需要进入不同的页面、点击不同的入口、走不同的操作路径，就是不同任务；
- 仅仅是表述、平台、参数（如商品名、联系人名）不同，但操作路径完全一致，是同一个任务。

指令A：{instruction_a}
指令B：{instruction_b}

你必须只输出合法 JSON（不要 markdown 标记、不要解释）：
{{"same_task": true 或 false, "reason": "判定理由，限30字"}}"""


# ---------------------------------------------------------------------------
# 基础 I/O 与 LLM/Embedding 客户端（自包含，不 import pipeline——
# pipeline 在 import 时读 config 且依赖 aiohttp，离线工具单独走 openai SDK）
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> t.List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def load_config() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[错误] 未找到 {CONFIG_PATH}：离线基准工具需要与 pipeline 相同的"
              f"config.json（含 chat_api_key / embedding_api_key），请在含该文件的目录运行。",
              file=sys.stderr)
        raise SystemExit(1)


def build_clients(config: dict):
    """构建 LLM（chat 兼容端点）与 Embedding 客户端。

    config.json 的 chat 端点与 pipeline 的 LLMAgent 保持一致：
      chat_base_url + "/chat/completions"、chat_model
    embedding 走 pipeline 同款 OpenAI 兼容 base_url。
    """
    from openai import AsyncOpenAI

    chat_base = config.get("chat_base_url",
                            "http://mlops.huawei.com/mlops-service/api/v2/agentService/v1")
    embedding_base = config.get("embedding_base_url", chat_base)
    model = config.get("chat_model", "qwen38-27b-vl")

    llm = AsyncOpenAI(api_key=config["chat_api_key"], base_url=f"{chat_base}/chat/completions")
    # OpenAI SDK 的 base_url 需指向不含 /chat/completions 的根
    embedding = AsyncOpenAI(api_key=config["embedding_api_key"], base_url=embedding_base)
    return llm, embedding, model


async def judge_same_task(llm, model: str, instruction_a: str, instruction_b: str,
                          retries: int = 3) -> t.Optional[dict]:
    """调 LLM 判两条指令是否同一任务，返回 {"same_task": bool, "reason": str}。"""
    prompt = PURITY_JUDGE_PROMPT.format(instruction_a=instruction_a, instruction_b=instruction_b)
    for attempt in range(retries):
        try:
            resp = await llm.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = resp.choices[0].message.content or ""
            return _parse_judge_json(content)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if attempt == retries - 1:
                print(f"[LLM 判定失败] {e}", file=sys.stderr)
                return None
            await asyncio.sleep(2 ** attempt)
    return None


def _parse_judge_json(content: str) -> t.Optional[dict]:
    """宽容解析判定 JSON（复用 json_repair 与结构切片，失败返回 None）。"""
    try:
        from json_repair import repair_json
        repaired = repair_json(content, return_objects=True)
        if isinstance(repaired, dict) and "same_task" in repaired:
            return {"same_task": bool(repaired["same_task"]), "reason": str(repaired.get("reason", ""))}
    except Exception:
        pass
    import re
    match = re.search(r'\{[^{}]*"same_task"[^{}]*\}', content, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            return {"same_task": bool(obj["same_task"]), "reason": str(obj.get("reason", ""))}
        except Exception:
            pass
    return None


async def embed_texts(client, texts: t.List[str], sem: asyncio.Semaphore,
                       model: str = EMBEDDING_MODEL) -> t.List[t.List[float]]:
    """批量 Embedding（并发信号量 + 指数退避），与 pipeline._intent_embedding 同口径。"""
    safe_texts = [x if x and x.strip() else " " for x in texts]
    max_retries = 3
    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.embeddings.create(model=model, input=safe_texts)
                sorted_data = sorted(resp.data, key=lambda x: x.index)
                return [d.embedding for d in sorted_data]
            except asyncio.CancelledError:
                raise
            except Exception:
                if attempt == max_retries - 1:
                    return [[] for _ in texts]
                await asyncio.sleep(2 ** attempt)
    return [[] for _ in texts]


def cosine_sim(a: t.List[float], b: t.List[float]) -> t.Optional[float]:
    if not a or not b or len(a) != len(b):
        return None
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return None
    return float(np.dot(va, vb) / (na * nb))


def build_composite_text(item: dict) -> str:
    """拼接向量文本（方案 3.2a）：core_action + target + attributes。"""
    action = str(item.get("core_action", "") or "").strip()
    target = str(item.get("target", "") or "").strip()
    attrs = item.get("attributes")
    attr_str = " ".join(str(a).strip() for a in attrs if str(a).strip()) if isinstance(attrs, list) else ""
    # 旧缓存条目缺结构化字段时回退 core_intent（方案 3.2 旧缓存兼容规则）
    if not (action or target):
        return str(item.get("intent", "") or "").strip()
    return f"{action} {target} {attr_str}".strip()


# ---------------------------------------------------------------------------
# 子命令 1：纯度基准 (purity)
# ---------------------------------------------------------------------------

async def cmd_purity(args) -> None:
    from tqdm import tqdm

    config = load_config()
    llm, _, model = build_clients(config)

    items = load_jsonl(args.cache)
    # 仅统计聚过类且向量可用的条目（与线上分桶口径一致：排除 -1 噪声）
    clustered = [it for it in items
                 if isinstance(it.get("cluster_label"), int) and it["cluster_label"] != -1
                 and it.get("instruction")]
    if not clustered:
        print("[错误] 缓存中没有可测的有效聚簇条目（cluster_label != -1 且有 instruction）",
              file=sys.stderr)
        raise SystemExit(1)

    # 按 (场景, app, 簇) 分组
    groups: t.Dict[tuple, t.List[dict]] = defaultdict(list)
    for it in clustered:
        cat = it.get("category", {})
        scenario = cat.get("category", "unknown") if isinstance(cat, dict) else "unknown"
        app = str(it.get("app", "") or "unknown")
        groups[(scenario, app, int(it["cluster_label"]))].append(it)

    rng = random.Random(args.seed)

    # 生成抽检任务：簇内采样最多 max_pairs 对成员（指令文本对）
    tasks: t.List[t.Tuple[tuple, str, str]] = []
    for key, members in groups.items():
        if len(members) < 2:
            continue
        texts = list(dict.fromkeys(m["instruction"] for m in members))
        n_pairs = min(args.max_pairs_per_cluster, len(texts) * (len(texts) - 1) // 2)
        for _ in range(n_pairs):
            a, b = rng.sample(texts, 2)
            tasks.append((key, a, b))

    print(f"共 {len(groups)} 个簇，生成 {len(tasks)} 个抽检对，开始 LLM 判定...")

    results: t.List[dict] = []
    judged: t.List[t.Optional[dict]] = [None] * len(tasks)

    async def _judge_at(idx: int):
        _, a, b = tasks[idx]
        judged[idx] = await judge_same_task(llm, model, a, b)
        pbar.update(1)

    # 按索引回写保证任务-结果映射不错位（as_completed 是完成序，不能直接 zip）
    with tqdm(total=len(tasks), desc="纯度抽检", unit="pair") as pbar:
        await asyncio.gather(*[_judge_at(i) for i in range(len(tasks))])

    for (key, a, b), res in zip(tasks, judged):
        results.append({"cluster": list(key), "instruction_a": a, "instruction_b": b,
                        "judgement": res})

    # 统计：簇纯度 = 簇内抽检对中 same_task 比例；无判定结果的对不计入分母
    per_cluster = defaultdict(lambda: {"total": 0, "same": 0, "unjudged": 0})
    for r in results:
        stat = per_cluster[tuple(r["cluster"])]
        if r["judgement"] is None:
            stat["unjudged"] += 1
            continue
        stat["total"] += 1
        if r["judgement"]["same_task"]:
            stat["same"] += 1

    purity_values = [s["same"] / s["total"] for s in per_cluster.values() if s["total"] > 0]
    mixed_clusters = sum(1 for s in per_cluster.values() if s["total"] > 0 and s["same"] < s["total"])
    overall = {
        "cache_file": str(args.cache),
        "total_clusters": len(groups),
        "sampled_clusters": len(per_cluster),
        "sampled_pairs": len(tasks),
        "unjudged_pairs": sum(s["unjudged"] for s in per_cluster.values()),
        "avg_purity": round(float(np.mean(purity_values)), 4) if purity_values else None,
        "median_purity": round(float(np.median(purity_values)), 4) if purity_values else None,
        "mixed_cluster_count": mixed_clusters,
        "mixed_cluster_ratio": round(mixed_clusters / len(per_cluster), 4) if per_cluster else None,
    }

    report = {
        "overall": overall,
        "per_cluster": [
            {
                "cluster": list(k),
                "purity": round(s["same"] / s["total"], 4) if s["total"] else None,
                "sampled": s["total"],
                "unjudged": s["unjudged"],
            }
            for k, s in sorted(per_cluster.items())
        ],
        "raw_judgements": results,
    }

    _write_output(args.out, report)
    print(json.dumps(overall, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# 子命令 2：灰度带校准 (calibration)
# ---------------------------------------------------------------------------

async def cmd_calibration(args) -> None:
    config = load_config()
    _, embedding, _ = build_clients(config)
    sem = asyncio.Semaphore(EMBEDDING_CONCURRENCY)

    pairs = load_jsonl(args.pairs)
    # 校验标注格式：{instruction_a, instruction_b, same_task}
    valid = [p for p in pairs
             if p.get("instruction_a") and p.get("instruction_b")
             and isinstance(p.get("same_task"), bool)]
    print(f"载入难例对 {len(valid)}/{len(pairs)} 条（需含 instruction_a/instruction_b/same_task）")

    # 三种文本一次批量编码：拼接文本（用两指令各自的结构化字段不可得时回退原文）
    composite_texts, target_texts, action_texts = [], [], []
    for p in valid:
        # 难例对是原始指令（无结构化字段），拼接向量退化为指令本身——
        # 校准报告需注明：正式校准应在提取后再跑一次（用提取后的结构化字段）
        composite_texts.append(p["instruction_a"])
        composite_texts.append(p["instruction_b"])
        target_texts.append(p.get("target_a", p["instruction_a"]))
        target_texts.append(p.get("target_b", p["instruction_b"]))
        action_texts.append(p.get("core_action_a", ""))
        action_texts.append(p.get("core_action_b", ""))

    composite_embs = await embed_texts(embedding, composite_texts, sem)
    target_embs = await embed_texts(embedding, target_texts, sem)
    action_embs = await embed_texts(embedding, action_texts, sem)

    rows = []
    for i, p in enumerate(valid):
        sim_composite = cosine_sim(composite_embs[2 * i], composite_embs[2 * i + 1])
        sim_target = cosine_sim(target_embs[2 * i], target_embs[2 * i + 1])
        sim_action = cosine_sim(action_embs[2 * i], action_embs[2 * i + 1])
        rows.append({
            "same_task": p["same_task"],
            "sim_composite": round(sim_composite, 4) if sim_composite is not None else None,
            "sim_target": round(sim_target, 4) if sim_target is not None else None,
            "sim_action": round(sim_action, 4) if sim_action is not None else None,
        })

    report = _calibrate_band(rows, "sim_composite")
    report["target_band"] = _calibrate_band(rows, "sim_target")
    report["action_band"] = _calibrate_band(rows, "sim_action")
    report["pairs_detail"] = rows
    report["note"] = ("难例对若为原始指令（无提取后结构化字段），sim_composite 实为指令原文相似度，"
                      "正式灰度带应在提取流程产出结构化字段后重跑校准")

    _write_output(args.out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "pairs_detail"},
                     ensure_ascii=False, indent=2))


def _calibrate_band(rows: t.List[dict], key: str) -> dict:
    """按 same_task 分组算分布，输出重叠区（灰度带）与阈值建议。

    灰度带定义：不同任务对的相似度 95 分位（下界附近）到同任务对的 5 分位
    （上界附近）之间的重叠区间；若分布无重叠则给出分离点。
    """
    sims_same = sorted(r[key] for r in rows if r["same_task"] and r.get(key) is not None)
    sims_diff = sorted(r[key] for r in rows if not r["same_task"] and r.get(key) is not None)
    if not sims_same or not sims_diff:
        return {"error": "标注对的某侧样本不足（需同/不同任务两侧都有）"}

    q = lambda arr, p: float(np.percentile(arr, p))
    band = {
        "same_task": {"n": len(sims_same), "p5": round(q(sims_same, 5), 4),
                      "median": round(q(sims_same, 50), 4), "p95": round(q(sims_same, 95), 4)},
        "diff_task": {"n": len(sims_diff), "p5": round(q(sims_diff, 5), 4),
                      "median": round(q(sims_diff, 50), 4), "p95": round(q(sims_diff, 95), 4)},
    }
    s_low_candidate = round(q(sims_diff, 95), 4)   # 不同任务对上沿：超过它的"远对"基本可免拆
    s_high_candidate = round(q(sims_same, 5), 4)   # 同任务对下沿：低于它的"近对"才免检合并
    band["s_low_candidate"] = s_low_candidate
    band["s_high_candidate"] = s_high_candidate
    band["overlap"] = s_low_candidate < s_high_candidate
    band["advice"] = (
        f"灰度带初始建议 ({s_low_candidate}, {s_high_candidate})；"
        f"按'宁高勿低'原则 S_high 可再上浮（错合代价更高），S_low 附近差异对稀疏时可下调"
        if s_low_candidate < s_high_candidate
        else f"分布分离良好，灰度带可取 ({s_low_candidate}, {s_high_candidate}) 区间，"
             f"甚至可能无需门控——先验证样本量是否足够"
    )
    return band


# ---------------------------------------------------------------------------
# 子命令 3：难例对候选清单生成 (pairs)
# ---------------------------------------------------------------------------

async def cmd_pairs(args) -> None:
    config = load_config()
    _, embedding, _ = build_clients(config)
    sem = asyncio.Semaphore(EMBEDDING_CONCURRENCY)

    items = load_jsonl(args.cache)
    usable = [it for it in items if it.get("instruction") and it.get("intent_embedding")]
    if not usable:
        print("[错误] 缓存条目缺少 intent_embedding（需先跑过线上阶段一）", file=sys.stderr)
        raise SystemExit(1)

    # 按 (场景, app) 分桶，桶内找跨簇相似度落在边界带的指令对
    band_low, band_high = (float(x) for x in args.band.split(","))
    groups: t.Dict[tuple, t.List[dict]] = defaultdict(list)
    for it in usable:
        cat = it.get("category", {})
        scenario = cat.get("category", "unknown") if isinstance(cat, dict) else "unknown"
        groups[(scenario, str(it.get("app", "") or "unknown"))].append(it)

    candidates = []
    rng = random.Random(args.seed)
    for key, members in groups.items():
        # 预算内桶大则先抽样，控制 O(n²) 对比规模
        if len(members) > args.bucket_sample:
            members = rng.sample(members, args.bucket_sample)
        vecs = [np.asarray(m["intent_embedding"], dtype=np.float32) for m in members]
        norms = [np.linalg.norm(v) if v.size else 0 for v in vecs]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if norms[i] == 0 or norms[j] == 0:
                    continue
                sim = float(np.dot(vecs[i], vecs[j]) / (norms[i] * norms[j]))
                if band_low <= sim <= band_high:
                    candidates.append({
                        "scenario": key[0], "app": key[1],
                        "sim": round(sim, 4),
                        "instruction_a": members[i]["instruction"],
                        "cluster_a": members[i].get("cluster_label"),
                        "instruction_b": members[j]["instruction"],
                        "cluster_b": members[j].get("cluster_label"),
                        "same_task": None,  # 待人工标注
                    })

    candidates.sort(key=lambda c: c["sim"], reverse=True)
    if args.limit:
        candidates = candidates[: args.limit]

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"生成 {len(candidates)} 个待标注候选对（边界带 [{band_low}, {band_high}]），"
          f"已写入 {out_path}")
    print("下一步：人工标注 same_task (true/false) 后，运行 calibration 子命令校准灰度带")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _write_output(out: str, report: dict) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"报告已写入 {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="意图聚类质量基准离线工具（纯度基准 / 灰度带校准 / 难例对生成）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_purity = sub.add_parser("purity", help="簇内纯度抽查：LLM 判成员对是否同一任务")
    p_purity.add_argument("--cache", required=True, help="global_intent_cache.jsonl 路径")
    p_purity.add_argument("--max-pairs-per-cluster", type=int, default=5)
    p_purity.add_argument("--seed", type=int, default=42)
    p_purity.add_argument("--out", default="purity_report.json")

    p_calib = sub.add_parser("calibration", help="灰度带校准：难例对相似度分布 → 阈值建议")
    p_calib.add_argument("--pairs", required=True, help="人工标注后的难例对 JSONL")
    p_calib.add_argument("--out", default="calibration_report.json")

    p_pairs = sub.add_parser("pairs", help="从缓存生成边界带待标注指令对清单")
    p_pairs.add_argument("--cache", required=True, help="global_intent_cache.jsonl 路径")
    p_pairs.add_argument("--band", default="0.70,0.90",
                         help="边界带，逗号分隔，默认 0.70,0.90")
    p_pairs.add_argument("--bucket-sample", type=int, default=300,
                         help="桶内超过该规模时先抽样再两两对比")
    p_pairs.add_argument("--limit", type=int, default=60)
    p_pairs.add_argument("--seed", type=int, default=42)
    p_pairs.add_argument("--out", default="pair_candidates.jsonl")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    if args.command == "purity":
        asyncio.run(cmd_purity(args))
    elif args.command == "calibration":
        asyncio.run(cmd_calibration(args))
    elif args.command == "pairs":
        asyncio.run(cmd_pairs(args))
    print(f"耗时 {time.perf_counter() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
