"""LocalDeviceRunner 调试 CLI（doc §10，仅供真机 smoke，不作为正式契约）。

在包含 ``scripts/`` 的部署环境中运行，按 ``OOD_RUNNER_INTERFACE.md`` §9.2
的清单逐项手工验证：

.. code-block:: bash

    python -m plan_generation_ood.runner_cli \\
        --instruction "帮我打开设置" \\
        --app "设置" \\
        --plan "" \\
        --timeout 600

内部走同一个 ``LocalDeviceRunner.run_attempt``，不另写执行逻辑。
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .local_runner import LocalDeviceRunner, ScriptDeviceTaskApi
from .runner import RUNNER_SCHEMA_VERSION, ExperienceMode, RunRequest

DEFAULT_RESULT_ROOT = "runner_smoke_results"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m plan_generation_ood.runner_cli",
        description="单条指令的真机 smoke 调试入口（无 plan 与有 plan 各跑一次即可）",
    )
    parser.add_argument("--instruction", required=True, help="实际下发的用户指令")
    parser.add_argument("--app", required=True, help="目标 App 名称")
    parser.add_argument(
        "--plan",
        default="",
        help="注入的经验文本；默认为空串（无经验基线）",
    )
    parser.add_argument(
        "--plan-file",
        default="",
        help="从文件读取经验文本；与 --plan 互斥，优先级高于 --plan",
    )
    parser.add_argument("--timeout", type=int, default=600, help="单次 attempt 硬超时（秒）")
    parser.add_argument(
        "--result-root",
        default=DEFAULT_RESULT_ROOT,
        help=f"产物根目录（默认 {DEFAULT_RESULT_ROOT}/）",
    )
    parser.add_argument(
        "--key",
        default="",
        help="自定义幂等键；默认按指令和时间戳生成，重跑会命中缓存",
    )
    parser.add_argument(
        "--task-mode",
        default="baseline",
        choices=["baseline", "accepted"],
        help="经验注入模式；携带 plan 时应使用 accepted（默认 baseline）",
    )
    return parser


def build_request(args: argparse.Namespace) -> RunRequest:
    plan_text = args.plan
    if args.plan_file:
        plan_text = Path(args.plan_file).read_text(encoding="utf-8")
    plan_text = plan_text.strip()
    if plan_text and args.task_mode == "baseline":
        raise SystemExit("携带 plan 时应使用 --task-mode accepted，而不是 baseline")
    mode = ExperienceMode.ACCEPTED if plan_text else ExperienceMode.BASELINE
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    key = args.key or f"cli-smoke-{timestamp}"
    return RunRequest(
        schema_version=RUNNER_SCHEMA_VERSION,
        idempotency_key=key,
        run_id=f"cli-run-{timestamp}",
        intent_id="cli-debug",
        experiment_id="cli-debug",
        instruction=args.instruction,
        app=args.app,
        attempt_index=1,
        experience_mode=mode,
        plan_text=plan_text,
        timeout_sec=args.timeout,
        trajectory_context={
            "category": {"category": "调试"},
            "app": args.app,
            "intent_cluster_name": "runner_cli_smoke",
        },
        metadata={"source": "runner_cli"},
    )


async def _run(request: RunRequest, result_root: str) -> Optional[dict]:
    runner = LocalDeviceRunner(
        result_root=result_root,
        task_api=ScriptDeviceTaskApi(),
        max_concurrency=1,
        default_timeout_sec=request.timeout_sec,
    )
    try:
        health = await runner.health_check()
        print(f"[health_check] ok={health.ok} devices={health.devices} "
              f"agent_endpoint_ok={health.agent_endpoint_ok} message={health.message}")
        if not health.ok:
            return None
        result = await runner.run_attempt(request)
        return result.to_dict()
    finally:
        await runner.close()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    request = build_request(args)
    result = asyncio.run(_run(request, args.result_root))
    if result is None:
        print(
            "[health_check] 未通过，已终止本次下发；请确认真机已连接、hdc 可用且部署环境包含 scripts/",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    execution_status = result.get("execution_status")
    print(
        f"\nexecution_status={execution_status} "
        f"infrastructure_status={result.get('infrastructure_status')} "
        f"raw_ref={result.get('raw_ref')}",
        file=sys.stderr,
    )
    return 0 if execution_status == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
