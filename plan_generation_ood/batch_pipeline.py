import asyncio
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .baseline import BaselinePlanner, FrozenBaselinePlanError
from .batch_value import BatchRunValueAssessor
from .config import OODBaselineConfig, OODBatchValueConfig
from .llm import LLMAgent
from .schemas import (
    BatchRunConstraints,
    BatchRunDecision,
    IntentKey,
    LoadedTrajectory,
    SCHEMA_VERSION,
    TriageStatus,
    content_hash,
)
from .storage import atomic_write_json, atomic_write_jsonl


class OODBatchValuePipeline:
    def __init__(
        self,
        triage_dir: str | Path,
        llm_agent: LLMAgent,
        config: OODBatchValueConfig | None = None,
        baseline_config: OODBaselineConfig | None = None,
        close_llm: bool = True,
    ):
        self.triage_dir = Path(triage_dir)
        self.llm = llm_agent
        self.config = config or OODBatchValueConfig()
        self.baseline_config = baseline_config or OODBaselineConfig()
        self.assessor = BatchRunValueAssessor(
            llm=llm_agent,
            config=self.config,
            baseline_config=self.baseline_config,
        )
        self.planner = BaselinePlanner(self.baseline_config)
        self.close_llm = close_llm

    def run(
        self,
        constraints_by_intent: Optional[Dict[str, BatchRunConstraints]] = None,
    ) -> Dict[str, object]:
        return asyncio.run(self.arun(constraints_by_intent))

    async def arun(
        self,
        constraints_by_intent: Optional[Dict[str, BatchRunConstraints]] = None,
    ) -> Dict[str, object]:
        constraints_by_intent = constraints_by_intent or {}
        try:
            return await self._run(constraints_by_intent)
        finally:
            if self.close_llm:
                await self.llm.close()

    async def _run(
        self,
        constraints_by_intent: Dict[str, BatchRunConstraints],
    ) -> Dict[str, object]:
        intents_root = self.triage_dir / "intents"
        if not intents_root.is_dir():
            raise FileNotFoundError(f"分诊意图目录不存在: {intents_root}")

        intent_summaries = []
        decision_counts = Counter()
        skipped_non_explorable = 0
        frozen_plan_count = 0
        reused_plan_count = 0

        for intent_dir in sorted(path for path in intents_root.iterdir() if path.is_dir()):
            triage_path = intent_dir / "triage.json"
            if not triage_path.is_file():
                continue
            triage_data = json.loads(triage_path.read_text(encoding="utf-8"))
            if triage_data.get("status") != TriageStatus.EXPLORABLE.value:
                skipped_non_explorable += 1
                continue

            key = IntentKey(**triage_data["intent_key"])
            if triage_data.get("intent_id") != key.intent_id:
                raise ValueError(f"分诊产物的 intent_id 不一致: {triage_path}")
            records = self._load_candidates(intent_dir / "exploration_candidates.jsonl")
            constraints = constraints_by_intent.get(
                key.intent_id,
                BatchRunConstraints(max_attempts=self.config.max_total_attempts),
            )
            assessment = await self.assessor.assess(key, records, constraints)

            plan = None
            plan_reused = False
            if assessment.decision is BatchRunDecision.RUN_BASELINE:
                try:
                    plan, plan_reused = self.planner.create_or_load(
                        intent_dir / "baseline_plan.json",
                        key,
                        records,
                    )
                    atomic_write_jsonl(
                        intent_dir / "baseline_attempts.jsonl",
                        [request.to_dict() for request in plan.to_attempt_requests()],
                    )
                    if plan_reused:
                        reused_plan_count += 1
                    else:
                        frozen_plan_count += 1
                except FrozenBaselinePlanError as error:
                    assessment.llm_decision = assessment.decision
                    assessment.decision = BatchRunDecision.MANUAL_REVIEW
                    assessment.rule_findings.append("FROZEN_BASELINE_PLAN_CONFLICT")
                    assessment.reason = (
                        f"{assessment.reason}；冻结计划校验失败，禁止重新抽样: {error}"
                    )

            atomic_write_json(intent_dir / "batch_value.json", assessment.to_dict())
            decision_counts[assessment.decision.value] += 1
            intent_summaries.append({
                "intent_id": key.intent_id,
                "intent_key": key.to_dict(),
                "decision": assessment.decision.value,
                "llm_decision": (
                    assessment.llm_decision.value
                    if assessment.llm_decision else None
                ),
                "candidate_count": assessment.candidate_count,
                "unique_instruction_count": assessment.unique_instruction_count,
                "planned_attempt_count": assessment.planned_attempt_count,
                "rule_findings": assessment.rule_findings,
                "baseline_plan_id": plan.plan_id if plan else None,
                "baseline_plan_reused": plan_reused,
                "total_attempts": plan.total_attempts if plan else 0,
            })

        summary = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "triage_dir": str(self.triage_dir),
            "evaluated_intent_count": len(intent_summaries),
            "skipped_non_explorable": skipped_non_explorable,
            "decision_counts": dict(sorted(decision_counts.items())),
            "frozen_plan_count": frozen_plan_count,
            "reused_plan_count": reused_plan_count,
            "baseline_config": self.baseline_config.to_dict(),
            "value_config": self.config.to_dict(),
            "runner_status": "PROTOCOL_ONLY",
            "intents": intent_summaries,
        }
        atomic_write_json(self.triage_dir / "batch_value_summary.json", summary)
        return summary

    @staticmethod
    def _load_candidates(path: Path) -> List[LoadedTrajectory]:
        if not path.is_file():
            raise FileNotFoundError(f"EXPLORABLE 意图缺少候选文件: {path}")
        records = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            raw_record = json.loads(line)
            if not isinstance(raw_record, dict):
                raise ValueError(f"候选轨迹必须是 JSON 对象: {path}:{line_number}")
            records.append(LoadedTrajectory(
                record=raw_record,
                content_hash=content_hash(raw_record),
                line_number=line_number,
            ))
        return records
