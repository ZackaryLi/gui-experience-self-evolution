import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from .config import OODBaselineConfig
from .schemas import (
    BaselinePlan,
    BaselineSample,
    IntentKey,
    LoadedTrajectory,
    content_hash,
)
from .storage import atomic_write_json


class FrozenBaselinePlanError(ValueError):
    pass


def normalize_instruction(instruction: str) -> str:
    return " ".join(instruction.split())


def instruction_id(intent_id: str, instruction: str) -> str:
    return content_hash({
        "intent_id": intent_id,
        "normalized_instruction": normalize_instruction(instruction),
    })[:20]


class BaselinePlanner:
    def __init__(self, config: OODBaselineConfig | None = None):
        self.config = config or OODBaselineConfig()

    def planned_attempt_count(self, records: List[LoadedTrajectory]) -> int:
        unique_instructions = {
            normalize_instruction(record.record.get("instruction", ""))
            for record in records
        }
        return min(
            self.config.instruction_sample_size,
            len(unique_instructions),
        ) * self.config.attempts_per_instruction

    def create_or_load(
        self,
        path: str | Path,
        key: IntentKey,
        records: List[LoadedTrajectory],
    ) -> Tuple[BaselinePlan, bool]:
        target = Path(path)
        if target.exists():
            return self._load_and_validate(target, key, records), True

        plan = self._create(key, records)
        atomic_write_json(target, plan.to_dict())
        return plan, False

    def _create(
        self,
        key: IntentKey,
        records: List[LoadedTrajectory],
    ) -> BaselinePlan:
        if not records:
            raise ValueError("无法为空候选集创建基线计划")

        grouped: Dict[str, List[LoadedTrajectory]] = {}
        for record in records:
            instruction = record.record.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("候选轨迹缺少有效 instruction")
            grouped.setdefault(normalize_instruction(instruction), []).append(record)

        candidates = []
        for normalized, group in grouped.items():
            ordered = sorted(group, key=lambda record: (record.sessionid, record.content_hash))
            first = ordered[0]
            candidates.append({
                "instruction_id": instruction_id(key.intent_id, normalized),
                "instruction": str(first.record["instruction"]).strip(),
                "records": ordered,
            })
        candidates.sort(key=lambda candidate: candidate["instruction_id"])
        random.Random(f"{self.config.random_seed}:{key.intent_id}").shuffle(candidates)
        selected = sorted(
            candidates[:self.config.instruction_sample_size],
            key=lambda candidate: candidate["instruction_id"],
        )

        samples = [
            BaselineSample(
                instruction_id=candidate["instruction_id"],
                instruction=candidate["instruction"],
                source_session_ids=[record.sessionid for record in candidate["records"]],
                source_content_hashes={
                    record.sessionid: record.content_hash
                    for record in candidate["records"]
                },
                attempt_indices=list(range(
                    1,
                    self.config.attempts_per_instruction + 1,
                )),
            )
            for candidate in selected
        ]
        source_fingerprint = self._source_fingerprint(records)
        plan_id = self._plan_id(
            key=key,
            samples=samples,
            source_fingerprint=source_fingerprint,
            config=self.config,
        )
        return BaselinePlan(
            plan_id=plan_id,
            intent_id=key.intent_id,
            intent_key=key,
            status="FROZEN",
            instruction_sample_size=self.config.instruction_sample_size,
            attempts_per_instruction=self.config.attempts_per_instruction,
            random_seed=self.config.random_seed,
            source_candidate_count=len(records),
            source_fingerprint=source_fingerprint,
            samples=samples,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def _load_and_validate(
        self,
        path: Path,
        key: IntentKey,
        records: List[LoadedTrajectory],
    ) -> BaselinePlan:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            samples = [
                BaselineSample(
                    instruction_id=sample["instruction_id"],
                    instruction=sample["instruction"],
                    source_session_ids=list(sample["source_session_ids"]),
                    source_content_hashes=dict(sample["source_content_hashes"]),
                    attempt_indices=list(sample["attempt_indices"]),
                )
                for sample in data["samples"]
            ]
            plan = BaselinePlan(
                plan_id=data["plan_id"],
                intent_id=data["intent_id"],
                intent_key=IntentKey(**data["intent_key"]),
                status=data["status"],
                instruction_sample_size=data["instruction_sample_size"],
                attempts_per_instruction=data["attempts_per_instruction"],
                random_seed=data["random_seed"],
                source_candidate_count=data["source_candidate_count"],
                source_fingerprint=data["source_fingerprint"],
                samples=samples,
                created_at=data["created_at"],
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise FrozenBaselinePlanError(f"冻结基线计划无法解析: {error}") from error

        if plan.intent_id != key.intent_id or plan.intent_key != key:
            raise FrozenBaselinePlanError("冻结基线计划的意图标识与当前意图不一致")
        if plan.status != "FROZEN" or not plan.samples:
            raise FrozenBaselinePlanError("冻结基线计划状态非法或没有抽样指令")
        if plan.instruction_sample_size <= 0 or plan.attempts_per_instruction <= 0:
            raise FrozenBaselinePlanError("冻结基线计划的 k/a 配置非法")

        current_by_session = {record.sessionid: record for record in records}
        seen_instructions = set()
        for sample in plan.samples:
            if sample.instruction_id in seen_instructions:
                raise FrozenBaselinePlanError("冻结基线计划包含重复 instruction_id")
            seen_instructions.add(sample.instruction_id)
            expected_id = instruction_id(key.intent_id, sample.instruction)
            if sample.instruction_id != expected_id:
                raise FrozenBaselinePlanError("冻结基线计划的 instruction_id 校验失败")
            if sample.attempt_indices != list(range(
                1,
                plan.attempts_per_instruction + 1,
            )):
                raise FrozenBaselinePlanError("冻结基线计划的 attempt 序列不完整")
            if set(sample.source_session_ids) != set(sample.source_content_hashes):
                raise FrozenBaselinePlanError("冻结基线计划的来源轨迹引用不一致")
            for sessionid, expected_hash in sample.source_content_hashes.items():
                current = current_by_session.get(sessionid)
                if current is None or current.content_hash != expected_hash:
                    raise FrozenBaselinePlanError(
                        f"冻结计划来源轨迹已变化或缺失: {sessionid}"
                    )

        expected_plan_id = self._plan_id(
            key=key,
            samples=plan.samples,
            source_fingerprint=plan.source_fingerprint,
            config=OODBaselineConfig(
                instruction_sample_size=plan.instruction_sample_size,
                attempts_per_instruction=plan.attempts_per_instruction,
                random_seed=plan.random_seed,
            ),
        )
        if plan.plan_id != expected_plan_id:
            raise FrozenBaselinePlanError("冻结基线计划的 plan_id 校验失败")
        return plan

    @staticmethod
    def _source_fingerprint(records: List[LoadedTrajectory]) -> str:
        return content_hash([
            {
                "sessionid": record.sessionid,
                "content_hash": record.content_hash,
            }
            for record in sorted(records, key=lambda item: item.sessionid)
        ])

    @staticmethod
    def _plan_id(
        key: IntentKey,
        samples: List[BaselineSample],
        source_fingerprint: str,
        config: OODBaselineConfig,
    ) -> str:
        return content_hash({
            "intent_id": key.intent_id,
            "source_fingerprint": source_fingerprint,
            "config": config.to_dict(),
            "samples": [sample.to_dict() for sample in samples],
        })[:24]
