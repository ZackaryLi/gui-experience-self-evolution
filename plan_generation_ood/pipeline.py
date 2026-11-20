from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .config import OODTriageConfig
from .loader import OODLoader
from .schemas import SCHEMA_VERSION, TriageStatus
from .storage import atomic_write_json, atomic_write_jsonl
from .triage import DeterministicTriage


class OODTriagePipeline:
    def __init__(
        self,
        output_dir: str | Path,
        config: OODTriageConfig | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.config = config or OODTriageConfig()
        self.loader = OODLoader()
        self.triage = DeterministicTriage(self.config)

    def run(self, input_file: str | Path) -> Dict[str, Any]:
        outcome = self.loader.load(input_file)
        intent_summaries = []
        status_counts = Counter()
        valid_record_count = 0

        for item in outcome.items:
            result = self.triage.evaluate(item)
            status_counts[result.status.value] += 1
            valid_record_count += len(item.records)

            intent_dir = self.output_dir / "intents" / result.intent_id
            atomic_write_jsonl(
                intent_dir / "input.jsonl",
                [record.record for record in item.records],
            )
            atomic_write_json(intent_dir / "triage.json", result.to_dict())

            candidates_path = intent_dir / "exploration_candidates.jsonl"
            if result.status is TriageStatus.EXPLORABLE:
                atomic_write_jsonl(
                    candidates_path,
                    [record.record for record in result.candidate_records],
                )
            elif candidates_path.exists():
                candidates_path.unlink()

            intent_summaries.append({
                "intent_id": result.intent_id,
                "intent_key": result.key.to_dict(),
                "status": result.status.value,
                "rule_id": result.rule_id,
                "total_records": result.total_records,
                "exploration_candidate_count": len(result.candidate_records),
            })

        atomic_write_jsonl(
            self.output_dir / "inconsistent_records.jsonl",
            [record.to_dict() for record in outcome.inconsistent_records],
        )

        summary = {
            "schema_version": SCHEMA_VERSION,
            "input_file": str(input_file),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_input_records": outcome.total_input_records,
            "valid_records_in_intents": valid_record_count,
            "skipped_cluster_mismatch": outcome.skipped_cluster_mismatch,
            "duplicate_records": outcome.duplicate_records,
            "inconsistent_record_count": len(outcome.inconsistent_records),
            "intent_count": len(outcome.items),
            "status_counts": dict(sorted(status_counts.items())),
            "intents": intent_summaries,
        }
        atomic_write_json(self.output_dir / "summary.json", summary)
        return summary
