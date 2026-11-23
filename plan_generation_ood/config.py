from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class OODTriageConfig:
    b_dominant_ratio: float = 0.6
    d_dominant_ratio: float = 0.5
    instruction_issue_ratio: float = 0.5
    majority_ratio: float = 0.5

    def __post_init__(self):
        for name in (
            "b_dominant_ratio",
            "d_dominant_ratio",
            "instruction_issue_ratio",
            "majority_ratio",
        ):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} 必须在 0 和 1 之间，实际为 {value}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "b_dominant_ratio": self.b_dominant_ratio,
            "d_dominant_ratio": self.d_dominant_ratio,
            "d_comparison": ">",
            "instruction_issue_ratio": self.instruction_issue_ratio,
            "majority_ratio": self.majority_ratio,
        }


@dataclass(frozen=True)
class OODBaselineConfig:
    instruction_sample_size: int = 3
    attempts_per_instruction: int = 2
    random_seed: int = 0

    def __post_init__(self):
        for name in ("instruction_sample_size", "attempts_per_instruction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed 必须是整数")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction_sample_size": self.instruction_sample_size,
            "attempts_per_instruction": self.attempts_per_instruction,
            "random_seed": self.random_seed,
        }


@dataclass(frozen=True)
class OODBatchValueConfig:
    min_llm_confidence: float = 0.6
    max_llm_attempts: int = 2
    max_total_attempts: Optional[int] = None

    def __post_init__(self):
        if (
            isinstance(self.min_llm_confidence, bool)
            or not isinstance(self.min_llm_confidence, (int, float))
            or not 0.0 <= self.min_llm_confidence <= 1.0
        ):
            raise ValueError("min_llm_confidence 必须位于 0 和 1 之间")
        for name in ("max_llm_attempts", "max_total_attempts"):
            value = getattr(self, name)
            if value is None and name == "max_total_attempts":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数" + ("或 None" if name == "max_total_attempts" else ""))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_llm_confidence": self.min_llm_confidence,
            "max_llm_attempts": self.max_llm_attempts,
            "max_total_attempts": self.max_total_attempts,
        }


@dataclass(frozen=True)
class OODRecoveryAgentConfig:
    max_tool_rounds: int = 4
    max_invalid_responses: int = 2

    def __post_init__(self):
        for name in ("max_tool_rounds", "max_invalid_responses"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} 必须是整数")
        if self.max_tool_rounds < 0:
            raise ValueError("max_tool_rounds 不能小于 0")
        if self.max_invalid_responses <= 0:
            raise ValueError("max_invalid_responses 必须大于 0")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_tool_rounds": self.max_tool_rounds,
            "max_invalid_responses": self.max_invalid_responses,
        }
