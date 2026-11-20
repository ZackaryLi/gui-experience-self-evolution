from dataclasses import dataclass
from typing import Any, Dict


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
