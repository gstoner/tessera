"""Scoring API for scorable tasks."""
from dataclasses import dataclass
from typing import Callable, Dict, Any

@dataclass
class ScoreResult:
    value: float
    metric: str
    metadata: Dict[str, Any]

class ScorableTask:
    def __init__(self, name: str, evaluator: Callable[[str], ScoreResult]):
        self.name = name
        self.evaluator = evaluator  # receives path to candidate workspace

    def score(self, workspace: str) -> ScoreResult:
        return self.evaluator(workspace)
