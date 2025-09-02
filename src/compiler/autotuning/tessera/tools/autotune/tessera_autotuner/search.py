
from typing import Iterable, Dict, Any
from .spaces import SearchSpace

class GridSearch:
    def __init__(self, space: SearchSpace):
        self.space = space
    def candidates(self) -> Iterable[Dict[str, Any]]:
        return self.space.grid()

class RandomSearch:
    def __init__(self, space: SearchSpace, max_trials: int, seed: int = 17):
        self.space = space; self.max_trials = max_trials; self.seed = seed
    def candidates(self) -> Iterable[Dict[str, Any]]:
        return self.space.random(self.max_trials, seed=self.seed)
