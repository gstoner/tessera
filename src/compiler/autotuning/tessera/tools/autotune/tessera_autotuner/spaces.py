
from dataclasses import dataclass, field
from typing import Any, Dict, List, Iterable
import itertools, random

@dataclass
class Param:
    name: str
    kind: str
    values: List[Any]

@dataclass
class SearchSpace:
    params: List[Param] = field(default_factory=list)
    def add(self, name: str, kind: str, values: List[Any]):
        self.params.append(Param(name, kind, values))
    def grid(self) -> Iterable[Dict[str, Any]]:
        names = [p.name for p in self.params]
        prods = [p.values for p in self.params]
        for combo in itertools.product(*prods):
            yield dict(zip(names, combo))
    def random(self, n: int, seed: int = 17) -> Iterable[Dict[str, Any]]:
        rng = random.Random(seed)
        for _ in range(n):
            yield {p.name: rng.choice(p.values) for p in self.params}
    def size_estimate(self) -> int:
        x = 1
        for p in self.params:
            x *= max(1, len(p.values))
        return x
