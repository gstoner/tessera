
from typing import Iterable, Dict, Any, List, Tuple
import math, random

class Hyperband:
    """Produces (config, budget_iters) pairs using Hyperband / Successive Halving.
    - configs are sampled from a given sampling function (callable(k) -> list[dict])
    - budget is expressed as number of iterations for the workload
    """
    def __init__(self, sampler, max_budget: int, min_budget: int = 5, eta: int = 3, seed: int = 17):
        assert max_budget >= min_budget
        self.sampler = sampler
        self.max_budget = max_budget
        self.min_budget = min_budget
        self.eta = eta
        self.rng = random.Random(seed)

    def _brackets(self):
        s_max = int(math.floor(math.log(self.max_budget / self.min_budget, self.eta)))
        B = (s_max + 1) * self.max_budget
        for s in reversed(range(s_max + 1)):
            n = int(math.ceil(B / self.max_budget / (s + 1) * (self.eta ** s)))
            r = self.max_budget * (self.eta ** (-s))
            yield s, n, int(r)

    def candidates(self) -> Iterable[Tuple[Dict[str, Any], int]]:
        # Note: yields pairs; the tuner must understand budgets.
        for s, n, r in self._brackets():
            # sample n configs for this bracket
            cfgs = self.sampler(n)
            # initial rung
            rung = [(cfg, int(r)) for cfg in cfgs]
            for i in range(s + 1):
                for cfg, budget in rung:
                    yield (cfg, budget)
                # after evaluation, the tuner should have scores to pick top 1/eta
                # This generator alone can't filter; selection happens in the tuner loop.
                # We signal rung boundary by yielding None; tuner can act then.
                yield None
