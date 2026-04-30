from __future__ import annotations

from dataclasses import dataclass
import random

from .dataset import Task
from .verifier import verify_answer


@dataclass(frozen=True)
class Rollout:
    completion: str
    reward: float
    metadata: dict[str, object]


class ToyPolicy:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def sample(self, task: Task, n: int) -> list[str]:
        correct = int(task.answer)
        completions = []
        for i in range(n):
            if i == 0 or self.rng.random() < 0.35:
                completions.append(str(correct))
            else:
                delta = self.rng.choice([-3, -2, -1, 1, 2, 3])
                if self.rng.random() < 0.3:
                    completions.append(f"I compute it as {correct + delta}.")
                else:
                    completions.append(str(correct + delta))
        return completions


def collect_grouped_rollouts(policy: ToyPolicy, task: Task, group_size: int) -> list[Rollout]:
    group = []
    for completion in policy.sample(task, group_size):
        reward, metadata = verify_answer(task, completion)
        group.append(Rollout(completion=completion, reward=reward, metadata=metadata))
    return group
