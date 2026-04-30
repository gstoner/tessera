from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str
    answer: str
    kind: str = "math"


def make_tasks(seed: int, count: int) -> list[Task]:
    rng = random.Random(seed)
    tasks: list[Task] = []
    for idx in range(count):
        a = rng.randint(2, 19)
        b = rng.randint(2, 19)
        c = rng.randint(1, 9)
        if idx % 2 == 0:
            prompt = f"Solve exactly: ({a} * {b}) + {c}. Return only the integer."
            answer = str(a * b + c)
        else:
            prompt = f"Solve exactly: ({a} + {b}) * {c}. Return only the integer."
            answer = str((a + b) * c)
        tasks.append(Task(task_id=f"math_{idx:04d}", prompt=prompt, answer=answer))
    return tasks
