from __future__ import annotations

import re

from .dataset import Task


_INT_RE = re.compile(r"-?\d+")


def extract_final_int(text: str) -> str | None:
    matches = _INT_RE.findall(text)
    return matches[-1] if matches else None


def verify_answer(task: Task, completion: str) -> tuple[float, dict[str, object]]:
    guess = extract_final_int(completion)
    correct = guess == task.answer
    reward = 1.0 if correct else 0.0
    format_ok = completion.strip() == (guess or "")
    return reward, {"guess": guess, "correct": correct, "format_ok": format_ok}
