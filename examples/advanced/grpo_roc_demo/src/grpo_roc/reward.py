
from typing import Optional
def answer_only_reward(pred: Optional[int], truth: int) -> int:
    return int(pred == truth)
