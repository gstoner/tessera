
import random
from typing import List, Dict, Any
from .metrics import penalty_score

def roc_select(rollouts: List[Dict[str, Any]], select_size: int) -> List[int]:
    pos_idx = [i for i,r in enumerate(rollouts) if r['reward'] == 1]
    neg_idx = [i for i,r in enumerate(rollouts) if r['reward'] == 0]
    k_half = select_size // 2
    chosen = []
    if len(neg_idx) <= k_half:
        chosen.extend(neg_idx)
    else:
        chosen.extend(random.sample(neg_idx, k_half))
    pos_needed = select_size - len(chosen)
    if pos_idx:
        weights = []
        for i in pos_idx:
            r = rollouts[i]
            p = penalty_score(r['tool_calls'], r['tool_errors'], r['answer_tags'])
            weights.append(1.0 / (1e-6 + p))
        s = sum(weights)
        probs = [w/s for w in weights]
        picked = []
        candidates = list(zip(pos_idx, probs))
        for _ in range(min(pos_needed, len(pos_idx))):
            if not candidates: break
            import random
            r = random.random()
            csum = 0.0
            for j,(idx,p) in enumerate(candidates):
                csum += p
                if r <= csum:
                    picked.append(idx)
                    del candidates[j]
                    total = sum(p for _,p in candidates) or 1.0
                    candidates = [(i, p/total) for i,p in candidates]
                    break
        chosen.extend(picked)
    pool = [i for i in range(len(rollouts)) if i not in chosen]
    while len(chosen) < select_size and pool:
        chosen.append(pool.pop(0))
    return chosen[:select_size]
