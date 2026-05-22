
import re

ANS_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)

def extract_answer(text: str):
    m = ANS_RE.search(text or "")
    if not m:
        return None
    return m.group(1).strip()

def is_number(x: str):
    try:
        float(x.replace(",", ""))
        return True
    except Exception:
        return False

def numeric_equal(a: str, b: str, eps: float = 1e-6):
    if is_number(a) and is_number(b):
        return abs(float(a.replace(",", "")) - float(b.replace(",", ""))) <= eps
    return a == b

def math_reward(generation: str, gold: str) -> float:
    \"\"\"Return 1.0 if <answer> equals gold (numeric-equal), else 0.0.\"\"\"
    pred = extract_answer(generation)
    if pred is None:
        return 0.0
    return 1.0 if numeric_equal(pred, str(gold)) else 0.0
