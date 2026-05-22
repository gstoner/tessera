
import random
from typing import List, Tuple

def gen_integer_math(n: int) -> List[Tuple[str,int]]:
    problems = []
    ops = ['+', '-', '*']
    for _ in range(n):
        a, b = random.randint(1, 99), random.randint(1, 99)
        op = random.choice(ops)
        q = f"Compute {a} {op} {b}. Return only the final integer in \\boxed{{like_this}} inside an answer tag."
        if op == '+': ans = a + b
        elif op == '-': ans = a - b
        else: ans = a * b
        problems.append((q, ans))
    return problems
