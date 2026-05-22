
from __future__ import annotations
import math
from typing import Callable

def _linear(t: float) -> float:
    return t

def _cosine(t: float) -> float:
    # cosine schedule in [0,1], monotone increasing
    return 1.0 - math.cos(0.5 * math.pi * t)

def _poly(t: float, p: float = 2.0) -> float:
    return t ** p

def beta_schedule(kind: str | Callable[[float], float]) -> Callable[[float], float]:
    if callable(kind): return kind
    if kind == "linear": return _linear
    if kind == "cosine": return _cosine
    if kind == "poly":   return _poly
    raise ValueError(f"unknown schedule: {kind}")
