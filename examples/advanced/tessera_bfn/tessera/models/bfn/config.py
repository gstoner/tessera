
from dataclasses import dataclass
from typing import Literal, Callable, Optional

@dataclass
class BFNConfig:
    data_kind: Literal["discrete", "discretized", "continuous"] = "discrete"
    family: Literal["categorical", "gaussian", "discretized_bins"] = "categorical"
    n_steps: int = 100
    accuracy_schedule: Literal["linear", "cosine", "poly"] | Callable[[float], float] = "cosine"
    net: str | None = None                     # name/handle in Tessera module registry
    eps: float = 1e-6
    dtype: str = "fp32"
