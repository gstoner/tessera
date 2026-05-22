
from __future__ import annotations
from typing import Any

def generate(prior_params, *, cfg, n_steps:int|None=None) -> Any:
    """Run the BFN generative process for `n_steps` (few-step generation is supported)."""
    raise NotImplementedError
