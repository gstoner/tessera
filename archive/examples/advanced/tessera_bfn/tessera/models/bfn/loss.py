
from __future__ import annotations
from typing import Any
from .accuracy import beta_schedule

def discrete_time_loss(prior_params, x, *, cfg) -> Any:
    """Sum_t KL(receiver_t || sender_t).
    Implement by unrolling t=1..cfg.n_steps and accumulating KL at each step.
    """
    raise NotImplementedError

def continuous_time_loss(prior_params, x, *, cfg, n_mc:int=1) -> Any:
    """Monte Carlo estimate of the continuous-time loss:
      E_{t ~ U(0,1)}[ KL_rate(receiver_t || sender_t) ].
    """
    raise NotImplementedError
