
"""Core BFN primitives (API sketch).
These are *placeholders* whose bodies should be swapped for Tessera tensor ops / kernels.
"""
from __future__ import annotations
from typing import Tuple, Dict, Any
import math

def sender(x, beta, *, family: str, cfg) -> Dict[str, Any]:
    """Return factorized *sender* distribution params given data `x` and accuracy `beta`.
    Discrete: returns probabilities with noise blended per schedule.
    Continuous: returns (mean,var) after injecting noise with accuracy `beta`.
    """
    raise NotImplementedError

def receiver(out_params, beta, *, family: str, cfg) -> Dict[str, Any]:
    """Receiver = convolution of output dist with noise of accuracy `beta`.
    Implement family-specific closed forms.
    """
    raise NotImplementedError

def kl_div(receiver_params, sender_params, *, family: str, cfg) -> Any:
    """KL(receiver || sender), summed over factorized variables (per-batch).
    For categorical in log-space, use logsumexp-stable forms.
    """
    raise NotImplementedError

def bayes_update(prior_params, sender_sample, *, family: str, cfg) -> Dict[str, Any]:
    """Closed-form Bayesian update for factorized prior given a sample from sender.
    Categorical: Dirichlet-like additive update on logits/probs.
    Gaussian: conjugate Normal update of (mean,var).
    """
    raise NotImplementedError

def step(prior_params, x, t, *, family: str, cfg, net=None) -> Dict[str, Any]:
    """One BFN step: net(prior_params,t) → out_params → receiver → sample(sender) → bayes_update.
    `net` is resolved via Tessera module registry by cfg.net if None.
    Returns updated `prior_params`.
    """
    # Pseudocode only
    # out_params = net(prior_params, t)
    # recv = receiver(out_params, beta=t, family=family, cfg=cfg)
    # send = sender(x, beta=t, family=family, cfg=cfg)
    # s = sample(send)   # reparameterized when possible
    # return bayes_update(prior_params, s, family=family, cfg=cfg)
    raise NotImplementedError
