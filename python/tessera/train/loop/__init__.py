"""tessera.train.loop — training loops.

RL post-training first (``rl.py``: GRPO / CISPO over ``tessera.rl``), because
that surface already ships VJP+JVP-complete policy losses and is the modern
reason to train MoE models. A next-token ``pretrain.py`` loop follows the same
shape once the Tier-2 autodiff tape is wired through ``nn.Module`` parameter
updates (the integration seam documented in ``rl.py``).
"""

from __future__ import annotations

from .optimizer import adamw_step
from .rl import (
    GRPOConfig,
    RolloutDiagnostics,
    RolloutTokenMetadata,
    grpo_step,
    grpo_surrogate,
    grpo_train_step,
)

__all__ = [
    "GRPOConfig",
    "RolloutDiagnostics",
    "RolloutTokenMetadata",
    "grpo_step",
    "grpo_surrogate",
    "grpo_train_step",
    "adamw_step",
]
