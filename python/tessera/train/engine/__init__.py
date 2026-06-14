"""tessera.train.engine — the MoE training engine layer.

One readable file per concern. ``moe.py`` holds routing + the load-balancing
auxiliary loss + the MoE feed-forward block. Pipeline/FSDP/optimizer wrappers
are thin re-exports of the existing standalone surfaces (``tessera.optim``,
``tessera.checkpoint``, ``tessera.distributed``) and are added here as those
execution paths light up — keeping this layer compact by construction.
"""

from __future__ import annotations

from .moe import MoERouter, MoEFeedForward, load_balancing_loss, router_z_loss

__all__ = ["MoERouter", "MoEFeedForward", "load_balancing_loss", "router_z_loss"]
