"""
checkpoint.py — Activation checkpointing support (Phase 5)

Provides ``@checkpoint_jit`` decorator and ``CollectiveCheckpointConfig``
for selective activation recomputation.  When applied to a function the
decorator attaches metadata so that the Tessera JIT can emit
``tessera_sr.checkpoint`` / ``tessera_sr.recompute_hint`` markers in the
compiled IR.

Usage::

    from tessera.compiler.checkpoint import checkpoint_jit, CheckpointPolicy

    @checkpoint_jit(interval=2)
    def forward(x):
        ...

    # Or with explicit config:
    cfg = CollectiveCheckpointConfig(interval=4, policy=CheckpointPolicy.SELECTIVE)
    annotator = CheckpointIRAnnotator(cfg)
    marked = annotator.annotate(["embed", "attn", "mlp", "norm"])
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Policy enum
# ---------------------------------------------------------------------------

class CheckpointPolicy(Enum):
    """Determines which op boundaries receive a checkpoint marker."""

    NONE = "none"
    SELECTIVE = "selective"   # every ``interval``-th layer (default)
    FULL = "full"             # every layer boundary


# ---------------------------------------------------------------------------
# Configuration object
# ---------------------------------------------------------------------------

@dataclass
class CollectiveCheckpointConfig:
    """
    Configuration for activation checkpointing.

    Parameters
    ----------
    interval : int
        Checkpoint every *interval* ops (only used for SELECTIVE policy).
    policy : CheckpointPolicy
        When to insert checkpoint markers.
    memory_budget_gb : float
        Soft memory budget in GiB.  The pass treats this as a hint for how
        aggressively to recompute.
    save_dir : str
        Directory where checkpoint blobs are written at runtime.
    enabled : bool
        Master switch; if False, no markers are inserted.
    """

    interval: int = 2
    policy: CheckpointPolicy = CheckpointPolicy.SELECTIVE
    memory_budget_gb: float = 40.0
    save_dir: str = "/tmp/tessera_checkpoints"
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError(f"interval={self.interval} must be >= 1")
        if self.memory_budget_gb <= 0:
            raise ValueError(
                f"memory_budget_gb={self.memory_budget_gb} must be > 0"
            )

    def checkpoint_layers(self, layer_names: List[str]) -> List[str]:
        """
        Return the subset of ``layer_names`` that get checkpoint markers.

        - NONE   → empty list
        - FULL   → all layers
        - SELECTIVE → every ``interval``-th layer (indices 0, interval, 2*interval, …)
        """
        if not self.enabled or self.policy == CheckpointPolicy.NONE:
            return []
        if self.policy == CheckpointPolicy.FULL:
            return list(layer_names)
        return [n for i, n in enumerate(layer_names) if i % self.interval == 0]

    def to_ir_attr(self) -> str:
        """Serialise to a tessera MLIR attribute string."""
        return (
            f'{{tessera_sr.checkpoint_config = {{'
            f'policy = "{self.policy.value}", '
            f'interval = {self.interval}, '
            f'memory_budget_gb = {self.memory_budget_gb}}}}}'
        )

    def to_mlir_attrs(self) -> str:
        return self.to_ir_attr()

    def __repr__(self) -> str:
        return (
            f"CollectiveCheckpointConfig(interval={self.interval}, "
            f"policy={self.policy.value!r}, "
            f"memory_budget_gb={self.memory_budget_gb})"
        )


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def checkpoint_jit(
    fn: Optional[Callable] = None,
    *,
    interval: int = 2,
    policy: CheckpointPolicy = CheckpointPolicy.SELECTIVE,
    memory_budget_gb: float = 40.0,
    save_dir: str = "/tmp/tessera_checkpoints",
) -> Callable:
    """
    Extend ``@tessera.jit`` with activation checkpointing.

    Can be used as a bare decorator or with keyword arguments::

        @checkpoint_jit
        def forward(x): ...

        @checkpoint_jit(interval=4, policy=CheckpointPolicy.SELECTIVE)
        def forward(x): ...

    The decorated function gains two attributes:

    - ``__tessera_checkpoint__``        → True
    - ``__tessera_checkpoint_config__`` → CollectiveCheckpointConfig
    """
    cfg = CollectiveCheckpointConfig(
        interval=interval,
        policy=policy,
        memory_budget_gb=memory_budget_gb,
        save_dir=save_dir,
    )

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__tessera_checkpoint__ = True
        wrapper.__tessera_checkpoint_config__ = cfg
        return wrapper

    if fn is not None:
        # Bare @checkpoint_jit usage
        return decorator(fn)
    return decorator


# ---------------------------------------------------------------------------
# IR annotator
# ---------------------------------------------------------------------------

class CheckpointIRAnnotator:
    """
    Annotates a list of layer names with ``tessera_sr.checkpoint`` markers
    according to a ``CollectiveCheckpointConfig``.

    This is the Python-layer analogue of ``InsertRecomputePass`` — it
    produces the attribute strings that the pass would emit.
    """

    def __init__(self, config: CollectiveCheckpointConfig) -> None:
        self.config = config

    def annotate(self, layer_names: List[str]) -> Dict[str, bool]:
        """
        Return a mapping *layer_name → bool* (True = checkpoint marker present).
        """
        checkpointed = set(self.config.checkpoint_layers(layer_names))
        return {name: (name in checkpointed) for name in layer_names}

    def ir_annotations(self, layer_names: List[str]) -> List[str]:
        """
        Return IR attribute strings for layers that receive checkpoint markers.
        """
        ann = self.annotate(layer_names)
        return [
            (
                f'tessera_sr.checkpoint {{layer = "{name}", '
                f'policy = "{self.config.policy.value}"}}'
            )
            for name, marked in ann.items()
            if marked
        ]

    def recompute_hints(self, layer_names: List[str]) -> List[str]:
        """
        Return the subset of layers that sit *between* two checkpoint markers —
        these are candidates for recomputation during backward.
        """
        ann = self.annotate(layer_names)
        hints = []
        in_segment = False
        for name in layer_names:
            if ann[name]:
                in_segment = True
                continue
            if in_segment:
                hints.append(name)
        return hints

    def __repr__(self) -> str:
        return f"CheckpointIRAnnotator(config={self.config!r})"
