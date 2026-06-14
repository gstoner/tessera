"""tessera.train.models — application layer: one self-contained file per model.

Each model is built by *direct instantiation* (principle #3: no implicit
indirection). To port a new architecture, copy ``qwen3_moe.py`` to a new file
and edit it in place — there is no shared layer skeleton, ``ModuleSpec``, or
plugin registry to thread a change through. The ``add-moe-model`` skill
(``tessera/train/skills/add-moe-model/``) encodes this procedure.
"""

from __future__ import annotations

from .qwen3_moe import Qwen3MoEConfig, Qwen3MoEBlock, Qwen3MoEModel
from .moba import MoBAConfig, MoBABlock, MoBAModel, moba_attention
from .traced_moe_policy import TracedMoEPolicy
from .traced_hard_moe_lm import TracedHardMoEConfig, TracedHardMoELM, traced_ce_loss

__all__ = [
    "Qwen3MoEConfig", "Qwen3MoEBlock", "Qwen3MoEModel",
    "MoBAConfig", "MoBABlock", "MoBAModel", "moba_attention",
    "TracedMoEPolicy",
    "TracedHardMoEConfig", "TracedHardMoELM", "traced_ce_loss",
]
