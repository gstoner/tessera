"""Neural network layers — both functional and stateful surfaces.

Functional API (stateless, no `Parameter` ownership):
    tessera.nn.linear / Linear (functional alias)  ← see also nn.layers.Linear (stateful)
    tessera.nn.rms_norm                            ← see also nn.layers.RMSNorm
    tessera.nn.swiglu                              ← see also nn.layers.MLP
    tessera.nn.multi_head_attention                ← see also nn.layers.MultiHeadAttention
    tessera.nn.flash_attention                     ← alias for tessera.ops.flash_attn
    tessera.nn.functional                          ← submodule (torch-style F = nn.functional)

Stateful API (Tier 1 of capability-gap audit — lands today):
    tessera.nn.Module / Parameter / Sequential / ModuleList / ModuleDict
    tessera.nn.Linear / RMSNorm / LayerNorm / Embedding / Dropout / MLP / MultiHeadAttention

Deferred phantoms (raise NotImplementedError with audit pointer):
    nn.BatchNorm1d, nn.Conv2d, nn.LSTM, nn.DynamicDepthwiseConv1d,
    nn.RotaryEmbedding, nn.KVCache, nn.MultiHeadCrossAttention,
    nn.CrossEntropyLoss, nn.CastedLinear, nn.CastedEmbedding, nn.SiLU,
    nn.Sigmoid, nn.GELU, nn.ReLU, nn.Tanh, nn.Identity, nn.utils
"""

from __future__ import annotations

import sys as _sys

# Functional API and submodule alias ------------------------------------------
from . import functional
from .functional import (
    linear,
    rms_norm,
    swiglu,
    multi_head_attention,
    flash_attention,
)

# Stateful API (Tier 1) -------------------------------------------------------
from .module import Module, Parameter, Sequential, ModuleList, ModuleDict
from .layers import (
    Linear,
    RMSNorm,
    LayerNorm,
    Embedding,
    Dropout,
    MLP,
    MultiHeadAttention,
    # Phase A4 additions
    MultiHeadCrossAttention,
    RotaryEmbedding,
    CastedLinear,
    CastedEmbedding,
    SiLU, Sigmoid, GELU, ReLU, Tanh, Identity,
    CrossEntropyLoss,
)
from . import utils


# ─────────────────────────────────────────────────────────────────────────────
# Phantom stateful surface — torch.nn-style names that are NOT yet implemented.
# Each raises `NotImplementedError` with a pointer to the capability-gap audit
# and a workaround hint. When a backlog item lands, replace the phantom with
# the real class and remove its entry below.
# ─────────────────────────────────────────────────────────────────────────────

_AUDIT = "docs/audit/advanced_examples_capability_gap.md"


def _phantom(name: str, hint: str) -> type:
    class _Phantom:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                f"tessera.nn.{name} is on the Tier 1+ backlog — see {_AUDIT}. "
                f"Until it lands, {hint}."
            )

        def __call__(self, *args, **kwargs):
            self.__init__(*args, **kwargs)

    _Phantom.__name__ = name
    _Phantom.__qualname__ = name
    return _Phantom


# Stateful structured modules awaiting follow-on backlog items
BatchNorm1d = _phantom(
    "BatchNorm1d",
    "BatchNorm running stats need a buffer protocol (Phase B1 of execution_roadmap.md)",
)
Conv2d = _phantom("Conv2d", "Conv2d Module is Phase H1; call tessera.ops.conv2d(x, w, ...) directly until then")
LSTM = _phantom("LSTM", "RNN cells are deferred (Phase H2 — out of scope this cycle)")
DynamicDepthwiseConv1d = _phantom(
    "DynamicDepthwiseConv1d",
    "depthwise_conv1d streaming kernels are Phase D of execution_roadmap.md",
)
KVCache = _phantom(
    "KVCache",
    "KVCacheHandle abstraction is Phase B2 + E of execution_roadmap.md; "
    "use tessera.ops.kv_cache_append / kv_cache_prune for now",
)


__all__ = [
    # Functional API
    "linear", "rms_norm", "swiglu", "multi_head_attention",
    "flash_attention", "functional",
    # Stateful API (Tier 1)
    "Module", "Parameter", "Sequential", "ModuleList", "ModuleDict",
    "Linear", "RMSNorm", "LayerNorm", "Embedding", "Dropout", "MLP",
    "MultiHeadAttention",
    # Phase A4 additions (real implementations, not phantoms)
    "MultiHeadCrossAttention", "RotaryEmbedding",
    "CastedLinear", "CastedEmbedding",
    "SiLU", "Sigmoid", "GELU", "ReLU", "Tanh", "Identity",
    "CrossEntropyLoss",
    "utils",
    # Remaining phantoms (deferred to later phases)
    "BatchNorm1d", "Conv2d", "LSTM", "DynamicDepthwiseConv1d", "KVCache",
]
