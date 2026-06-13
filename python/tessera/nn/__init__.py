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
    adaptive_pool,
    alibi,
    avg_pool,
    bidirectional_scan,
    conv1d,
    conv_transpose,
    einsum,
    group_norm,
    gqa_attention,
    gru_cell,
    instance_norm,
    linear,
    linear_general,
    lora_linear,
    max_pool,
    min_pool,
    mla_decode,
    moe_layer,
    mqa_attention,
    ntk_rope,
    rms_norm,
    simple_rnn_cell,
    spectral_norm,
    swiglu,
    weight_norm,
    multi_head_attention,
    flash_attention,
)

# Stateful API (Tier 1) -------------------------------------------------------
from .module import Module, Parameter, Buffer, Sequential, ModuleList, ModuleDict
from .layers import (
    Linear,
    LinearGeneral,
    Einsum,
    LoRALinear,
    Conv1d,
    ConvTranspose1d,
    ConvTranspose,
    RMSNorm,
    LayerNorm,
    GroupNorm,
    InstanceNorm,
    WeightNorm,
    SpectralNorm,
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
    # Phase C1 (depends on B1 buffer protocol)
    BatchNorm1d,
    # Phase C2 (depends on B2 KVCacheHandle)
    KVCache,
    # Phase D4 (depends on D1 + B1)
    DynamicDepthwiseConv1d,
    # Phase H1 (NHWC default + NCHW shim)
    Conv2d,
    Conv2dNCHW,
    # Phase H2 — RNN cells with state-propagation primitive (ops.lstm_cell + extractors)
    SimpleRNNCell,
    GRUCell,
    LSTMCell,
    LSTM,
    # attention_variants_plan, NSA-3 — DeepSeek Native Sparse Attention.
    NativeSparseAttention,
    # execution_roadmap.md, Phase F-MoR — Mixture of Recursions.
    MixtureOfRecursions,
    # MSA Phase 4 — MiniMax Sparse Attention GQA block-sparse layer.
    MinimaxSparseAttention,
)
from . import utils


# ─────────────────────────────────────────────────────────────────────────────
# Phantom stateful surface — torch.nn-style names that are NOT yet implemented.
# Each raises `NotImplementedError` with a pointer to the capability-gap audit
# and a workaround hint. When a backlog item lands, replace the phantom with
# the real class and remove its entry below.
# ─────────────────────────────────────────────────────────────────────────────

_AUDIT = "docs/audit/coverage/COVERAGE_AUDIT.md"


def _phantom(name: str, hint: str) -> type:
    class _Phantom:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                f"tessera.nn.{name} is on the Tier 1+ backlog — see {_AUDIT}. "
                f"Until it lands, {hint}."
            )

        def __call__(self, *args, **kwargs):
            # Reuse __init__'s "this is a phantom" error message. Calling
            # `type(self).__init__(self, ...)` instead of
            # `self.__init__(...)` keeps mypy --check-untyped-defs happy
            # (instance.__init__ binding is reported as unsound).
            type(self).__init__(self, *args, **kwargs)

    _Phantom.__name__ = name
    _Phantom.__qualname__ = name
    return _Phantom


# Stateful structured modules awaiting follow-on backlog items


__all__ = [
    # Functional API
    "linear", "linear_general", "einsum", "conv1d", "conv_transpose",
    "group_norm", "instance_norm", "weight_norm", "spectral_norm",
    "max_pool", "avg_pool", "min_pool", "adaptive_pool",
    "simple_rnn_cell", "gru_cell", "bidirectional_scan", "lora_linear",
    "alibi", "ntk_rope", "gqa_attention", "mqa_attention", "mla_decode",
    "rms_norm", "swiglu", "moe_layer", "multi_head_attention",
    "flash_attention", "functional",
    # Stateful API (Tier 1)
    "Module", "Parameter", "Buffer", "Sequential", "ModuleList", "ModuleDict",
    "Linear", "LinearGeneral", "Einsum", "LoRALinear",
    "Conv1d", "ConvTranspose1d", "ConvTranspose",
    "RMSNorm", "LayerNorm", "GroupNorm", "InstanceNorm", "WeightNorm", "SpectralNorm",
    "Embedding", "Dropout", "MLP",
    "MultiHeadAttention",
    # Phase A4 additions (real implementations, not phantoms)
    "MultiHeadCrossAttention", "RotaryEmbedding",
    "CastedLinear", "CastedEmbedding",
    "SiLU", "Sigmoid", "GELU", "ReLU", "Tanh", "Identity",
    "CrossEntropyLoss",
    "utils",
    # Phase C (real implementations using B1/B2 protocols)
    "BatchNorm1d", "KVCache",
    # Phase D4 (real, depends on D1 streaming kernels + B1 buffers)
    "DynamicDepthwiseConv1d",
    # Phase H1 (NHWC default; NCHW shim transposes in/out)
    "Conv2d", "Conv2dNCHW",
    # Phase H2 (RNN cells)
    "SimpleRNNCell", "GRUCell", "LSTMCell", "LSTM",
    # attention_variants_plan, NSA-3
    "NativeSparseAttention",
    # Phase F-MoR — Mixture of Recursions
    "MixtureOfRecursions",
    # MSA Phase 4 — MiniMax Sparse Attention GQA layer
    "MinimaxSparseAttention",
]
