"""``tessera.stdlib`` — shared, model-class production primitives.

The standard library the large-MoE-model frontends (DeepSeek-V3.2 / GLM-5.2 /
Kimi-K2 / MiniMax-M3) compose against.  Distinct from ``tessera.ops`` (the op
catalog) and
``tessera.nn`` (stateful modules): ``stdlib`` is the *compiler-owned lowering
surface* — packed quant layouts + fused dequant-GEMM (``quant``), capacity-aware
MoE dispatch (``moe``), DSpark draft-block proposals (``dspark``), and the
M3/M4 attention surface (``attention``).

See ``docs/audit/roadmap/MODEL_CLASS_ROADMAP.md`` for the milestone map.
"""

from __future__ import annotations

from . import attention, delta_rule, dspark, hybrid, moe, quant

__all__ = ["quant", "moe", "dspark", "attention", "delta_rule", "hybrid"]
