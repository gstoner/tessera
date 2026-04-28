"""
tessera.compiler — frontend compiler passes for the Python API layer.

Components:
    constraints.py  — ConstraintSolver: Divisible, Range, Equal predicates
                       checked at @jit decoration time
    effects.py      — EffectLattice: infers random/io/memory/pure through
                       the call graph; enforces @deterministic contracts
    graph_ir.py     — Python → Graph IR lowering (emits MLIR text)
    jit.py          — @jit and @kernel decorators that drive the pipeline

Build order for Phase 1:
    1. constraints.py (no deps)
    2. effects.py (no deps)
    3. graph_ir.py (depends on constraints, effects, distributed.*)
    4. jit.py (depends on graph_ir)
"""

from .constraints import ConstraintSolver, Divisible, Range, Equal, TesseraConstraintError
from .effects import EffectLattice, Effect, TesseraEffectError
from .graph_ir import NumericPolicy, KVCacheSpec
from .jit import jit, TesseraJitError

__all__ = [
    "ConstraintSolver",
    "Divisible",
    "Range",
    "Equal",
    "TesseraConstraintError",
    "EffectLattice",
    "Effect",
    "TesseraEffectError",
    "NumericPolicy",
    "KVCacheSpec",
    "jit",
    "TesseraJitError",
]
