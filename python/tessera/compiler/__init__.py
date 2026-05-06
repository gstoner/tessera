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
from .schedule_ir import ScheduleIRModule, ScheduleFunction, ScheduleOp, lower_graph_to_schedule_ir
from .tile_ir import TileIRModule, TileFunction, TileOp, lower_schedule_to_tile_ir
from .jit import jit, TesseraJitError
from .driver import CompileArtifactBundle, CompileRequest, CompileTraceEvent, compile_graph_module
from .frontend import FrontendSemanticError, FrontendSyntaxError, lower_text_to_graph_ir, parse_text

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
    "ScheduleIRModule",
    "ScheduleFunction",
    "ScheduleOp",
    "lower_graph_to_schedule_ir",
    "TileIRModule",
    "TileFunction",
    "TileOp",
    "lower_schedule_to_tile_ir",
    "jit",
    "TesseraJitError",
    "CompileArtifactBundle",
    "CompileRequest",
    "CompileTraceEvent",
    "compile_graph_module",
    "FrontendSemanticError",
    "FrontendSyntaxError",
    "lower_text_to_graph_ir",
    "parse_text",
]
