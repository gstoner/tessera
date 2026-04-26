# Tessera source tree
# Wires all canonical production components into the build.
# Called from the top-level CMakeLists.txt after LLVM/MLIR cmake modules are loaded.
#
# Layout (post April-2026 reorganization — see PROJECT_STRUCTURE.md):
#
#   compiler/ir/               Core Tessera IR dialect (TesseraDialect, TesseraOps)
#   transforms/                Canonicalize / verify / migrate / lowering passes
#   runtime/                   C++ execution engine + CUDA/HIP backends
#   compiler/tile_opt_fa4/     Attn + Queue dialects, FA4 conversion passes
#   compiler/programming_model/PM v1.1: tile/cache/schedule/moe ODS + pipeline driver
#   compiler/tessera_neighbors/Neighbor topology + stencil lowering passes
#   solvers/                   Scaling-resilience / TPP / spectral / sparse solvers
#   collectives/               Collective-communication dialect
#
# NOT in this file (intentionally excluded from production build):
#   research/pddl_instruct        — experimental
#   research/sandbox_compilers    — experimental

# ------------------------------------------------------------------
# Core IR layer (must come first — other components link against it)
# ------------------------------------------------------------------
add_subdirectory(compiler/ir)          # TesseraIR dialect + tablegen
add_subdirectory(transforms)           # All lowering passes (Phases 1–3)

# ------------------------------------------------------------------
# Runtime
# ------------------------------------------------------------------
add_subdirectory(runtime)

# ------------------------------------------------------------------
# Compiler components (dialects and passes under compiler/)
# ------------------------------------------------------------------
add_subdirectory(compiler/tile_opt_fa4)         # Attn + Queue dialects, FA4 conversions
add_subdirectory(compiler/programming_model)    # PM v1.1: Schedule IR ODS
add_subdirectory(compiler/tessera_neighbors)    # Neighbor topology + stencil passes

# ------------------------------------------------------------------
# Solver suite and collectives
# ------------------------------------------------------------------
add_subdirectory(solvers)              # scaling_resilience, tpp, spectral, sparse
add_subdirectory(collectives)          # Collective-communication dialect
