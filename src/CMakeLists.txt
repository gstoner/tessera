# Tessera source tree
# Wires all canonical production components into the build.
# Called from the top-level CMakeLists.txt after LLVM/MLIR cmake modules are loaded.
#
# Layout (post April-2026 reorganization — see PROJECT_STRUCTURE.md):
#
#   ir/                 Core Tessera IR dialect (TesseraDialect, TesseraOps)
#   transforms/         Canonicalize / verify / migrate passes over Tessera IR
#   runtime/            C++ execution engine + CUDA/HIP backends
#   scaling_resilience/ SR dialect: sharding, recompute, resilience passes
#   tile_opt_fa4/       Attn + Queue dialects, FA4 conversion passes
#   tpp/                TPP (Tile Processing Primitives) dialect
#   programming_model/  PM v1.1: tile/cache/schedule/moe ODS + pipeline driver
#   solvers/            Sparse, RNG, physics solver passes
#   tessera_neighbors/  Neighbor topology + stencil lowering passes
#   collectives/        Collective-communication dialect
#   spectral/           Spectral / FFT dialect
#
# NOT in this file (intentionally excluded from production build):
#   research/pddl_instruct        — experimental
#   research/sandbox_compilers    — experimental

# ------------------------------------------------------------------
# Core IR layer (must come first — other components link against it)
# ------------------------------------------------------------------
add_subdirectory(ir)
add_subdirectory(transforms)

# ------------------------------------------------------------------
# Runtime
# ------------------------------------------------------------------
add_subdirectory(runtime)

# ------------------------------------------------------------------
# Canonical dialects and passes
# ------------------------------------------------------------------
add_subdirectory(scaling_resilience)   # SR dialect: resilience/sharding passes
add_subdirectory(tile_opt_fa4)         # Attn + Queue dialects, FA4 conversions
add_subdirectory(tpp)                  # TPP dialect + CI workflow
add_subdirectory(programming_model)    # PM v1.1: tile/cache/schedule/moe ODS
add_subdirectory(solvers)              # Sparse/RNG/physics solvers
add_subdirectory(tessera_neighbors)    # Neighbor topology + stencil passes
add_subdirectory(collectives)          # Collective-communication dialect
add_subdirectory(spectral)             # Spectral/FFT dialect
