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
add_subdirectory(compiler/layout_algebra) # shared host-free layout authority
add_subdirectory(compiler/ir)          # TesseraIR dialect + tablegen
add_subdirectory(collectives)          # Collective Target dialect + transport ABI

# The Tile->x86 pass is part of the shared transform library, but its typed
# Target marker is owned by the optional hardware-free x86 dialect.  Build that
# small IR target before TesseraPasses so the pass can declare the dialect via
# getDependentDialects(); loading it lazily from runOnOperation() is forbidden
# by MLIR's multi-threaded pass-manager contract.  Native x86 kernels remain in
# the later backend subdirectory and are not pulled into the shared layer.
if(TESSERA_BUILD_X86_BACKEND)
  add_subdirectory(compiler/codegen/tessera_x86_backend/lib/IR)
endif()

# The canonical Schedule dialect is shared by distribution lowering and the
# programming-model passes. Define it before TesseraPasses so no transform may
# invent a permissive dialect with the same namespace.
add_subdirectory(compiler/programming_model)    # PM v1.1: Schedule IR ODS
add_subdirectory(transforms)           # All lowering passes (Phases 1–3)

# ------------------------------------------------------------------
# Runtime
# ------------------------------------------------------------------
add_subdirectory(runtime)

# ------------------------------------------------------------------
# Compiler components (dialects and passes under compiler/)
# ------------------------------------------------------------------
add_subdirectory(compiler/tile_opt_fa4)         # Attn + Queue dialects, FA4 conversions
if(NOT ((TESSERA_BUILD_ROCM_BACKEND AND NOT TESSERA_ENABLE_CUDA AND NOT TESSERA_ENABLE_HIP) OR
        (TESSERA_BUILD_NVIDIA_BACKEND AND NOT TESSERA_ENABLE_CUDA)))
  add_subdirectory(compiler/tessera_neighbors)  # Neighbor topology + stencil passes
else()
  message(STATUS "Skipping Neighbors dialect for hardware-free target artifact build")
endif()

# Native x86 kernels only. These are real AVX-512/AMX intrinsics
# (`immintrin.h`), so they cannot compile for a non-x86 host — but the
# hardware-free Target IR dialect added above can, and is all the x86 lit
# fixtures need. Skipping just this half lets an arm64 host build
# `tessera-opt` with `TESSERA_BUILD_X86_BACKEND=ON` and run those fixtures,
# instead of failing the whole `ninja` and leaving them unevaluated.
#
# That matters beyond convenience: Decision #19's standing lesson is that a
# host WITH the ISA cannot falsify a host-portability claim. The arm64 Mac is
# the only machine in the fleet without AVX-512, so it is the only one whose
# green result on those fixtures means anything.
if(TESSERA_BUILD_X86_BACKEND)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86)$")
    add_subdirectory(compiler/codegen/tessera_x86_backend)
  else()
    message(STATUS
      "x86 native kernels skipped on ${CMAKE_SYSTEM_PROCESSOR} "
      "(hardware-free tessera_x86 Target IR dialect still built)")
  endif()
endif()
if(TESSERA_BUILD_NVIDIA_BACKEND)
  add_subdirectory(compiler/codegen/tessera_gpu_backend_NVIDIA)
endif()
if(TESSERA_BUILD_ROCM_BACKEND)
  add_subdirectory(compiler/codegen/Tessera_ROCM_Backend)
endif()
if(TESSERA_BUILD_APPLE_BACKEND)
  add_subdirectory(compiler/codegen/Tessera_Apple_Backend)
endif()

# ------------------------------------------------------------------
# Solver suite and collectives
# ------------------------------------------------------------------
if(NOT ((TESSERA_BUILD_ROCM_BACKEND AND NOT TESSERA_ENABLE_CUDA AND NOT TESSERA_ENABLE_HIP) OR
        (TESSERA_BUILD_NVIDIA_BACKEND AND NOT TESSERA_ENABLE_CUDA)))
  add_subdirectory(solvers)            # scaling_resilience, tpp, spectral, sparse
else()
  message(STATUS "Skipping solver suite for hardware-free target artifact build")
endif()
