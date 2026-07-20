
add_executable(tessera-opt tessera-opt.cpp)
target_link_libraries(tessera-opt PRIVATE
  MLIRIR
  MLIRPass
  MLIRParser
  MLIRSupport
  MLIRTransforms
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIRROCDLDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRVectorDialect
  MLIRTensorDialect
  MLIRUBDialect
  # Phase 4 GPU emission (2026-06-17): the linalg→gpu→NVVM passes (emission only;
  # launch stays hardware-gated). Transitive deps resolve via MLIR's targets.
  MLIRGPUDialect
  MLIRGPUTransforms
  MLIRSCFToGPU
  MLIRGPUToNVVMTransforms
  MLIRGPUToROCDLTransforms
  MLIRAffineToStandard
  MLIROptLib
)

# Quiet the macOS linker's "ignoring duplicate libraries" warning.
# The duplicates come from the subdir libraries (TesseraIR /
# tessera_solvers_passes / TesseraApple) listing the same MLIR
# libs we list above; the linker correctly drops the dupes — it
# just warns about doing so.  The warning was added in Xcode 15;
# the `-no_warn_duplicate_libraries` flag silences it without
# changing link behavior.
if(APPLE)
  target_link_options(tessera-opt PRIVATE
    "LINKER:-no_warn_duplicate_libraries")
endif()

# Link core Tessera IR + passes (the `tessera` dialect, `tessera-to-linalg`, the
# Phase-1 folders/canonicalizers, and the `--tessera-emit-{nvvm,rocdl}` MLIR-graph
# route) UNLESS this is a hardware-free *target-artifact* build — a backend built
# with no real toolchain present, kept lean to avoid pulling every upstream MLIR
# component into the artifact driver. A backend built against a real toolchain
# (CUDA or HIP) IS a full build and gets the core dialect + the emit route.
#
# The CUDA carve-out (`AND NOT TESSERA_ENABLE_CUDA`) already gave real NVIDIA
# builds the full route; HIP was simply missing the symmetric carve-out, which
# silently stripped the `tessera` dialect + `tessera-to-linalg` from tessera-opt
# on a real ROCm box and left `--tessera-emit-rocdl` unable to parse a tessera
# kernel (so the ROCDL emitter rode the direct rocdl_emit.py LLVM-IR path).
if(NOT ((TESSERA_BUILD_ROCM_BACKEND AND NOT TESSERA_ENABLE_CUDA AND NOT TESSERA_ENABLE_HIP) OR
        (TESSERA_BUILD_NVIDIA_BACKEND AND NOT TESSERA_ENABLE_CUDA)))
  target_compile_definitions(tessera-opt PRIVATE TESSERA_HAVE_CORE_TESSERA_IR)
  target_link_libraries(tessera-opt PRIVATE
    TesseraIR
    TesseraPasses
  )
endif()

# Sprint V7 (2026-05-22): link the FA-4 attention dialect into
# tessera-opt so `tessera_attn.scaled_dot_product` and friends are
# recognized at parse time.  The library is exported by
# src/compiler/tile_opt_fa4/CMakeLists.txt as TesseraAttnDialect.
if(TARGET TesseraAttnDialect)
  target_compile_definitions(tessera-opt PRIVATE TESSERA_HAVE_FA4_ATTN)
  target_include_directories(tessera-opt PRIVATE
    ${CMAKE_SOURCE_DIR}/src/compiler/tile_opt_fa4/include
    ${CMAKE_BINARY_DIR}/src/compiler/tile_opt_fa4
  )
  target_link_libraries(tessera-opt PRIVATE TesseraAttnDialect)
endif()

# Sprint V8 (2026-05-22): link the FA-4 tile-queue dialect into
# tessera-opt so `tessera.queue.create / push / pop` ops + the
# `!tessera.queue.tile_queue` / `!tessera.queue.token` types are
# recognized at parse time.  Mirrors the V7 Attn pattern above.
if(TARGET TesseraQueueDialect)
  target_compile_definitions(tessera-opt PRIVATE TESSERA_HAVE_FA4_QUEUE)
  target_include_directories(tessera-opt PRIVATE
    ${CMAKE_SOURCE_DIR}/src/compiler/tile_opt_fa4/include
    ${CMAKE_BINARY_DIR}/src/compiler/tile_opt_fa4
  )
  target_link_libraries(tessera-opt PRIVATE TesseraQueueDialect)
endif()

if(TARGET tessera_solvers_passes AND TARGET tessera_linalg_solver_passes)
  target_compile_definitions(tessera-opt PRIVATE TESSERA_HAVE_SOLVERS)
  target_include_directories(tessera-opt PRIVATE
    ${CMAKE_SOURCE_DIR}/src/solvers/core/passes
    ${CMAKE_SOURCE_DIR}/src/solvers/linalg/include
    ${CMAKE_BINARY_DIR}/src/solvers/linalg
  )
  target_link_libraries(tessera-opt PRIVATE
    tessera_solvers_passes
    tessera_linalg_solver_passes
  )
endif()

if(TARGET TesseraSR)
  target_compile_definitions(tessera-opt PRIVATE TESSERA_HAVE_SCALING_RESILIENCE)
  target_include_directories(tessera-opt PRIVATE
    ${CMAKE_SOURCE_DIR}/src/solvers/scaling_resilience/include
  )
  target_link_libraries(tessera-opt PRIVATE TesseraSR)
endif()

if(TARGET TesseraNeighbors)
  target_compile_definitions(tessera-opt PRIVATE TESSERA_HAVE_NEIGHBORS)
  target_include_directories(tessera-opt PRIVATE
    ${CMAKE_SOURCE_DIR}/src/compiler/tessera_neighbors/include
  )
  target_link_libraries(tessera-opt PRIVATE TesseraNeighbors)
endif()

# TPP solver dialect + passes + `-tpp-space-time` alias.
if(TARGET TesseraTPPInit)
  target_compile_definitions(tessera-opt PRIVATE TESSERA_HAVE_TPP)
  target_include_directories(tessera-opt PRIVATE
    ${CMAKE_SOURCE_DIR}/src/solvers/tpp/include
    ${CMAKE_BINARY_DIR}/src/solvers/tpp/include/tpp/Dialect/TPP
  )
  target_link_libraries(tessera-opt PRIVATE
    TesseraTPPInit
    TesseraTPPDialect
    TesseraTPPPasses
  )
endif()

# Phase 8: Apple Silicon backend (TESSERA_BUILD_APPLE_BACKEND).
# Only linked when the backend subdirectory has been added by src/CMakeLists.txt.
if(TARGET TesseraApple)
  target_compile_definitions(tessera-opt PRIVATE TESSERA_HAVE_APPLE_BACKEND)
  target_link_libraries(tessera-opt PRIVATE TesseraApple)
endif()

# ROCm backend: hardware-free Target IR / ROCDL artifact pipeline.
if(TARGET TesseraROCMConversion)
  target_compile_definitions(tessera-opt PRIVATE TESSERA_HAVE_ROCM_BACKEND)
  target_include_directories(tessera-opt PRIVATE
    ${CMAKE_SOURCE_DIR}/src/compiler/codegen/Tessera_ROCM_Backend/include
    ${CMAKE_BINARY_DIR}/src/compiler/codegen/Tessera_ROCM_Backend/lib/IR
  )
  target_link_libraries(tessera-opt PRIVATE TesseraROCMIR TesseraROCMConversion)
  # Stage L3 — in-process MLIR -> hsaco serialization (no mlir-opt shell-out):
  # the GPU/ROCDL -> LLVM-IR translations + the #rocdl.target serializer that
  # gpu-module-to-binary drives (AMDGPU codegen from the shared libLLVM; ld.lld
  # from the platform LLVM). Only needed in a full ROCm build (real toolchain).
  if(TESSERA_BUILD_ROCM_BACKEND AND (TESSERA_ENABLE_HIP OR TESSERA_ENABLE_CUDA))
    target_link_libraries(tessera-opt PRIVATE
      MLIRROCDLTarget
      MLIRGPUToLLVMIRTranslation
      MLIRROCDLToLLVMIRTranslation
      MLIRBuiltinToLLVMIRTranslation
      MLIRLLVMToLLVMIRTranslation
      MLIRTargetLLVMIRExport
      MLIRSCFToControlFlow            # convert-scf-to-cf (finish-lower)
      MLIRReconcileUnrealizedCasts    # reconcile-unrealized-casts (finish-lower)
      MLIRMathToLLVM                  # flash_attn softmax exp (math.exp -> llvm)
      MLIRMathDialect
      # ConvertToLLVM external models gathered by convert-gpu-to-rocdl.
      MLIRControlFlowToLLVM
      MLIRArithToLLVM
      MLIRFuncToLLVM
      MLIRMemRefToLLVM
      MLIRVectorToLLVM
      MLIRIndexToLLVM
      MLIRUBToLLVM
    )
  endif()
endif()

# NVIDIA backend: hardware-free Hopper/Blackwell Target IR / NVVM artifact pipeline.
if(TARGET TesseraNVIDIAConversion)
  target_compile_definitions(tessera-opt PRIVATE TESSERA_HAVE_NVIDIA_BACKEND)
  target_include_directories(tessera-opt PRIVATE
    ${CMAKE_SOURCE_DIR}/src/compiler/codegen/tessera_gpu_backend_NVIDIA/include
    ${CMAKE_BINARY_DIR}/src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/IR
  )
  target_link_libraries(tessera-opt PRIVATE
    TesseraNVIDIAIR
    TesseraNVIDIAConversion
    MLIRNVVMDialect
  )
endif()
