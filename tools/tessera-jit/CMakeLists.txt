# Phase 0 production-lane plumbing — experimental CPU JIT (NOT "runtime v2").
# docs/spec/PRODUCTION_COMPILER_PLAN.md, RUNTIME_ABI_SPEC.md §12.
#
# Builds a shared library `libtessera_jit.{dylib,so}` with a tiny C ABI that
# Python loads via ctypes to prove: MLIR lowers → C-ABI wrapper exists → Python
# calls it → oracle passes (and there is NO numpy fallback in this lane).
#
# IMPORTANT (link model): use a plain add_library + the static MLIR* component
# targets, mirroring tools/tessera-opt. Do NOT use add_mlir_library(SHARED): on
# this Homebrew toolchain that pulls libMLIR.dylib *in addition to* the static
# MLIR .a libs, giving the process TWO copies of MLIR's TypeID / dialect registry
# — which segfaults the moment a dialect/type is constructed. Static MLIR + the
# libLLVM.dylib dependency (exactly what tessera-opt does) is a single copy.

add_library(tessera_jit SHARED tessera_jit.cpp)

target_link_libraries(tessera_jit PRIVATE
  # Tessera
  TesseraIR
  TesseraPasses
  # MLIR core
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRPass
  MLIRTransforms
  # dialects
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRMemRefTransforms
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRControlFlowDialect
  MLIRLLVMDialect
  # conversions used by the pipeline
  MLIRFuncToLLVM
  MLIRArithToLLVM
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRControlFlowToLLVM
  MLIRSCFToControlFlow
  MLIRReconcileUnrealizedCasts
)

# Execution + LLVM-IR translation: link the static .a FILES directly via
# $<TARGET_FILE:...>. Linking these *targets* would drag in their
# INTERFACE_LINK_LIBRARIES — which include the `MLIR` aggregate dylib
# (verified in MLIRTargets.cmake: MLIRExecutionEngine -> "...;MLIR"). That dylib
# coexisting with the static MLIR*.a above = two TypeID/dialect registries =
# segfault on first dialect construction. Linking the archive file bypasses the
# interface and keeps the build single-copy static (LLVM stays the one shared
# libLLVM.dylib, exactly like tessera-opt).
target_link_libraries(tessera_jit PRIVATE
  $<TARGET_FILE:MLIRExecutionEngine>
  $<TARGET_FILE:MLIRExecutionEngineUtils>
  $<TARGET_FILE:MLIRLLVMToLLVMIRTranslation>
  $<TARGET_FILE:MLIRBuiltinToLLVMIRTranslation>
  $<TARGET_FILE:MLIRTargetLLVMIRExport>
  $<TARGET_FILE:MLIRTargetLLVMIRTransforms>
  $<TARGET_FILE:MLIRLLVMIRTransforms>
)

target_include_directories(tessera_jit
  PRIVATE
    ${CMAKE_SOURCE_DIR}/src/transforms/include
    ${CMAKE_SOURCE_DIR}/src/compiler/ir/include
)

# Mirror tessera-opt: silence the Xcode-15 duplicate-library warning (the dupes
# come from TesseraIR/TesseraPasses re-listing the same MLIR libs; the linker
# correctly drops them).
if(APPLE)
  target_link_options(tessera_jit PRIVATE "LINKER:-no_warn_duplicate_libraries")
endif()

# libffi: dynamic-arity dispatch of the c-iface wrapper (any number of memref
# args). System libffi on macOS (/usr/lib) and standard on Linux.
target_link_libraries(tessera_jit PRIVATE ffi)
