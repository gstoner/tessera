
add_executable(tessera-opt tessera-opt.cpp)
target_link_libraries(tessera-opt PRIVATE
  TesseraPM
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
  MLIRRegisterAllExtensions
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

# ---------------------------------------------------------------------------
# Feature ledger.
#
# Every optional capability compiled into tessera-opt registers itself here, so
# that (a) the leanness contract below can be validated instead of assumed, and
# (b) the binary can report what it is at runtime (`--tessera-build-info`).
# Record a feature with `tessera_opt_feature(<label> <TESSERA_HAVE_MACRO>)`
# rather than calling target_compile_definitions directly — a feature that is
# defined but not recorded is invisible to both of those.
# ---------------------------------------------------------------------------
set(TESSERA_OPT_FEATURES "")
function(tessera_opt_feature label macro)
  target_compile_definitions(tessera-opt PRIVATE ${macro})
  set(TESSERA_OPT_FEATURES "${TESSERA_OPT_FEATURES};${label}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# Leanness contract.
#
# A hardware-free *target-artifact* build — a NVIDIA/ROCm backend configured
# with no real toolchain present — deliberately stays lean, registering only
# what the target contract spine needs instead of pulling in every upstream
# MLIR component. A backend built against a real toolchain (CUDA or HIP) IS a
# full build and gets the core dialect plus the `--tessera-emit-*` route.
#
# This is an intent about the *driver*, so it is named once, here, as
# TESSERA_OPT_LEAN_ARTIFACT_DRIVER. It used to be re-derived inside
# tessera-opt.cpp from `(ROCM || NVIDIA) && !CORE_TESSERA_IR`, which made the
# lean path swallow every unrelated feature: configuring NVIDIA (no CUDA)
# together with the Apple backend left TesseraApple linked and
# TESSERA_HAVE_APPLE_BACKEND defined while its registration block — living in
# the non-lean arm — was compiled out, with no diagnostic. Same for Solvers,
# Neighbors, TPP, and scaling-resilience. Explicitly requested incompatible
# features remain a configure error. Ambient default-source targets such as
# the two FA-4 dialects are excluded from the lean driver's link/feature ledger
# instead of making the legitimate backend-only configuration impossible.
if((TESSERA_BUILD_ROCM_BACKEND AND NOT TESSERA_ENABLE_CUDA AND NOT TESSERA_ENABLE_HIP) OR
   (TESSERA_BUILD_NVIDIA_BACKEND AND NOT TESSERA_ENABLE_CUDA))
  set(TESSERA_OPT_LEAN_ARTIFACT_DRIVER ON)
  target_compile_definitions(tessera-opt PRIVATE TESSERA_OPT_LEAN_ARTIFACT_DRIVER)
else()
  set(TESSERA_OPT_LEAN_ARTIFACT_DRIVER OFF)
  tessera_opt_feature(core-tessera-ir TESSERA_HAVE_CORE_TESSERA_IR)
  target_link_libraries(tessera-opt PRIVATE
    TesseraIR
    TesseraPasses
  )
endif()

# Sprint V7 (2026-05-22): link the FA-4 attention dialect into
# tessera-opt so `tessera_attn.scaled_dot_product` and friends are
# recognized at parse time.  The library is exported by
# src/compiler/tile_opt_fa4/CMakeLists.txt as TesseraAttnDialect.
# The FA-4 targets are part of the default source build, but they are not part
# of the hardware-free NVIDIA/ROCm artifact driver.  Treating their mere target
# existence as driver intent made the documented lean backend configuration
# self-conflicting: fa4-attn/fa4-queue were linked and then (correctly) rejected
# as registrations the lean arm cannot install.  Full drivers still own both
# dialects; lean drivers exclude them before feature accounting/linking.
if(TARGET TesseraAttnDialect AND NOT TESSERA_OPT_LEAN_ARTIFACT_DRIVER)
  tessera_opt_feature(fa4-attn TESSERA_HAVE_FA4_ATTN)
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
if(TARGET TesseraQueueDialect AND NOT TESSERA_OPT_LEAN_ARTIFACT_DRIVER)
  tessera_opt_feature(fa4-queue TESSERA_HAVE_FA4_QUEUE)
  target_include_directories(tessera-opt PRIVATE
    ${CMAKE_SOURCE_DIR}/src/compiler/tile_opt_fa4/include
    ${CMAKE_BINARY_DIR}/src/compiler/tile_opt_fa4
  )
  target_link_libraries(tessera-opt PRIVATE TesseraQueueDialect)
endif()

# W0.10 (2026-08-02): link the hardware-free x86 Target IR dialect so
# `tessera_x86.*` ops and the `!tessera_x86.tile` type are recognized at parse
# time. Until this landed, x86 was the ONE backend with no Target IR dialect,
# though Decision #19 says backends MUST expose one. Mirrors the Attn/Queue
# pattern above.
if(TARGET TesseraX86IR AND NOT TESSERA_OPT_LEAN_ARTIFACT_DRIVER)
  tessera_opt_feature(x86-target-ir TESSERA_HAVE_X86_TARGET_IR)
  target_include_directories(tessera-opt PRIVATE
    ${CMAKE_SOURCE_DIR}/src/compiler/codegen/tessera_x86_backend/include
    ${CMAKE_BINARY_DIR}/src/compiler/codegen/tessera_x86_backend/lib/IR
  )
  target_link_libraries(tessera-opt PRIVATE TesseraX86IR)
endif()

if(TARGET tessera_solvers_passes AND TARGET tessera_linalg_solver_passes)
  tessera_opt_feature(solvers TESSERA_HAVE_SOLVERS)
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
  tessera_opt_feature(scaling-resilience TESSERA_HAVE_SCALING_RESILIENCE)
  target_include_directories(tessera-opt PRIVATE
    ${CMAKE_SOURCE_DIR}/src/solvers/scaling_resilience/include
  )
  target_link_libraries(tessera-opt PRIVATE TesseraSR)
endif()

if(TARGET TesseraNeighbors)
  tessera_opt_feature(neighbors TESSERA_HAVE_NEIGHBORS)
  target_include_directories(tessera-opt PRIVATE
    ${CMAKE_SOURCE_DIR}/src/compiler/tessera_neighbors/include
  )
  target_link_libraries(tessera-opt PRIVATE TesseraNeighbors)
endif()

# TPP solver dialect + passes + `-tpp-space-time` alias.
if(TARGET TesseraTPPInit)
  tessera_opt_feature(tpp TESSERA_HAVE_TPP)
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
  tessera_opt_feature(apple-backend TESSERA_HAVE_APPLE_BACKEND)
  target_link_libraries(tessera-opt PRIVATE TesseraApple)
endif()

# ROCm backend: hardware-free Target IR / ROCDL artifact pipeline.
if(TARGET TesseraROCMConversion)
  tessera_opt_feature(rocm-backend TESSERA_HAVE_ROCM_BACKEND)
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
  tessera_opt_feature(nvidia-backend TESSERA_HAVE_NVIDIA_BACKEND)
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

# ---------------------------------------------------------------------------
# Validate the leanness contract, and stamp the build identity.
#
# The lean artifact driver registers only the NVIDIA/ROCm target-contract
# spine. Any *other* feature that got linked in would have its registration
# silently compiled out, producing a binary that claims a capability it does
# not have. That must be a configure error, not a runtime surprise.
# ---------------------------------------------------------------------------
set(_tessera_opt_lean_permitted core-tessera-ir nvidia-backend rocm-backend)
set(_tessera_opt_conflicts "")
if(TESSERA_OPT_LEAN_ARTIFACT_DRIVER)
  foreach(_feature IN LISTS TESSERA_OPT_FEATURES)
    if(_feature AND NOT _feature IN_LIST _tessera_opt_lean_permitted)
      list(APPEND _tessera_opt_conflicts ${_feature})
    endif()
  endforeach()
endif()
if(_tessera_opt_conflicts)
  string(REPLACE ";" ", " _tessera_opt_conflict_text "${_tessera_opt_conflicts}")
  message(FATAL_ERROR
    "tessera-opt is configured as a lean hardware-free artifact driver "
    "(TESSERA_OPT_LEAN_ARTIFACT_DRIVER), which registers only the NVIDIA/ROCm "
    "target-contract spine — but these features are also linked in and would "
    "be silently unregistered: ${_tessera_opt_conflict_text}.\n"
    "Pick one:\n"
    "  * full driver — enable the matching real toolchain "
    "(-DTESSERA_ENABLE_CUDA=ON or -DTESSERA_ENABLE_HIP=ON), or configure "
    "without -DTESSERA_BUILD_{NVIDIA,ROCM}_BACKEND=ON;\n"
    "  * lean driver — drop the extra backends/dialects from this build "
    "(for example -DTESSERA_BUILD_APPLE_BACKEND=OFF).\n"
    "Building both in one tessera-opt is not supported: the lean registration "
    "path has no arm for them.")
endif()

# Runtime build identity. Without this the only way to tell two tessera-opt
# binaries apart is to diff `--help`, which is how a stale build directory can
# masquerade as a current one.
list(REMOVE_ITEM TESSERA_OPT_FEATURES "")
list(SORT TESSERA_OPT_FEATURES)
string(REPLACE ";" " " _tessera_opt_feature_text "${TESSERA_OPT_FEATURES}")
if(NOT _tessera_opt_feature_text)
  set(_tessera_opt_feature_text "none")
endif()
if(TESSERA_OPT_LEAN_ARTIFACT_DRIVER)
  set(_tessera_opt_profile "lean-artifact-driver")
else()
  set(_tessera_opt_profile "full")
endif()
target_compile_definitions(tessera-opt PRIVATE
  TESSERA_OPT_BUILD_PROFILE="${_tessera_opt_profile}"
  TESSERA_OPT_BUILD_FEATURES="${_tessera_opt_feature_text}"
)
