# tessera-translate-mlir — MLIR-side translation driver.
#
# Python-side `tessera-translate` (installed as a console script via
# `pyproject.toml`) covers AOT artifact → StableHLO / GGUF / SafeTensors
# format export.  This C++ binary covers MLIR-native translations
# (--mlir-to-llvmir, --import-llvm, --serialize-spirv, ...) over the
# union of MLIR's upstream translations plus the Tessera dialects.

add_executable(tessera-translate-mlir tessera-translate.cpp)
target_link_libraries(tessera-translate-mlir PRIVATE
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
  MLIRTargetLLVMIRExport
  MLIRTargetLLVMIRImport
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRFromLLVMIRTranslationRegistration
  MLIRToLLVMIRTranslationRegistration
  # SPIR-V translation surface.
  MLIRSPIRVDialect
  MLIRSPIRVSerialization
  MLIRSPIRVDeserialization
  MLIRSPIRVTranslateRegistration
)

# Pull in every upstream MLIR dialect.  Registration is conditional on
# what mlir-translate ships against — `registerAllDialects()` /
# `registerAllTranslations()` declared in upstream headers expect the
# linker to resolve every translation registration.
target_link_libraries(tessera-translate-mlir PRIVATE
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIRROCDLDialect
  MLIRArmNeonToLLVMIRTranslation
  MLIRArmSMEToLLVMIRTranslation
  MLIRArmSVEToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIROpenACCToLLVMIRTranslation
  MLIROpenMPToLLVMIRTranslation
  MLIRROCDLToLLVMIRTranslation
  MLIRVCIXToLLVMIRTranslation
)

# Tessera core dialects (Graph IR / Tile IR / Schedule IR).
if(NOT ((TESSERA_BUILD_ROCM_BACKEND AND NOT TESSERA_ENABLE_CUDA) OR
        (TESSERA_BUILD_NVIDIA_BACKEND AND NOT TESSERA_ENABLE_CUDA)))
  target_compile_definitions(tessera-translate-mlir PRIVATE
    TESSERA_HAVE_CORE_TESSERA_IR)
  target_link_libraries(tessera-translate-mlir PRIVATE TesseraIR)
endif()

# Neighbors dialect (Phase 7).
if(TARGET TesseraNeighbors)
  target_compile_definitions(tessera-translate-mlir PRIVATE TESSERA_HAVE_NEIGHBORS)
  target_include_directories(tessera-translate-mlir PRIVATE
    ${CMAKE_SOURCE_DIR}/src/compiler/tessera_neighbors/include
  )
  target_link_libraries(tessera-translate-mlir PRIVATE TesseraNeighbors)
endif()

# TPP dialect.
if(TARGET TesseraTPPInit)
  target_compile_definitions(tessera-translate-mlir PRIVATE TESSERA_HAVE_TPP)
  target_include_directories(tessera-translate-mlir PRIVATE
    ${CMAKE_SOURCE_DIR}/src/solvers/tpp/include
    ${CMAKE_BINARY_DIR}/src/solvers/tpp/include/tpp/Dialect/TPP
  )
  target_link_libraries(tessera-translate-mlir PRIVATE
    TesseraTPPInit
    TesseraTPPDialect
  )
endif()

# Apple Silicon Target IR.
if(TARGET TesseraApple)
  target_compile_definitions(tessera-translate-mlir PRIVATE TESSERA_HAVE_APPLE_BACKEND)
  target_link_libraries(tessera-translate-mlir PRIVATE TesseraApple)
endif()

# Quiet the macOS linker's "ignoring duplicate libraries" warning
# (Xcode 15+).  The duplicates come from the subdir libs pulling in
# the same MLIR libs we list above; the linker correctly drops them.
if(APPLE)
  target_link_options(tessera-translate-mlir PRIVATE
    "LINKER:-no_warn_duplicate_libraries")
endif()
