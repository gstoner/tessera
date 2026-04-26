# Tessera Spectral / FFT dialect
#
# Can be built standalone or as part of the Tessera monorepo.
# When built as a subdir, TESSERA_ENABLE_CUDA / TESSERA_ENABLE_HIP come
# from the top-level options; the defaults below only apply standalone.

# ------------------------------------------------------------------
# Standalone-only preamble
# ------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.20)
    project(TesseraSpectral LANGUAGES C CXX)

    find_package(LLVM REQUIRED CONFIG)
    find_package(MLIR REQUIRED CONFIG)
    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)
    include(HandleLLVMOptions)
    include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
    add_definitions(${LLVM_DEFINITIONS})

    option(TESSERA_ENABLE_CUDA "Enable CUDA target hooks" ON)
    option(TESSERA_ENABLE_HIP  "Enable HIP target hooks"  ON)

    message(STATUS "Found LLVM: ${LLVM_PACKAGE_VERSION} at ${LLVM_DIR}")
    message(STATUS "Found MLIR: ${MLIR_PACKAGE_VERSION} at ${MLIR_DIR}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ------------------------------------------------------------------
# Dialect + Ops tablegen
# ------------------------------------------------------------------
set(SPECTRAL_TD ${CMAKE_CURRENT_SOURCE_DIR}/lib/Dialect/Spectral/SpectralOps.td)

mlir_tablegen(SpectralOps.h.inc       -gen-op-decls      -dialect=tessera_spectral DEPENDS ${SPECTRAL_TD})
mlir_tablegen(SpectralOps.cpp.inc     -gen-op-defs       -dialect=tessera_spectral DEPENDS ${SPECTRAL_TD})
mlir_tablegen(SpectralDialect.h.inc   -gen-dialect-decls                           DEPENDS ${SPECTRAL_TD})
mlir_tablegen(SpectralDialect.cpp.inc -gen-dialect-defs                            DEPENDS ${SPECTRAL_TD})
add_public_tablegen_target(TesseraSpectralTableGen)

# Passes tablegen (stubs for registration)
mlir_tablegen(SpectralPasses.h.inc -gen-pass-decls
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/tessera/Spectral/SpectralPasses.td)
add_public_tablegen_target(TesseraSpectralPassesIncGen)

# ------------------------------------------------------------------
# Spectral library
# ------------------------------------------------------------------
add_mlir_library(TesseraSpectral
    lib/Dialect/Spectral/SpectralDialect.cpp
    lib/Dialect/Spectral/SpectralOps.cpp
    lib/Passes/LegalizeSpectral.cpp
    lib/Passes/SpectralMXP.cpp
    lib/Passes/TransposePlan.cpp
    lib/Passes/Autotune.cpp
    lib/Passes/LowerToTargetIR.cpp
    lib/Passes/DistributedFFT.cpp

    DEPENDS
        TesseraSpectralTableGen
        TesseraSpectralPassesIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRFuncDialect
        MLIRPass
        MLIRTransforms
)

# ------------------------------------------------------------------
# Optional CUDA/HIP target hooks
# ------------------------------------------------------------------
if(TESSERA_ENABLE_CUDA)
    add_library(TesseraSpectralCUDA OBJECT lib/TargetHooks/NVIDIA/StockhamRadix4.cu)
    set_property(TARGET TesseraSpectralCUDA PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

if(TESSERA_ENABLE_HIP)
    add_library(TesseraSpectralHIP OBJECT lib/TargetHooks/AMD/StockhamRadix4.hip)
    set_property(TARGET TesseraSpectralHIP PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

# ------------------------------------------------------------------
# Tool: ts-spectral-opt
# ------------------------------------------------------------------
add_executable(ts-spectral-opt tools/ts-spectral-opt.cpp)
target_link_libraries(ts-spectral-opt PRIVATE TesseraSpectral MLIRIR MLIRPass MLIRSupport)

# Benchmarks
add_executable(ts-spectral-correctness benchmarks/correctness_microbench.cpp)
target_link_libraries(ts-spectral-correctness PRIVATE TesseraSpectral)

# Lit tests
set(LLVM_LIT ${LLVM_TOOLS_BINARY_DIR}/llvm-lit)
add_custom_target(check-spectral
    COMMAND ${LLVM_LIT} -v ${CMAKE_CURRENT_SOURCE_DIR}/test
    DEPENDS ts-spectral-opt
)
