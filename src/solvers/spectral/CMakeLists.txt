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
    if(NOT LLVM_VERSION_MAJOR EQUAL 23 OR
       NOT DEFINED MLIR_VERSION_MAJOR OR NOT MLIR_VERSION_MAJOR EQUAL 23)
        message(FATAL_ERROR
            "TesseraSpectral requires LLVM/MLIR 23.x exactly; found LLVM "
            "${LLVM_PACKAGE_VERSION}, MLIR ${MLIR_PACKAGE_VERSION}")
    endif()
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

if(TESSERA_ENABLE_HIP AND NOT CMAKE_HIP_COMPILER)
    enable_language(HIP)
    find_package(hip REQUIRED CONFIG)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# TableGen writes the generated *.inc into the binary dir; the dialect/pass
# headers include them by bare name, so it must be on the include path.
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# ------------------------------------------------------------------
# Dialect + Ops tablegen
# ------------------------------------------------------------------
# mlir_tablegen() reads the input .td from the LLVM_TARGET_DEFINITIONS
# variable — it must be set before each group, otherwise mlir-tblgen is
# handed the include directory as its input file and fails with
# "Is a directory".
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/lib/Dialect/Spectral/SpectralOps.td)
mlir_tablegen(SpectralOps.h.inc       -gen-op-decls      -dialect=tessera_spectral)
mlir_tablegen(SpectralOps.cpp.inc     -gen-op-defs       -dialect=tessera_spectral)
mlir_tablegen(SpectralTypes.h.inc     -gen-typedef-decls -typedefs-dialect=tessera_spectral)
mlir_tablegen(SpectralTypes.cpp.inc   -gen-typedef-defs  -typedefs-dialect=tessera_spectral)
mlir_tablegen(SpectralDialect.h.inc   -gen-dialect-decls -dialect=tessera_spectral)
mlir_tablegen(SpectralDialect.cpp.inc -gen-dialect-defs  -dialect=tessera_spectral)
add_public_tablegen_target(TesseraSpectralTableGen)

# Passes tablegen (stubs for registration)
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/include/tessera/Spectral/SpectralPasses.td)
mlir_tablegen(SpectralPasses.h.inc -gen-pass-decls)
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
    # The package exports one exact architecture identity through
    # ts_spectral_composite_arch_amd().  A fat binary would make that identity
    # ambiguous, so architecture coverage is produced as one build directory
    # and one package per target.
    if(DEFINED CMAKE_HIP_ARCHITECTURES)
        set(_tessera_spectral_hip_arches ${CMAKE_HIP_ARCHITECTURES})
        list(LENGTH _tessera_spectral_hip_arches _tessera_spectral_hip_arch_count)
        if(_tessera_spectral_hip_arch_count GREATER 1)
            message(FATAL_ERROR
                "TesseraSpectralHIP requires exactly one CMAKE_HIP_ARCHITECTURES entry")
        endif()
    endif()
    add_library(TesseraSpectralHIP SHARED
        lib/TargetHooks/AMD/StockhamRadix4.hip
        lib/TargetHooks/AMD/SpectralComposite.hip)
    set_property(TARGET TesseraSpectralHIP PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_target_properties(TesseraSpectralHIP PROPERTIES
        OUTPUT_NAME tessera_spectral_rocm
    )
    if(DEFINED CMAKE_HIP_ARCHITECTURES)
        set_property(TARGET TesseraSpectralHIP PROPERTY
            HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
        target_compile_definitions(TesseraSpectralHIP PRIVATE
            TESSERA_ROCM_SPECTRAL_ARCH="${CMAKE_HIP_ARCHITECTURES}")
    endif()
    install(TARGETS TesseraSpectralHIP LIBRARY DESTINATION lib)
endif()

# ------------------------------------------------------------------
# Tool: ts-spectral-opt
# ------------------------------------------------------------------
add_executable(ts-spectral-opt tools/ts-spectral-opt.cpp)
# MLIROptLib provides MlirOptMain; MLIRFuncDialect is needed so the test IR's
# func.func parses in the driver.
target_link_libraries(ts-spectral-opt PRIVATE
    TesseraSpectral
    MLIRIR MLIRPass MLIRSupport MLIROptLib MLIRFuncDialect)

# Benchmarks
add_executable(ts-spectral-correctness benchmarks/correctness_microbench.cpp)
target_link_libraries(ts-spectral-correctness PRIVATE TesseraSpectral)

# Lit tests
# `llvm-lit` ships with an LLVM *build tree*, not with every distribution —
# Homebrew's llvm keg has none, which silently made this target fail with
# "No such file or directory". Fall back to a standalone `lit` (brew/pip) the
# way the other lit targets in this repo already do.
find_program(TESSERA_LIT_EXECUTABLE
    NAMES llvm-lit lit.py lit
    HINTS ${LLVM_TOOLS_BINARY_DIR}
)
if(TESSERA_LIT_EXECUTABLE)
  set(LLVM_LIT ${TESSERA_LIT_EXECUTABLE})
else()
  set(LLVM_LIT ${LLVM_TOOLS_BINARY_DIR}/llvm-lit)
endif()
add_custom_target(check-spectral
    COMMAND ${CMAKE_COMMAND} -E env "BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            ${LLVM_LIT} -v ${CMAKE_CURRENT_SOURCE_DIR}/test
    DEPENDS ts-spectral-opt
)
