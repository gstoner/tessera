# Tessera Clifford / Geometric Algebra dialect (GA7).
#
# Mirrors the structure of `src/solvers/spectral/CMakeLists.txt`:
#   - tablegen the dialect + ops decls
#   - tablegen pass declarations
#   - link a TesseraClifford static library
#   - build the ts-clifford-opt driver
#   - register a `check-clifford` lit target
#
# This subdirectory is added from `src/solvers/CMakeLists.txt` when the
# TESSERA_BUILD_CLIFFORD_BACKEND option is ON (off by default in v1 — the
# Python surface ships without requiring the C++ dialect to be built).

# ------------------------------------------------------------------
# Standalone-only preamble (same pattern as the spectral solver).
# ------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.20)
    project(TesseraClifford LANGUAGES C CXX)

    find_package(LLVM REQUIRED CONFIG)
    find_package(MLIR REQUIRED CONFIG)
    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)
    include(HandleLLVMOptions)
    include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
    add_definitions(${LLVM_DEFINITIONS})

    message(STATUS "Found LLVM: ${LLVM_PACKAGE_VERSION} at ${LLVM_DIR}")
    message(STATUS "Found MLIR: ${MLIR_PACKAGE_VERSION} at ${MLIR_DIR}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ------------------------------------------------------------------
# Dialect + Ops tablegen.
# ------------------------------------------------------------------
set(CLIFFORD_TD ${CMAKE_CURRENT_SOURCE_DIR}/lib/Dialect/Clifford/CliffordOps.td)

mlir_tablegen(CliffordOps.h.inc       -gen-op-decls      -dialect=tessera_clifford DEPENDS ${CLIFFORD_TD})
mlir_tablegen(CliffordOps.cpp.inc     -gen-op-defs       -dialect=tessera_clifford DEPENDS ${CLIFFORD_TD})
mlir_tablegen(CliffordDialect.h.inc   -gen-dialect-decls                            DEPENDS ${CLIFFORD_TD})
mlir_tablegen(CliffordDialect.cpp.inc -gen-dialect-defs                             DEPENDS ${CLIFFORD_TD})
add_public_tablegen_target(TesseraCliffordTableGen)

# Passes tablegen (registration headers).
mlir_tablegen(CliffordPasses.h.inc -gen-pass-decls
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/tessera/Clifford/CliffordPasses.td)
add_public_tablegen_target(TesseraCliffordPassesIncGen)

# ------------------------------------------------------------------
# Clifford library.
# ------------------------------------------------------------------
add_mlir_library(TesseraClifford
    lib/Dialect/Clifford/CliffordDialect.cpp
    lib/Dialect/Clifford/CliffordOps.cpp
    lib/Passes/AnnotateAlgebra.cpp
    lib/Passes/ExpandProductTable.cpp
    lib/Passes/GradeFusion.cpp
    lib/Passes/RotorSandwichFold.cpp

    DEPENDS
        TesseraCliffordTableGen
        TesseraCliffordPassesIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRFuncDialect
        MLIRArithDialect
        MLIRTensorDialect
        MLIRPass
        MLIRTransforms
)

# ------------------------------------------------------------------
# Tool: ts-clifford-opt.
# ------------------------------------------------------------------
add_executable(ts-clifford-opt tools/ts-clifford-opt.cpp)
target_link_libraries(ts-clifford-opt PRIVATE TesseraClifford MLIRIR MLIRPass MLIRSupport)

# ------------------------------------------------------------------
# Lit tests.
# ------------------------------------------------------------------
set(LLVM_LIT ${LLVM_TOOLS_BINARY_DIR}/llvm-lit)
add_custom_target(check-clifford
    COMMAND ${LLVM_LIT} -v ${CMAKE_CURRENT_SOURCE_DIR}/test
    DEPENDS ts-clifford-opt
)
