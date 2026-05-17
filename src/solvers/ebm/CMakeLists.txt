# Tessera EBM (Energy-Based Models) dialect (EBM5).
#
# Mirrors the structure of `src/solvers/spectral/CMakeLists.txt`.

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.20)
    project(TesseraEBM LANGUAGES C CXX)

    find_package(LLVM REQUIRED CONFIG)
    find_package(MLIR REQUIRED CONFIG)
    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)
    include(HandleLLVMOptions)
    include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
    add_definitions(${LLVM_DEFINITIONS})
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/lib/Dialect/EBM/EBMOps.td)

mlir_tablegen(EBMOps.h.inc       -gen-op-decls      -dialect=tessera_ebm)
mlir_tablegen(EBMOps.cpp.inc     -gen-op-defs       -dialect=tessera_ebm)
mlir_tablegen(EBMDialect.h.inc   -gen-dialect-decls -dialect=tessera_ebm)
mlir_tablegen(EBMDialect.cpp.inc -gen-dialect-defs  -dialect=tessera_ebm)
add_public_tablegen_target(TesseraEBMTableGen)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/include/tessera/EBM/EBMPasses.td)
mlir_tablegen(EBMPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(TesseraEBMPassesIncGen)

add_mlir_library(TesseraEBM
    lib/Dialect/EBM/EBMDialect.cpp
    lib/Dialect/EBM/EBMOps.cpp
    lib/Passes/Canonicalize.cpp
    lib/Passes/FuseEnergyGrad.cpp
    lib/Passes/CheckpointInnerLoop.cpp
    lib/Passes/PipelineCandidates.cpp

    DEPENDS
        TesseraEBMTableGen
        TesseraEBMPassesIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRFuncDialect
        MLIRSCFDialect
        MLIRPass
        MLIRTransforms
)

add_executable(ts-ebm-opt tools/ts-ebm-opt.cpp)
target_link_libraries(ts-ebm-opt PRIVATE
    TesseraEBM
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIROptLib
    MLIRSCFDialect
    MLIRFuncDialect
)

set(LLVM_LIT ${LLVM_TOOLS_BINARY_DIR}/llvm-lit)
add_custom_target(check-ebm
    COMMAND ${LLVM_LIT} -v ${CMAKE_CURRENT_SOURCE_DIR}/test
    DEPENDS ts-ebm-opt
)
