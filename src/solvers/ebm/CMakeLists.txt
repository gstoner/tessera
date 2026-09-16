# Tessera EBM (Energy-Based Models) dialect (EBM5).
#
# Mirrors the structure of `src/solvers/spectral/CMakeLists.txt`.

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.20)
    project(TesseraEBM LANGUAGES C CXX)

    find_package(LLVM REQUIRED CONFIG)
    find_package(MLIR REQUIRED CONFIG)
    if(NOT LLVM_VERSION_MAJOR EQUAL 23 OR
       NOT DEFINED MLIR_VERSION_MAJOR OR NOT MLIR_VERSION_MAJOR EQUAL 23)
        message(FATAL_ERROR
            "TesseraEBM requires LLVM/MLIR 23.x exactly; found LLVM "
            "${LLVM_PACKAGE_VERSION}, MLIR ${MLIR_PACKAGE_VERSION}")
    endif()
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

# The bivector integrator (2026-09-16, M2) projects the gradient and the noise
# onto a grade with the Clifford dialect's own `grade` op rather than growing a
# second grade projection here (Decision #31 — one lowering per boundary). That
# makes TesseraEBM depend on TesseraClifford for that integrator ONLY, so the
# dependency is gated: built without the Clifford backend, `manifold =
# "bivector"` fails closed with a diagnostic naming the missing build option
# instead of silently taking a Euclidean step (Decision #21a).
#
# These are set at DIRECTORY scope, before the library is declared, because
# add_mlir_library compiles the sources in a companion object library whose
# compile options are fixed when it is created — setting them on the
# TesseraEBM target afterwards leaves every .cpp compiled without them, and
# the integrator then silently takes its fail-closed branch. Only this
# directory is affected, and the flag is a feature define, never a host ISA
# flag (Decision #19's build-flag leak).
if(TARGET TesseraClifford)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../clifford/include
                        ${CMAKE_CURRENT_BINARY_DIR}/../clifford)
    add_compile_definitions(TESSERA_EBM_HAVE_CLIFFORD=1)
endif()

add_mlir_library(TesseraEBM
    lib/Dialect/EBM/EBMDialect.cpp
    lib/Dialect/EBM/EBMOps.cpp
    lib/Passes/Canonicalize.cpp
    lib/Passes/FuseEnergyGrad.cpp
    lib/Passes/CheckpointInnerLoop.cpp
    lib/Passes/PipelineCandidates.cpp
    lib/Passes/LowerLangevin.cpp

    DEPENDS
        TesseraEBMTableGen
        TesseraEBMPassesIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRArithDialect
        MLIRFuncDialect
        MLIRLinalgDialect
        MLIRMathDialect
        MLIRSCFDialect
        MLIRTensorDialect
        MLIRPass
        MLIRTransforms
)



if(TARGET TesseraClifford)
    target_link_libraries(TesseraEBM PUBLIC TesseraClifford)
    add_dependencies(TesseraEBM TesseraCliffordTableGen)
    if(TARGET obj.TesseraEBM)
        add_dependencies(obj.TesseraEBM TesseraCliffordTableGen)
    endif()
endif()

add_executable(ts-ebm-opt tools/ts-ebm-opt.cpp)
target_link_libraries(ts-ebm-opt PRIVATE
    TesseraEBM
    MLIRArithDialect
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRTensorDialect
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIROptLib
    MLIRSCFDialect
    MLIRFuncDialect
)

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
add_custom_target(check-ebm
    COMMAND ${CMAKE_COMMAND} -E env "BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            ${LLVM_LIT} -v ${CMAKE_CURRENT_SOURCE_DIR}/test
    DEPENDS ts-ebm-opt
)
