# TPP (Tile Processing Primitives) dialect
# Canonical component: promoted from tpp/tpp_v0_2
# (v0_2 adds TPPAttrs.td, TPPTypes.td, CI workflow over v0_1)
#
# Can be built standalone or as part of the Tessera monorepo.

# ------------------------------------------------------------------
# Standalone-only preamble
# ------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.20)
    project(TesseraTPP LANGUAGES CXX C)

    find_package(LLVM REQUIRED CONFIG)
    find_package(MLIR REQUIRED CONFIG)
    if(NOT LLVM_VERSION_MAJOR EQUAL 23 OR
       NOT DEFINED MLIR_VERSION_MAJOR OR NOT MLIR_VERSION_MAJOR EQUAL 23)
        message(FATAL_ERROR
            "TesseraTPP requires LLVM/MLIR 23.x exactly; found LLVM "
            "${LLVM_PACKAGE_VERSION}, MLIR ${MLIR_PACKAGE_VERSION}")
    endif()
    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)
    include(HandleLLVMOptions)
    include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
    link_directories(${LLVM_LIBRARY_DIRS})
    add_definitions(${LLVM_DEFINITIONS})
endif()

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(benchmarks)
add_subdirectory(test)
add_subdirectory(docs)
add_subdirectory(ci)
