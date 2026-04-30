
cmake_minimum_required(VERSION 3.20)
project(TesseraP3D LANGUAGES CXX)

# Options
option(TESSERA_BUILD_P3D "Build Tessera P3D package" ON)

# MLIR/LLVM discovery
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message(STATUS "Using MLIR at: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddMLIR)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${MLIR_LIBRARY_DIRS})

# ---- Dialect ----
set(P3D_TD_SOURCES
  lib/Dialect/P3D/P3DBase.td
  lib/Dialect/P3D/P3DOps.td
)

mlir_tablegen(P3DAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=P3D -I lib/Dialect/P3D)
mlir_tablegen(P3DAttributes.cpp.inc -gen-attrdef-defs  -attrdefs-dialect=P3D -I lib/Dialect/P3D)
mlir_tablegen(P3DTypes.h.inc      -gen-typedef-decls   -typedefs-dialect=P3D -I lib/Dialect/P3D)
mlir_tablegen(P3DTypes.cpp.inc    -gen-typedef-defs    -typedefs-dialect=P3D -I lib/Dialect/P3D)
mlir_tablegen(P3DOps.h.inc        -gen-op-decls        -I lib/Dialect/P3D)
mlir_tablegen(P3DOps.cpp.inc      -gen-op-defs         -I lib/Dialect/P3D)
add_public_tablegen_target(P3DTableGen)

add_library(TesseraP3DDialect
  lib/Dialect/P3D/P3DDialect.cpp
  lib/Dialect/P3D/P3DOps.cpp
)
add_dependencies(TesseraP3DDialect P3DTableGen)
target_include_directories(TesseraP3DDialect PUBLIC lib)
target_link_libraries(TesseraP3DDialect PUBLIC MLIRIR)

# ---- Transforms ----
add_library(TesseraP3DTransforms
  lib/Transforms/LowerP3D.cpp
  lib/Transforms/AutotuneP3D.cpp
  lib/Passes/Passes.cpp
)
target_link_libraries(TesseraP3DTransforms
  PUBLIC
    TesseraP3DDialect
    MLIRIR
    MLIRPass
    MLIRRewrite
    MLIRTransformUtils
)

# ---- Tools hook (register passes with tessera-opt if present) ----
add_library(TesseraP3DInit OBJECT lib/InitAllP3D.cpp)
target_link_libraries(TesseraP3DInit PUBLIC TesseraP3DTransforms)

# ---- Tests ----
enable_testing()
add_subdirectory(test)
