cmake_minimum_required(VERSION 3.20)

option(TESSERA_APPLE_WITH_MLIR "Build Apple Silicon MLIR target dialect" ON)

if(NOT TESSERA_APPLE_WITH_MLIR)
  message(STATUS "Tessera Apple backend: TESSERA_APPLE_WITH_MLIR=OFF, skipping")
  return()
endif()

# When this directory is added by the top-level Tessera build the parent
# project has already configured MLIR. When built standalone, find it now.
if(NOT MLIR_FOUND)
  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Found MLIR: ${MLIR_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  include(TableGen)
  include(AddMLIR)
endif()

# ------------------------------------------------------------------
# Tablegen: dialect + op declarations / definitions.
# ------------------------------------------------------------------
set(LLVM_TARGET_DEFINITIONS include/Tessera/Target/Apple/TesseraAppleOps.td)
mlir_tablegen(TesseraAppleOps.h.inc -gen-op-decls)
mlir_tablegen(TesseraAppleOps.cpp.inc -gen-op-defs)
mlir_tablegen(TesseraAppleDialect.h.inc -gen-dialect-decls -dialect=tessera_apple)
mlir_tablegen(TesseraAppleDialect.cpp.inc -gen-dialect-defs -dialect=tessera_apple)
add_public_tablegen_target(TesseraAppleOpsIncGen)

# ------------------------------------------------------------------
# TesseraApple: dialect + Tile->Apple lowering passes + pipeline registrations.
# ------------------------------------------------------------------
add_library(TesseraApple STATIC
  lib/Target/Apple/TesseraAppleDialect.cpp
  lib/Target/Apple/Passes.cpp
  lib/Target/Apple/Lowering/TileToApple.cpp
)

# mlir_tablegen emits .h.inc / .cpp.inc into CMAKE_CURRENT_BINARY_DIR/Tessera/Target/Apple/
# but with the layout above they land in CMAKE_CURRENT_BINARY_DIR. Expose both
# so #include "Tessera/Target/Apple/TesseraAppleOps.h.inc" resolves cleanly.
target_include_directories(TesseraApple
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)

add_dependencies(TesseraApple TesseraAppleOpsIncGen)

target_link_libraries(TesseraApple
  PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
  PRIVATE
    MLIRFuncDialect
    MLIRTransforms
)

# Some MLIR build flavors put the tablegen output under
# CMAKE_CURRENT_BINARY_DIR/include/Tessera/Target/Apple. Also accept the flat
# layout (CMAKE_CURRENT_BINARY_DIR/<file>.inc) by symlinking on demand.
set(_TBG_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
foreach(_inc TesseraAppleOps.h.inc TesseraAppleOps.cpp.inc
             TesseraAppleDialect.h.inc TesseraAppleDialect.cpp.inc)
  set(_src ${_TBG_OUT_DIR}/${_inc})
  set(_dst ${CMAKE_CURRENT_BINARY_DIR}/include/Tessera/Target/Apple/${_inc})
  add_custom_command(
    OUTPUT ${_dst}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_CURRENT_BINARY_DIR}/include/Tessera/Target/Apple
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_src} ${_dst}
    DEPENDS ${_src} TesseraAppleOpsIncGen
  )
  list(APPEND _STAGED_INC ${_dst})
endforeach()
add_custom_target(TesseraAppleStagedIncs DEPENDS ${_STAGED_INC})
add_dependencies(TesseraApple TesseraAppleStagedIncs)
