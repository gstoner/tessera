cmake_minimum_required(VERSION 3.20)
project(TesseraMetaliumBackend LANGUAGES CXX)

option(TESSERA_METALIUM_WITH_MLIR "Build MLIR pass plugin for tessera-metalium" OFF)
option(TESSERA_METALIUM_BUILD_DEMO "Build metalium-codegen-demo tool" ON)
option(TESSERA_METALIUM_ENABLE_TESTS "Enable lit tests (requires LLVM_EXTERNAL_LIT or llvm-lit)" OFF)

if (TESSERA_METALIUM_WITH_MLIR)
  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Found MLIR: ${MLIR_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  include(TableGen)
  include(AddMLIR)

  # TableGen ops (uncomment to enable once .td ops are integrated)
  # set(TESSERA_METALIUM_TD ${CMAKE_CURRENT_SOURCE_DIR}/include/Tessera/Target/Metalium/TesseraMetaliumOps.td)
  # mlir_tablegen(TesseraMetaliumOps.h.inc -gen-op-decls ${TESSERA_METALIUM_TD})
  # mlir_tablegen(TesseraMetaliumOps.cpp.inc -gen-op-defs  ${TESSERA_METALIUM_TD})
  # add_public_tablegen_target(TesseraMetaliumOpsIncGen)

  add_library(tessera_metalium_passes SHARED
    lib/Target/Metalium/Passes.cpp
    lib/Target/Metalium/Lowering/TileToMetalium.cpp
    lib/Target/Metalium/TesseraMetaliumDialect.cpp
    lib/Target/Metalium/Util/MemSpace.cpp
  )
  target_include_directories(tessera_metalium_passes PUBLIC
    ${MLIR_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(tessera_metalium_passes PRIVATE
    MLIRTransforms
    MLIRPass
    MLIRIR
    MLIRSupport
    MLIRMemRefDialect
    MLIRArithDialect
  )
  # Pass plugin for mlir-opt: -load-pass-plugin path/to/libtessera_metalium_passes.so
  set_target_properties(tessera_metalium_passes PROPERTIES
    OUTPUT_NAME "tessera_metalium_passes"
  )
endif()

# Demo (header-only shim, no MLIR dependency)
if (TESSERA_METALIUM_BUILD_DEMO)
  add_library(tessera_metalium_codegen
    lib/Target/Metalium/Codegen/MetaliumCodegen.cpp
    lib/Target/Metalium/Codegen/MetaliumBufferPlanner.cpp
  )
  target_include_directories(tessera_metalium_codegen PUBLIC include)

  add_executable(metalium-codegen-demo
    tools/metalium-codegen-demo/main.cpp
  )
  target_link_libraries(metalium-codegen-demo PRIVATE tessera_metalium_codegen)
endif()

# Tests with lit
if (TESSERA_METALIUM_ENABLE_TESTS)
  enable_testing()
  # LLVM_EXTERNAL_LIT can point to llvm-lit or lit.py
  if (NOT LLVM_EXTERNAL_LIT)
    find_program(LLVM_EXTERNAL_LIT llvm-lit lit.py lit)
  endif()
  if (LLVM_EXTERNAL_LIT)
    message(STATUS "Using LLVM_EXTERNAL_LIT: ${LLVM_EXTERNAL_LIT}")
    add_test(NAME metalium-tests
             COMMAND ${LLVM_EXTERNAL_LIT} -sv ${CMAKE_CURRENT_SOURCE_DIR}/test/metalium)
  else()
    message(WARNING "llvm-lit not found; tests disabled")
  endif()
endif()


# Optional mlir-opt style wrapper with statically linked passes/dialect
option(TESSERA_METALIUM_BUILD_OPT "Build tessera-metalium-opt wrapper (requires MLIR)" OFF)
if (TESSERA_METALIUM_WITH_MLIR AND TESSERA_METALIUM_BUILD_OPT)
  add_executable(tessera-metalium-opt
    tools/tessera-metalium-opt/main.cpp
  )
  target_include_directories(tessera-metalium-opt PRIVATE
    ${MLIR_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(tessera-metalium-opt PRIVATE
    tessera_metalium_passes
    MLIROptLib
    MLIRParser
    MLIRTransforms
    MLIRPass
    MLIRIR
    MLIRSupport
    MLIRMemRefDialect
    MLIRArithDialect
    MLIRFuncDialect
  )
endif()
