cmake_minimum_required(VERSION 3.20)
project(TesseraCerebrasStarter LANGUAGES C CXX)

option(TESSERA_ENABLE_MLIR "Build with MLIR integration (dialect & passes)" ON)

# If you have LLVM+MLIR installed, set LLVM_DIR and MLIR_DIR accordingly.
# Example:
#   -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir

if(TESSERA_ENABLE_MLIR)
  find_package(LLVM QUIET CONFIG)
  find_package(MLIR QUIET CONFIG)
  if(LLVM_FOUND AND MLIR_FOUND)
    message(STATUS "Found LLVM/MLIR: enabling MLIR targets")
    list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})
    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)
    include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
    add_definitions(${LLVM_DEFINITIONS})
    set(HAVE_MLIR TRUE)
  else()
    message(WARNING "LLVM/MLIR not found: disabling MLIR targets")
    set(HAVE_MLIR FALSE)
  endif()
else()
  set(HAVE_MLIR FALSE)
endif()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(tests)
