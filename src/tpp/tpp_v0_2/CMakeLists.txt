
cmake_minimum_required(VERSION 3.20)
project(TesseraTPP LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(docs)
add_subdirectory(ci)
