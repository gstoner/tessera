# Tessera: Next-Generation Deep Learning Programming Model
cmake_minimum_required(VERSION 3.20)

project(Tessera 
    VERSION 0.1.0 
    DESCRIPTION "Next-Generation Deep Learning Programming Model"
    LANGUAGES CXX CUDA
)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build options
option(TESSERA_BUILD_TESTS "Build Tessera tests" ON)
option(TESSERA_BUILD_EXAMPLES "Build Tessera examples" ON)
option(TESSERA_BUILD_PYTHON "Build Python bindings" ON)
option(TESSERA_ENABLE_CUDA "Enable CUDA support" ON)

# Find required packages
find_package(LLVM 18 REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

if(TESSERA_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
endif()

# Include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(src/mlir/include)
include_directories(src/runtime/include)

# Add subdirectories
if(TESSERA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(TESSERA_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Configuration summary
message(STATUS "Tessera Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  CUDA support: ${TESSERA_ENABLE_CUDA}")
message(STATUS "  Build tests: ${TESSERA_BUILD_TESTS}")
message(STATUS "  Build examples: ${TESSERA_BUILD_EXAMPLES}")
