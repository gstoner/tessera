cmake_minimum_required(VERSION 3.22)
project(TesseraVerticalSliceCUDAStreams LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
enable_testing()

option(TESSERA_ENABLE_CUDA "Build CUDA kernels" ON)

# CPU runtime object library
add_library(tessera_runtime_cpu OBJECT
  runtime/src/api_cpu.cpp
)
target_include_directories(tessera_runtime_cpu PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
)

# CUDA runtime object library
if(TESSERA_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(tessera_runtime_cuda OBJECT
    runtime/src/cuda/kernels.cu
    runtime/src/cuda/gemm_wmma.cu
    runtime/src/api_cuda.cpp
  )
  target_include_directories(tessera_runtime_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
  )
  target_link_libraries(tessera_runtime_cuda PUBLIC CUDA::cudart)
  set_target_properties(tessera_runtime_cuda PROPERTIES
    CUDA_ARCHITECTURES "80;86;89;90"  # adjust for your env
  )
endif()

# Shared lib combining CPU and optional CUDA objects
add_library(tessera_runtime SHARED
  $<TARGET_OBJECTS:tessera_runtime_cpu>
  $<$<BOOL:${TESSERA_ENABLE_CUDA}>:$<TARGET_OBJECTS:tessera_runtime_cuda>>
)
target_include_directories(tessera_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
)
target_link_libraries(tessera_runtime PUBLIC
  $<$<BOOL:${TESSERA_ENABLE_CUDA}>:CUDA::cudart>
)

# Tests wired into CTest
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME PythonGoldenTestCPU
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_runtime.py --device cpu
)
set_tests_properties(PythonGoldenTestCPU PROPERTIES
  ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR};PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
)

if(TESSERA_ENABLE_CUDA)
  add_test(NAME PythonGoldenTestCUDA
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_runtime.py --device cuda
  )
  set_tests_properties(PythonGoldenTestCUDA PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR};PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
  )
endif()
