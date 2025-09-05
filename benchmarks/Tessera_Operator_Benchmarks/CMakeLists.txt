cmake_minimum_required(VERSION 3.20)
project(tessera_opbench LANGUAGES CXX)

option(OPBENCH_WITH_TESSERA "Enable Tessera runtime hooks" OFF)
option(OPBENCH_WITH_NVTX "Enable NVTX ranges" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(opbench_common
  common/timer.h
  common/nvtx_helpers.h
  common/device_utils.h
)
target_include_directories(opbench_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(OPBENCH_WITH_NVTX)
  find_path(NVTX_INCLUDE_DIR nvToolsExt.h)
  find_library(NVTX_LIBRARY nvToolsExt)
  if(NVTX_INCLUDE_DIR AND NVTX_LIBRARY)
    target_compile_definitions(opbench_common PUBLIC OPBENCH_WITH_NVTX=1)
    target_include_directories(opbench_common PUBLIC ${NVTX_INCLUDE_DIR})
    target_link_libraries(opbench_common PUBLIC ${NVTX_LIBRARY})
  endif()
endif()

add_library(ops
  harness/op_registry.cpp
  ops/matmul/matmul_op.cpp
  ops/conv2d/conv2d_op.cpp
  ops/attention/flash_attention_op.cpp
  ops/reduce/reduce_op.cpp
  ops/elementwise/elementwise_op.cpp
  ops/softmax_layernorm/softmax_layernorm_op.cpp
  ops/transpose_gather/transpose_gather_op.cpp
)
target_include_directories(ops PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ops PUBLIC opbench_common)

add_executable(opbench harness/opbench_main.cpp)
target_link_libraries(opbench PRIVATE ops opbench_common)
