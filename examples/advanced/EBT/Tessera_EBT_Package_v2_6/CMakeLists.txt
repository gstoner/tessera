cmake_minimum_required(VERSION 3.20)
project(tessera_ebt_v26 LANGUAGES CXX)

option(SMOKE_WITH_CUDA "Build CUDA smoke stub" OFF)
option(SMOKE_WITH_HIP "Build HIP smoke stub" OFF)

add_executable(run_smoke tools/smoke/run_smoke.cpp)
target_compile_features(run_smoke PRIVATE cxx_std_17)

if (SMOKE_WITH_CUDA)
  enable_language(CUDA)
  add_library(smoke_cuda OBJECT tools/smoke/backends/cuda_stub.cu)
  target_compile_definitions(run_smoke PRIVATE SMOKE_WITH_CUDA=1)
  target_link_libraries(run_smoke PRIVATE smoke_cuda)
endif()

if (SMOKE_WITH_HIP)
  add_library(smoke_hip OBJECT tools/smoke/backends/hip_stub.cpp)
  target_compile_definitions(run_smoke PRIVATE SMOKE_WITH_HIP=1)
  target_link_libraries(run_smoke PRIVATE smoke_hip)
endif()
