# tools/profiler/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(tessera_profiler LANGUAGES C CXX)

option(TPROF_WITH_NVTX "Enable NVTX shims" OFF)
option(TPROF_WITH_CUPTI "Enable CUPTI shims" OFF)
option(TPROF_WITH_MLIR "Build MLIR pass (stub)" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

if (TPROF_WITH_NVTX)
  find_package(NVTX QUIET)
  if (NVTX_FOUND)
    add_compile_definitions(TPROF_WITH_NVTX=1)
  else()
    message(WARNING "NVTX requested but not found; building without NVTX.")
  endif()
endif()

if (TPROF_WITH_CUPTI)
  find_package(CUPTI QUIET)
  if (CUPTI_FOUND)
    add_compile_definitions(TPROF_WITH_CUPTI=1)
  else()
    message(WARNING "CUPTI requested but not found; building without CUPTI.")
  endif()
endif()

add_library(tprof_runtime
  src/runtime/tprof_runtime.cpp
  src/runtime/nvtx_shim.cpp
  src/runtime/cupti_shim.cpp
  src/exporters/chrome_trace_exporter.cpp
  src/exporters/perfetto_exporter.cpp
)
target_include_directories(tprof_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if (NVTX_FOUND)
  target_link_libraries(tprof_runtime PRIVATE NVTX::nvtx3)
endif()
if (CUPTI_FOUND)
  target_link_libraries(tprof_runtime PRIVATE CUPTI::cupti)
endif()

add_library(tprof_device INTERFACE)
target_include_directories(tprof_device INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(tprof cli/tprof.cpp)
target_link_libraries(tprof PRIVATE tprof_runtime)

if (TPROF_WITH_MLIR)
  add_library(tprof_mlir STATIC passes/tprof_markers_stub.cpp)
  target_include_directories(tprof_mlir PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()
