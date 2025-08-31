cmake_minimum_required(VERSION 3.20)
project(tessera_runtime LANGUAGES C CXX)

option(TESSERA_ENABLE_CUDA "Enable CUDA backend (stubs in this patch)" OFF)
option(TESSERA_ENABLE_HIP  "Enable HIP backend (stubs in this patch)"  OFF)
option(TESSERA_BUILD_TESTS "Build unit tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(tessera_runtime
  src/tessera_runtime.cpp
  src/scheduler/tile_scheduler.cpp
  src/backend/cpu_backend.cpp
  src/backend/cuda_backend.cpp
  src/backend/hip_backend.cpp
)

target_include_directories(tessera_runtime
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (UNIX AND NOT APPLE)
  target_link_libraries(tessera_runtime PUBLIC pthread)
endif()

target_compile_definitions(tessera_runtime PRIVATE
  $<$<BOOL:${TESSERA_ENABLE_CUDA}>:TESSERA_ENABLE_CUDA>
  $<$<BOOL:${TESSERA_ENABLE_HIP}>:TESSERA_ENABLE_HIP>
)

# -------- Tests --------
if (TESSERA_BUILD_TESTS)
  enable_testing()
  add_executable(tessera_runtime_tests tests/test_runtime.cpp tests/test_tile_reduction.cpp)
  target_link_libraries(tessera_runtime_tests PRIVATE tessera_runtime)
  add_test(NAME tessera_runtime_tests COMMAND tessera_runtime_tests)
endif()

# Install targets
include(GNUInstallDirs)
install(TARGETS tessera_runtime
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
