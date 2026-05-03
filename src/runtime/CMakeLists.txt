# Tessera Runtime — C++ execution engine + CPU/CUDA/HIP backends
#
# Can be built standalone or as part of the Tessera monorepo.
# When built as a subdir the top-level TESSERA_ENABLE_CUDA / TESSERA_ENABLE_HIP
# options take precedence over the standalone defaults below.
#
# Exported targets:
#   tessera_runtime   — core runtime library

# ------------------------------------------------------------------
# Standalone-only preamble
# ------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.20)
    project(tessera_runtime LANGUAGES C CXX)

    option(TESSERA_ENABLE_CUDA "Enable CUDA backend" OFF)
    option(TESSERA_ENABLE_HIP  "Enable HIP  backend" OFF)
    option(TESSERA_BUILD_TESTS "Build unit tests"    ON)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------------------------------
# Runtime library
# ------------------------------------------------------------------
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

if(UNIX AND NOT APPLE)
    target_link_libraries(tessera_runtime PUBLIC pthread)
endif()

target_compile_definitions(tessera_runtime PRIVATE
    $<$<BOOL:${TESSERA_ENABLE_CUDA}>:TESSERA_ENABLE_CUDA>
    $<$<BOOL:${TESSERA_ENABLE_HIP}>:TESSERA_ENABLE_HIP>
)

# ------------------------------------------------------------------
# Unit tests (standalone build or if explicitly requested)
# ------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND TESSERA_BUILD_TESTS)
    enable_testing()
    add_executable(tessera_runtime_basic_test tests/test_runtime.cpp)
    target_link_libraries(tessera_runtime_basic_test PRIVATE tessera_runtime)
    target_compile_options(tessera_runtime_basic_test PRIVATE -UNDEBUG)
    add_test(NAME tessera_runtime_basic_test COMMAND tessera_runtime_basic_test)

    add_executable(tessera_runtime_tile_reduction_test tests/test_tile_reduction.cpp)
    target_link_libraries(tessera_runtime_tile_reduction_test PRIVATE tessera_runtime)
    target_compile_options(tessera_runtime_tile_reduction_test PRIVATE -UNDEBUG)
    add_test(NAME tessera_runtime_tile_reduction_test COMMAND tessera_runtime_tile_reduction_test)
endif()

# ------------------------------------------------------------------
# Install rules (active in both standalone and monorepo installs)
# ------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS tessera_runtime
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
