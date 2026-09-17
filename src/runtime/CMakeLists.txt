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
    src/exception_heap.cpp
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

# When the HIP backend is compiled (TESSERA_ENABLE_HIP), hip_backend.cpp needs
# the HIP headers (and amdhip64 to link). The top-level build runs
# find_package(hip CONFIG); prefer its imported target (carries include dirs +
# the runtime lib), and fall back to a ROCm-prefix include so a bare
# -DTESSERA_ENABLE_HIP=ON without the package still finds hip/hip_runtime.h.
if(TESSERA_ENABLE_HIP)
    if(DEFINED ENV{ROCM_PATH})
        set(_tessera_rocm_prefix "$ENV{ROCM_PATH}")
    else()
        set(_tessera_rocm_prefix "/opt/rocm")
    endif()
    if(NOT TARGET hip::host)
        find_package(hip QUIET CONFIG)
    endif()
    if(TARGET hip::host)
        target_link_libraries(tessera_runtime PUBLIC hip::host)
    else()
        target_include_directories(tessera_runtime PUBLIC "${_tessera_rocm_prefix}/include")
        message(STATUS "tessera_runtime: HIP enabled, using ROCm include ${_tessera_rocm_prefix}/include (hip::host target not found)")
    endif()
    find_library(TESSERA_RUNTIME_HIP_LIBRARY amdhip64
                 HINTS "${_tessera_rocm_prefix}/lib" "${_tessera_rocm_prefix}/lib64"
                       /opt/rocm/lib /opt/rocm/core/lib)
endif()

# Mirror of the HIP block for CUDA: when cuda_backend.cpp is compiled
# (TESSERA_ENABLE_CUDA), it needs cuda_runtime.h (and cudart to link). Prefer the
# CUDAToolkit imported target (carries include dirs + the runtime lib); fall back
# to a CUDA-prefix include so a bare -DTESSERA_ENABLE_CUDA=ON without the package
# still finds cuda_runtime.h.
if(TESSERA_ENABLE_CUDA)
    if(NOT TARGET CUDA::cudart)
        find_package(CUDAToolkit QUIET)
    endif()
    if(DEFINED ENV{CUDA_PATH})
        set(_tessera_cuda_prefix "$ENV{CUDA_PATH}")
    elseif(DEFINED CUDAToolkit_ROOT)
        set(_tessera_cuda_prefix "${CUDAToolkit_ROOT}")
    else()
        set(_tessera_cuda_prefix "/usr/local/cuda")
    endif()
    if(TARGET CUDA::cudart)
        target_link_libraries(tessera_runtime PUBLIC CUDA::cudart)
    else()
        target_include_directories(tessera_runtime PUBLIC "${_tessera_cuda_prefix}/include")
        message(STATUS "tessera_runtime: CUDA enabled, using include ${_tessera_cuda_prefix}/include (CUDA::cudart target not found)")
    endif()
    find_library(TESSERA_RUNTIME_CUDA_LIBRARY cudart
                 HINTS "${_tessera_cuda_prefix}/lib64" "${_tessera_cuda_prefix}/lib"
                       /usr/local/cuda/lib64)
endif()

# `libtessera_runtime.a` carries the device-backend objects whenever the build
# enabled them, and those objects need `libamdhip64` / `libcudart` at link time.
# A consumer inside this build gets that from the target's usage requirements; a
# consumer *outside* it does not -- and the runtime C-ABI tests compile a harness
# against the archive directly with their own command line. On a host that built
# with HIP enabled they got a page of `undefined reference to hipMalloc` and read
# as a broken ABI (six tests, red on both ROCm boxes and invisible to CI, whose
# runtime is built without either backend).
#
# Publish what such a consumer must add, beside the archive it must add it to.
# One path per line, empty when no device backend is compiled in. The consumer is
# `tests/unit/test_runtime_artifact_abi*.py`; `tests/unit/test_runtime_link_requirements.py`
# holds the file and the CMake that writes it together.
set(_tessera_runtime_consumer_link "")
if(TESSERA_ENABLE_HIP AND TESSERA_RUNTIME_HIP_LIBRARY)
    string(APPEND _tessera_runtime_consumer_link "${TESSERA_RUNTIME_HIP_LIBRARY}\n")
endif()
if(TESSERA_ENABLE_CUDA AND TESSERA_RUNTIME_CUDA_LIBRARY)
    string(APPEND _tessera_runtime_consumer_link "${TESSERA_RUNTIME_CUDA_LIBRARY}\n")
endif()
file(GENERATE
     OUTPUT "$<TARGET_FILE_DIR:tessera_runtime>/tessera_runtime.consumer-link.txt"
     CONTENT "${_tessera_runtime_consumer_link}")

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

    add_executable(tessera_runtime_cpu_native_test tests/test_cpu_native.cpp)
    target_link_libraries(tessera_runtime_cpu_native_test PRIVATE tessera_runtime)
    target_compile_options(tessera_runtime_cpu_native_test PRIVATE -UNDEBUG)
    add_test(NAME tessera_runtime_cpu_native_test COMMAND tessera_runtime_cpu_native_test)
endif()

# ------------------------------------------------------------------
# Runtime ABI smoke — built in both standalone and monorepo builds.
#
# Exercises the public `tsr*` C ABI under whatever sanitizer the
# build was configured with (ASAN red-zones catch handle-lifetime
# corruption, TSAN catches counter races, UBSAN catches arithmetic
# and aliasing UB).  This is the sanitizer-target counterpart to
# `tessera-collective-runtime-smoke` — that one drives the collective
# `tessera_*` surface, this one drives the device runtime `tsr*` ABI.
# Together they cover both halves of the C-side surface.
#
# Drive under each sanitizer via `scripts/run_sanitizers.sh`; the
# unsanitized build runs it as a normal ctest target.
# ------------------------------------------------------------------
if(TARGET tessera_runtime)
    add_executable(tessera-runtime-abi-smoke
        tools/tessera-runtime-abi-smoke/runtime_abi_smoke.cpp)
    target_link_libraries(tessera-runtime-abi-smoke PRIVATE tessera_runtime)
    target_compile_options(tessera-runtime-abi-smoke PRIVATE -UNDEBUG)
    if(TESSERA_BUILD_TESTS)
        enable_testing()
        add_test(NAME tessera_runtime_abi_smoke
                 COMMAND tessera-runtime-abi-smoke)
    endif()
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
