# Sprint G-6 + H-6 (2026-05-11) — Tessera toolchain version pins.
#
# Validates that the CUDA / HIP toolchains present on the build box
# match the versions Tessera's NVIDIA / ROCm backends are pinned to:
#
#   * CUDA Toolkit 13.3 (matches TESSERA_TARGET_CUDA_TOOLKIT
#     in python/tessera/compiler/gpu_target.py)
#   * ROCm 7.2.3 + HIP 7.2.3 (matches TESSERA_TARGET_ROCM in
#     python/tessera/compiler/rocm_target.py)
#
# Designed to be hardware-free — only the *toolchain* is required, not
# a GPU.  `nvcc -ptx` and `hipcc -S` produce PTX / SASS / MFMA assembly
# without any device interaction.
#
# Usage from top-level CMakeLists.txt:
#
#     include(cmake/TesseraToolchainPins.cmake)
#     tessera_pin_cuda_toolkit(13.3)        # exits with error if version <13.3
#     tessera_pin_rocm(7.2.3)               # exits with error if version <7.2.3
#
# Both functions are no-ops when TESSERA_SKIP_TOOLCHAIN_PIN is set, so
# CI configurations that intentionally use older toolchains can still
# build for development.

cmake_minimum_required(VERSION 3.20)

# Pinned versions — kept in sync with the Python source of truth.
set(TESSERA_REQUIRED_CUDA_VERSION   "13.3"      CACHE STRING "Required CUDA Toolkit major.minor version")
set(TESSERA_REQUIRED_CUDA_DRIVER    "610.43.02" CACHE STRING "Required minimum CUDA driver version")
set(TESSERA_REQUIRED_PTX_ISA        "9.3"       CACHE STRING "Required minimum PTX ISA version")
set(TESSERA_REQUIRED_NCCL_VERSION   "2.22"      CACHE STRING "Required minimum NCCL version (floor; 13.3 bundles 2.30.7)")

set(TESSERA_REQUIRED_ROCM_VERSION   "7.2.3"  CACHE STRING "Required minimum ROCm version")
set(TESSERA_REQUIRED_HIP_VERSION    "7.2.3"  CACHE STRING "Required minimum HIP version")
set(TESSERA_REQUIRED_RCCL_VERSION   "2.22"   CACHE STRING "Required minimum RCCL version")
set(TESSERA_REQUIRED_ROCBLAS_VERSION "5.0.0" CACHE STRING "Required minimum rocBLAS version")


function(tessera_pin_cuda_toolkit required_version)
    if(DEFINED TESSERA_SKIP_TOOLCHAIN_PIN AND TESSERA_SKIP_TOOLCHAIN_PIN)
        message(STATUS "Tessera CUDA pin skipped (TESSERA_SKIP_TOOLCHAIN_PIN=ON)")
        return()
    endif()

    find_package(CUDAToolkit ${required_version} REQUIRED)

    # Resolve the actual installed version.
    if(NOT DEFINED CUDAToolkit_VERSION)
        message(FATAL_ERROR
            "Tessera requires CUDA Toolkit ${required_version}+ but no "
            "CUDAToolkit version was reported by find_package.")
    endif()

    # CUDAToolkit_VERSION has format like "13.3.0"
    if(CUDAToolkit_VERSION VERSION_LESS ${required_version})
        message(FATAL_ERROR
            "Tessera requires CUDA Toolkit >= ${required_version} (matching the "
            "TESSERA_TARGET_CUDA_TOOLKIT pin in gpu_target.py), but found "
            "${CUDAToolkit_VERSION}.  Set -DTESSERA_SKIP_TOOLCHAIN_PIN=ON to "
            "override (development only).")
    endif()

    # Locate nvcc explicitly so the compile-only validator can find it.
    find_program(TESSERA_NVCC_EXECUTABLE NAMES nvcc
        HINTS ${CUDAToolkit_BIN_DIR}
        REQUIRED)
    message(STATUS "Tessera: pinned CUDA Toolkit ${CUDAToolkit_VERSION} "
                   "(nvcc=${TESSERA_NVCC_EXECUTABLE})")

    # Export for downstream targets.
    set(TESSERA_CUDA_TOOLKIT_VERSION "${CUDAToolkit_VERSION}" PARENT_SCOPE)
    set(TESSERA_NVCC_EXECUTABLE "${TESSERA_NVCC_EXECUTABLE}"   PARENT_SCOPE)
endfunction()


function(tessera_pin_rocm required_version)
    if(DEFINED TESSERA_SKIP_TOOLCHAIN_PIN AND TESSERA_SKIP_TOOLCHAIN_PIN)
        message(STATUS "Tessera ROCm pin skipped (TESSERA_SKIP_TOOLCHAIN_PIN=ON)")
        return()
    endif()

    # hip exposes hip_VERSION; full ROCm version is sometimes in a separate
    # rocm-cmake config.
    find_package(hip ${required_version} REQUIRED CONFIG)

    if(NOT DEFINED hip_VERSION)
        message(FATAL_ERROR
            "Tessera requires HIP ${required_version}+ but no hip_VERSION "
            "was reported by find_package(hip).")
    endif()

    if(hip_VERSION VERSION_LESS ${required_version})
        message(FATAL_ERROR
            "Tessera requires HIP >= ${required_version} (matching the "
            "TESSERA_TARGET_HIP pin in rocm_target.py), but found ${hip_VERSION}. "
            "Set -DTESSERA_SKIP_TOOLCHAIN_PIN=ON to override.")
    endif()

    find_program(TESSERA_HIPCC_EXECUTABLE NAMES hipcc
        HINTS ${HIP_PATH}/bin /opt/rocm/bin
        REQUIRED)
    message(STATUS "Tessera: pinned HIP ${hip_VERSION} "
                   "(hipcc=${TESSERA_HIPCC_EXECUTABLE})")

    set(TESSERA_HIP_VERSION "${hip_VERSION}"   PARENT_SCOPE)
    set(TESSERA_HIPCC_EXECUTABLE "${TESSERA_HIPCC_EXECUTABLE}" PARENT_SCOPE)
endfunction()


# Convenience: register a compile-only target that runs the
# `nvcc -ptx` validator over every G-4 lit fixture.
#
# Usage:
#     tessera_add_nvcc_compile_check(
#         NAME tessera_check_nvcc_ptx
#         FIXTURES ${CMAKE_SOURCE_DIR}/tests/tessera-ir/phase3/cuda13
#     )
#
function(tessera_add_nvcc_compile_check)
    cmake_parse_arguments(PARSE_ARGV 0 _ARG "" "NAME;FIXTURES" "")
    if(NOT _ARG_NAME OR NOT _ARG_FIXTURES)
        message(FATAL_ERROR "tessera_add_nvcc_compile_check requires NAME and FIXTURES")
    endif()
    if(NOT TESSERA_NVCC_EXECUTABLE)
        message(STATUS "Skipping ${_ARG_NAME}: nvcc not pinned (call tessera_pin_cuda_toolkit first)")
        return()
    endif()
    add_custom_target(${_ARG_NAME}
        COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${CMAKE_SOURCE_DIR}/python:${CMAKE_SOURCE_DIR}"
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/scripts/validate_nvcc_compile.py
                --nvcc ${TESSERA_NVCC_EXECUTABLE}
                --fixtures ${_ARG_FIXTURES}
        COMMENT "Sprint G-8: nvcc -ptx compile-only check over G-4 fixtures"
        VERBATIM
    )
endfunction()


function(tessera_add_hipcc_compile_check)
    cmake_parse_arguments(PARSE_ARGV 0 _ARG "" "NAME;FIXTURES" "")
    if(NOT _ARG_NAME OR NOT _ARG_FIXTURES)
        message(FATAL_ERROR "tessera_add_hipcc_compile_check requires NAME and FIXTURES")
    endif()
    if(NOT TESSERA_HIPCC_EXECUTABLE)
        message(STATUS "Skipping ${_ARG_NAME}: hipcc not pinned (call tessera_pin_rocm first)")
        return()
    endif()
    add_custom_target(${_ARG_NAME}
        COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${CMAKE_SOURCE_DIR}/python:${CMAKE_SOURCE_DIR}"
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/scripts/validate_hipcc_compile.py
                --hipcc ${TESSERA_HIPCC_EXECUTABLE}
                --fixtures ${_ARG_FIXTURES}
        COMMENT "Sprint H-8: hipcc -S compile-only check over H-4 fixtures"
        VERBATIM
    )
endfunction()
