# Sprint G-6 + H-6 (2026-05-11) — Tessera toolchain version pins.
#
# Validates that the CUDA / HIP toolchains present on the build box
# match the versions Tessera's NVIDIA / ROCm backends are pinned to:
#
#   * CUDA Toolkit 13.4 (matches TESSERA_TARGET_CUDA_TOOLKIT
#     in python/tessera/compiler/gpu_target.py)
#   * ROCm 10.0 + HIP 7.15 (matches TESSERA_TARGET_ROCM in
#     python/tessera/compiler/rocm_target.py)
#
# Designed to be hardware-free — only the *toolchain* is required, not
# a GPU.  `nvcc -ptx` and `hipcc -S` produce PTX / SASS / MFMA assembly
# without any device interaction.
#
# Usage from top-level CMakeLists.txt:
#
#     include(cmake/TesseraToolchainPins.cmake)
#     tessera_pin_cuda_toolkit(13.4)        # exits with error if version <13.4
#     tessera_pin_rocm(7.15)                # exits with error if HIP version <7.15 (ROCm 10.0)
#
# Both functions are no-ops when TESSERA_SKIP_TOOLCHAIN_PIN is set, so
# CI configurations that intentionally use older toolchains can still
# build for development.

cmake_minimum_required(VERSION 3.20)

# Pinned versions — kept in sync with the Python source of truth.
set(TESSERA_REQUIRED_CUDA_VERSION   "13.4"      CACHE STRING "Required CUDA Toolkit major.minor version (measured 13.4.59 on The-Super-Bear, 2026-09-15)")
set(TESSERA_REQUIRED_CUDA_DRIVER    "610.88"    CACHE STRING "Required minimum CUDA driver version (measured working driver for 13.4; NVIDIA floor may be lower)")
set(TESSERA_REQUIRED_PTX_ISA        "9.4"       CACHE STRING "Required minimum PTX ISA version (nvcc 13.4.59 emits .version 9.4)")
set(TESSERA_REQUIRED_NCCL_VERSION   "2.22"      CACHE STRING "Required minimum NCCL version (floor; 13.3 bundled 2.30.7, 13.4 bundle not measured)")

# LLVM/MLIR is the one toolchain the whole compiler is built ON, and it was the
# one with no pin here. It is an EXACT match, not a floor, unlike CUDA/ROCm
# below: the others are vendor runtimes where a newer version is normally
# compatible, while MLIR's C++ API changes between patch releases and every
# fleet box must agree for a lit/contract result on one to mean anything on
# another. Measured 2026-09-20: all four boxes on 23.1.1 (Homebrew keg on the
# Mac, apt.llvm.org on Princess-Luna and The-Super-Bear, from-source prefixes
# incl. the assertions build on Tajasarus).
#
# The live hazard is DRIFT UPWARD, not a stale box: apt.llvm.org already offers
# 23.1.2 snapshots on both Ubuntu boxes, so a routine `apt upgrade` moves them
# off the pin silently. Hold the packages there (`sudo apt-mark hold llvm-23
# llvm-23-dev libmlir-23-dev mlir-23-tools clang-23 lld-23`) and raise this pin
# deliberately when the fleet moves together.
set(TESSERA_REQUIRED_LLVM_VERSION   "23.1.1" CACHE STRING "Exact LLVM/MLIR version every fleet box must match (measured on all four, 2026-09-20)")

set(TESSERA_REQUIRED_ROCM_VERSION   "10.0"   CACHE STRING "Required minimum ROCm version (measured 10.0.0 on Princess-Luna + Tajasarus, 2026-09-15)")
set(TESSERA_REQUIRED_HIP_VERSION    "7.15"   CACHE STRING "Required minimum HIP version (measured 7.15.26333)")
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

    # EXACT major.minor, not a floor (changed 2026-09-20).
    #
    # A floor reads as safe and is not: the sm_120 Lion lane lost a week to
    # exactly this. nvcc 13.4 emits PTX 9.4 while driver 610.88 JITs only
    # <= 9.3, so a "newer toolkit is fine" assumption produced opaque rc=3 at
    # every driver-JIT'd launch. A newer vendor toolkit is a DIFFERENT
    # toolchain, and Decision #11 says a measurement is only valid for the code
    # that produced it -- a benchmark taken under 13.4 is not evidence about
    # 13.6. The patch level is deliberately ignored: nvcc 13.4.59 vs 13.4.62 is
    # the same pinned toolkit.
    string(REGEX MATCH "^[0-9]+\\.[0-9]+" _tessera_cuda_found "${CUDAToolkit_VERSION}")
    if(NOT _tessera_cuda_found VERSION_EQUAL ${required_version})
        message(FATAL_ERROR
            "Tessera pins CUDA Toolkit ${required_version} but this box has "
            "${_tessera_cuda_found} (${CUDAToolkit_VERSION}).\n"
            "  Moving the fleet? Run: python scripts/bump_toolchain_pins.py --check\n"
            "  on the box that HAS this toolkit, then --write, then re-measure "
            "every performance row taken under the old pin.\n"
            "  One-off override: -DTESSERA_SKIP_TOOLCHAIN_PIN=ON")
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


function(tessera_pin_llvm required_version)
    if(DEFINED TESSERA_SKIP_TOOLCHAIN_PIN AND TESSERA_SKIP_TOOLCHAIN_PIN)
        message(STATUS "Tessera LLVM pin skipped (TESSERA_SKIP_TOOLCHAIN_PIN=ON)")
        return()
    endif()

    if(NOT DEFINED LLVM_PACKAGE_VERSION)
        message(FATAL_ERROR
            "Tessera pins LLVM/MLIR ${required_version} but no LLVM_PACKAGE_VERSION "
            "is defined -- call this after find_package(LLVM CONFIG).")
    endif()

    # Compare major.minor.patch only: apt.llvm.org appends a snapshot suffix
    # (`23.1.2~++2026...`) that is not part of the version identity.
    string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" _tessera_llvm_found "${LLVM_PACKAGE_VERSION}")

    # EXACT, not a floor. A newer MLIR is not "at least as good": its C++ API
    # moves between patch releases, and two boxes on different patches cannot
    # be compared -- which is the whole reason a fleet result means anything.
    if(NOT _tessera_llvm_found VERSION_EQUAL ${required_version})
        message(FATAL_ERROR
            "Tessera pins LLVM/MLIR ${required_version} but this box has "
            "${_tessera_llvm_found} (${LLVM_PACKAGE_VERSION}) at ${LLVM_DIR}.\n"
            "  If the fleet is moving, raise TESSERA_REQUIRED_LLVM_VERSION and move "
            "EVERY box together -- a lit or contract result is only comparable "
            "across boxes on the same MLIR.\n"
            "  If this box drifted (apt.llvm.org ships 23.1.2 snapshots), pin it back "
            "and hold it: sudo apt-mark hold llvm-23 llvm-23-dev libmlir-23-dev "
            "mlir-23-tools clang-23 lld-23\n"
            "  To override for a one-off: -DTESSERA_SKIP_TOOLCHAIN_PIN=ON")
    endif()

    message(STATUS "Tessera LLVM/MLIR pin satisfied: ${_tessera_llvm_found}")
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

    # EXACT major.minor, for the same reason as CUDA above.
    #
    # NOTE the argument: this function takes the **HIP** version (7.15), not the
    # ROCm version (10.0), while the file defines BOTH
    # TESSERA_REQUIRED_HIP_VERSION and TESSERA_REQUIRED_ROCM_VERSION. Passing
    # the ROCm one -- which is what the function's name invites -- demands
    # hip >= 10.0 and fails on every AMD box in the fleet.
    string(REGEX MATCH "^[0-9]+\\.[0-9]+" _tessera_hip_found "${hip_VERSION}")
    if(NOT _tessera_hip_found VERSION_EQUAL ${required_version})
        message(FATAL_ERROR
            "Tessera pins HIP ${required_version} but this box has "
            "${_tessera_hip_found} (${hip_VERSION}).\n"
            "  Moving the fleet? Run: python scripts/bump_toolchain_pins.py --check\n"
            "  on the box that HAS this toolkit, then --write, then re-measure "
            "every performance row taken under the old pin.\n"
            "  Wiring this in? Pass TESSERA_REQUIRED_HIP_VERSION, not "
            "TESSERA_REQUIRED_ROCM_VERSION.\n"
            "  One-off override: -DTESSERA_SKIP_TOOLCHAIN_PIN=ON")
    endif()

    find_program(TESSERA_HIPCC_EXECUTABLE NAMES hipcc
        HINTS ${HIP_PATH}/bin /opt/rocm/bin
        REQUIRED)
    message(STATUS "Tessera: pinned HIP ${hip_VERSION} "
                   "(hipcc=${TESSERA_HIPCC_EXECUTABLE})")

    set(TESSERA_HIP_VERSION "${hip_VERSION}"   PARENT_SCOPE)
    set(TESSERA_HIPCC_EXECUTABLE "${TESSERA_HIPCC_EXECUTABLE}" PARENT_SCOPE)
endfunction()


# Convenience: register a compile-only target that runs the explicit
# `nvcc -ptx` instruction-probe catalog.
#
# Usage:
#     tessera_add_nvcc_compile_check(
#         NAME tessera_check_nvcc_ptx
#     )
#
function(tessera_add_nvcc_compile_check)
    cmake_parse_arguments(PARSE_ARGV 0 _ARG "" "NAME" "")
    if(NOT _ARG_NAME)
        message(FATAL_ERROR "tessera_add_nvcc_compile_check requires NAME")
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
        COMMENT "Sprint G-8: nvcc -ptx compile-only instruction probes"
        VERBATIM
    )
endfunction()


function(tessera_add_hipcc_compile_check)
    cmake_parse_arguments(PARSE_ARGV 0 _ARG "" "NAME" "")
    if(NOT _ARG_NAME)
        message(FATAL_ERROR "tessera_add_hipcc_compile_check requires NAME")
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
        COMMENT "Sprint H-8: hipcc -S compile-only instruction probes"
        VERBATIM
    )
endfunction()
