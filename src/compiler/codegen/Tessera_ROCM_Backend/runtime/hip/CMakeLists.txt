cmake_minimum_required(VERSION 3.20)
find_package(hip QUIET)
add_library(tessera_rocm_runtime STATIC loader.cpp launcher.cpp)
target_include_directories(tessera_rocm_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if (hip_FOUND)
  target_link_libraries(tessera_rocm_runtime PUBLIC hip::host)
endif()
add_executable(launch_demo examples/launch_vec_add.cpp)
target_link_libraries(launch_demo PRIVATE tessera_rocm_runtime)

# Shipped ROCm WMMA GEMM runtime symbol (tessera_rocm_wmma_gemm_f16). Built only
# when HIP is available: it HIPRTC-compiles the WMMA kernel at load time, so the
# object itself needs only the ordinary host compiler + the HIP runtime/HIPRTC
# at link time. This is the shipped symbol the backend_manifest
# `hardware_verified` contract requires; the execute_compare_fixture
# (tests/unit/test_rocm_wmma_runtime_symbol.py) dlopens and numerically validates it.
if (hip_FOUND)
  find_library(TESSERA_HIPRTC_LIB NAMES hiprtc HINTS "${ROCM_PATH}/lib" /opt/rocm/lib)
  add_library(tessera_rocm_gemm SHARED tessera_rocm_gemm.cpp)
  target_link_libraries(tessera_rocm_gemm PUBLIC hip::host)
  if (TESSERA_HIPRTC_LIB)
    target_link_libraries(tessera_rocm_gemm PUBLIC ${TESSERA_HIPRTC_LIB})
  endif()

  # Shipped ROCm WMMA flash-attention runtime symbol
  # (tessera_rocm_wmma_flash_attn_{f16,bf16}). Same HIPRTC-at-load model as the
  # GEMM symbol; the second op after matmul to execute natively on ROCm. The
  # execute_compare_fixture (tests/unit/test_rocm_flash_attn_runtime_symbol.py)
  # dlopens and numerically validates it against a CPU attention reference.
  add_library(tessera_rocm_flash_attn SHARED tessera_rocm_flash_attn.cpp)
  target_link_libraries(tessera_rocm_flash_attn PUBLIC hip::host)
  if (TESSERA_HIPRTC_LIB)
    target_link_libraries(tessera_rocm_flash_attn PUBLIC ${TESSERA_HIPRTC_LIB})
  endif()
endif()
