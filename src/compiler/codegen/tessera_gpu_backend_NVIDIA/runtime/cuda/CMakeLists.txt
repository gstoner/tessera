# Shipped NVIDIA mma.sync GEMM runtime symbols (tessera_nvidia_mma_gemm_*).
# The NVIDIA analog of Tessera_ROCM_Backend/runtime/hip. Built only when CUDA is
# available: it NVRTC-compiles the mma.sync kernel for the live device arch at
# load time, so the object itself needs only the ordinary host compiler + the
# CUDA driver (libcuda) + NVRTC at link time (no nvcc device pass). This is the
# shipped symbol the backend_manifest `hardware_verified` contract requires; the
# execute_compare_fixture (tests/unit/test_nvidia_mma_runtime_symbol.py) dlopens
# and numerically validates each dtype against a numpy reference.
if(TESSERA_ENABLE_CUDA)
  if(NOT TARGET CUDA::cudart)
    find_package(CUDAToolkit QUIET)
  endif()
  add_library(tessera_nvidia_gemm SHARED tessera_nvidia_gemm.cpp)

  # Driver API (libcuda) — prefer the toolkit imported target, else the stub.
  if(TARGET CUDA::cuda_driver)
    target_link_libraries(tessera_nvidia_gemm PUBLIC CUDA::cuda_driver)
  else()
    target_link_libraries(tessera_nvidia_gemm PUBLIC cuda)
  endif()

  # NVRTC — imported target, else find the lib by name.
  if(TARGET CUDA::nvrtc)
    target_link_libraries(tessera_nvidia_gemm PUBLIC CUDA::nvrtc)
  else()
    find_library(TESSERA_NVRTC_LIB NAMES nvrtc
                 HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/local/cuda/lib64)
    if(TESSERA_NVRTC_LIB)
      target_link_libraries(tessera_nvidia_gemm PUBLIC ${TESSERA_NVRTC_LIB})
    endif()
  endif()

  # cuda.h / nvrtc.h — imported target carries these; fall back to a CUDA prefix.
  if(CUDAToolkit_FOUND)
    target_include_directories(tessera_nvidia_gemm PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  else()
    if(DEFINED ENV{CUDA_PATH})
      set(_tessera_cuda_prefix "$ENV{CUDA_PATH}")
    else()
      set(_tessera_cuda_prefix "/usr/local/cuda")
    endif()
    target_include_directories(tessera_nvidia_gemm PUBLIC "${_tessera_cuda_prefix}/include")
    message(STATUS "tessera_nvidia_gemm: using CUDA include ${_tessera_cuda_prefix}/include (CUDAToolkit not found)")
  endif()
endif()
