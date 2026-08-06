cmake_minimum_required(VERSION 3.16)
project(tessera_spectral_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

option(USE_CUFFT "Build with cuFFT if available" ON)
option(USE_ROCFFT "Build with rocFFT if available" ON)
option(USE_FFTW  "Build with FFTW (CPU) if available" ON)

add_executable(spectral_fft_bench spectral_fft_bench.cpp)

# Find the single-precision FFTW ABI.  complex64 is the common physical dtype
# with rocFFT and Tessera's current packages; linking FFTW3's double ABI would
# make the CPU comparison numerically and operationally different.
if (USE_FFTW)
  find_path(FFTW3_INCLUDE_DIR fftw3.h)
  find_library(FFTW3F_LIB fftw3f)
  if (FFTW3_INCLUDE_DIR AND FFTW3F_LIB)
    target_compile_definitions(spectral_fft_bench PRIVATE TESSERA_HAVE_FFTW=1)
    target_include_directories(spectral_fft_bench PRIVATE ${FFTW3_INCLUDE_DIR})
    target_link_libraries(spectral_fft_bench PRIVATE ${FFTW3F_LIB})
  endif()
endif()

# Try cuFFT
if (USE_CUFFT)
  find_package(CUDAToolkit)
  if (CUDAToolkit_FOUND AND CUDAToolkit_cufft_library)
    target_compile_definitions(spectral_fft_bench PRIVATE TESSERA_HAVE_CUFFT=1)
    target_include_directories(spectral_fft_bench PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(spectral_fft_bench PRIVATE CUDA::cufft)
  endif()
endif()

# Try rocFFT
if (USE_ROCFFT)
  find_library(ROCFFT_LIB rocfft PATHS /opt/rocm/lib /opt/rocm/lib64)
  find_library(HIP_RUNTIME_LIB amdhip64 PATHS /opt/rocm/lib /opt/rocm/lib64)
  find_path(ROCFFT_INCLUDE_DIR rocfft/rocfft.h PATHS /opt/rocm/include)
  find_path(HIP_INCLUDE_DIR hip/hip_runtime_api.h PATHS /opt/rocm/include)
  if (ROCFFT_LIB AND HIP_RUNTIME_LIB AND ROCFFT_INCLUDE_DIR AND HIP_INCLUDE_DIR)
    target_compile_definitions(spectral_fft_bench PRIVATE
      TESSERA_HAVE_ROCFFT=1 __HIP_PLATFORM_AMD__=1)
    target_link_libraries(spectral_fft_bench PRIVATE ${ROCFFT_LIB} ${HIP_RUNTIME_LIB})
    target_include_directories(spectral_fft_bench PRIVATE
      ${ROCFFT_INCLUDE_DIR} ${HIP_INCLUDE_DIR})
  endif()
endif()
