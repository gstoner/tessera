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

# Try FFTW
if (USE_FFTW)
  find_package(FFTW3 COMPONENTS FFTW3 FFTW3_THREADS)
  if (FFTW3_FOUND)
    target_compile_definitions(spectral_fft_bench PRIVATE TESSERA_HAVE_FFTW=1)
    target_link_libraries(spectral_fft_bench PRIVATE FFTW3::FFTW3 FFTW3::FFTW3_THREADS)
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
  if (ROCFFT_LIB)
    target_compile_definitions(spectral_fft_bench PRIVATE TESSERA_HAVE_ROCFFT=1)
    target_link_libraries(spectral_fft_bench PRIVATE ${ROCFFT_LIB})
    target_include_directories(spectral_fft_bench PRIVATE /opt/rocm/include)
  endif()
endif()
