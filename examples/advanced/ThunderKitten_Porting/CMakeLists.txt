
cmake_minimum_required(VERSION 3.24)
project(TesseraPorting LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
enable_language(CUDA)

option(TESSERA_USE_WGMMA "Enable WGMMA codepath (requires SM90+ and CUDA 12+)" OFF)
option(TESSERA_USE_TMA   "Enable TMA codepath (requires Hopper/Blackwell and CUDA 12+)" OFF)

if (TESSERA_USE_WGMMA)
  add_compile_definitions(TESSERA_USE_WGMMA=1)
endif()
if (TESSERA_USE_TMA)
  add_compile_definitions(TESSERA_USE_TMA=1)
endif()

# Supported arches (extend for Blackwell locally)
set(CMAKE_CUDA_ARCHITECTURES 80;90)

add_subdirectory(tessera/tools/porting/microbench)


option(TESSERA_REAL_WGMMA "Enable real WGMMA inline PTX hooks (trap until wired)" OFF)
option(TESSERA_REAL_TMA   "Enable real TMA bulk tensor hooks (trap until wired)" OFF)
if (TESSERA_REAL_WGMMA)
  add_compile_definitions(TESSERA_REAL_WGMMA=1)
endif()
if (TESSERA_REAL_TMA)
  add_compile_definitions(TESSERA_REAL_TMA=1)
endif()

add_subdirectory(tessera/tools/porting/rope)

add_subdirectory(tessera/tools/porting/tests)
