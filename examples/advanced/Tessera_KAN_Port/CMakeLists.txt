cmake_minimum_required(VERSION 3.22)
project(TesseraKAN LANGUAGES CXX)

option(TESSERA_BUILD_KAN "Build Tessera KAN components" ON)

if (TESSERA_BUILD_KAN)
  add_library(tessera_kan_ref STATIC src/runtime/kernels/kan_bspline_ref.cpp)
  target_include_directories(tessera_kan_ref PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
  # TODO: link to Tessera runtime if needed
endif()