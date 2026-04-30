cmake_minimum_required(VERSION 3.20)
project(tessera_ebt_v23 LANGUAGES CXX)
add_library(tessera_ebt_passes STATIC
  models/ebt/passes/register_pipelines.cc
  models/ebt/passes/grad_path_select.cc)
target_include_directories(tessera_ebt_passes PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(tessera-ebt-opt tools/tessera-ebt-opt/main.cpp)
target_include_directories(tessera-ebt-opt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tessera-ebt-opt PRIVATE tessera_ebt_passes)
