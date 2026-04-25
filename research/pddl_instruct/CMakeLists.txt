
# MERGE-START: CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(tessera_pddl_instruct LANGUAGES CXX)

# TODO: find_package(MLIR REQUIRED CONFIG)
# add_subdirectory for dialects/ if using ODS codegen

add_library(tessera_pddl_passes STATIC
  passes/pddl_instruct_pipeline.cpp
)

target_include_directories(tessera_pddl_passes PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# install(TARGETS tessera_pddl_passes DESTINATION lib)
# MERGE-END: CMakeLists.txt
