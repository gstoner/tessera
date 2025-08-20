cmake_minimum_required(VERSION 3.14)
project(tessera_runtime_mock C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(tessera SHARED
    tessera_runtime_mock.c
)
target_include_directories(tessera PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(tessera_demo demo.c)
target_include_directories(tessera_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tessera_demo PRIVATE tessera)
