add_executable(tessera-trace-demo trace_demo.cpp)
target_include_directories(tessera-trace-demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../.. ${CMAKE_CURRENT_BINARY_DIR}/../..)
