cmake_minimum_required(VERSION 3.20)
project(tessera_cli_tools LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(tessera_common STATIC tools/common/manifest.cpp)
target_include_directories(tessera_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tools)

function(tessera_add_tool name)
  add_executable(${name} tools/${name}/main.cpp)
  target_link_libraries(${name} PRIVATE tessera_common)
endfunction()

tessera_add_tool(tessera-opt)
tessera_add_tool(tessera-compile)
tessera_add_tool(tessera-profiler)
tessera_add_tool(tessera-run)
tessera_add_tool(tessera-autotune)
tessera_add_tool(tessera-inspect)

# Unit tests (header-only minimal)
add_executable(tessera_cli_unit tests/unit/smoke.cpp)
target_link_libraries(tessera_cli_unit PRIVATE tessera_common)

enable_testing()
add_test(NAME smoke COMMAND tessera_cli_unit)
