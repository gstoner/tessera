cmake_minimum_required(VERSION 3.20)
project(tessera_cli_tools VERSION 0.4.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Build settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release if no build type specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Defaulting to Release build")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# LTO for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  include(CheckIPOSupported)
  check_ipo_supported(RESULT lto_supported OUTPUT lto_output)
  if(lto_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  else()
    message(STATUS "LTO not supported: ${lto_output}")
  endif()
endif()

# Strict warnings
if(MSVC)
  add_compile_options(/W4 /WX)
else()
  add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

# ---------------------------------------------------------------------------
# Version header generation
# ---------------------------------------------------------------------------
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TesseraCLIVersion.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/include/TesseraCLIVersion.h"
  @ONLY
)

# ---------------------------------------------------------------------------
# Common library
# ---------------------------------------------------------------------------
add_library(tessera_common STATIC
  tools/common/manifest.cpp
)
target_include_directories(tessera_common PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/tools"
  "${CMAKE_CURRENT_BINARY_DIR}/include"
)

# ---------------------------------------------------------------------------
# Tool builder macro
# ---------------------------------------------------------------------------
function(tessera_add_tool name)
  add_executable(${name} tools/${name}/main.cpp)
  target_link_libraries(${name} PRIVATE tessera_common)
  target_include_directories(${name} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/tools"
    "${CMAKE_CURRENT_BINARY_DIR}/include"
  )
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  install(TARGETS ${name}
    RUNTIME DESTINATION bin
    COMPONENT cli_tools
  )
endfunction()

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
tessera_add_tool(tessera-opt)
tessera_add_tool(tessera-compile)
tessera_add_tool(tessera-profiler)
tessera_add_tool(tessera-run)
tessera_add_tool(tessera-autotune)
tessera_add_tool(tessera-inspect)
tessera_add_tool(tessera-new)

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
add_executable(tessera_cli_unit tests/unit/smoke.cpp)
target_link_libraries(tessera_cli_unit PRIVATE tessera_common)
target_include_directories(tessera_cli_unit PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/tools"
  "${CMAKE_CURRENT_BINARY_DIR}/include"
)

enable_testing()
add_test(NAME tessera_cli_smoke COMMAND tessera_cli_unit)
add_test(NAME tessera_opt_version
  COMMAND tessera-opt --version)
add_test(NAME tessera_compile_version
  COMMAND tessera-compile --version)
add_test(NAME tessera_compile_dry_run
  COMMAND tessera-compile data/sample.mlir
          --platform=cuda --arch=sm_90 --to=ptx
          --dry-run --out-dir "${CMAKE_BINARY_DIR}/test_out/compile")
add_test(NAME tessera_opt_print_pipeline
  COMMAND tessera-opt --alias=graph-to-schedule --print-pipeline)
add_test(NAME tessera_opt_verify_only
  COMMAND tessera-opt "${CMAKE_CURRENT_SOURCE_DIR}/data/sample.mlir" --verify-only)
add_test(NAME tessera_inspect_dry_run
  COMMAND tessera-inspect "${CMAKE_BINARY_DIR}/test_out/compile"
          --format=json --dry-run)
add_test(NAME tessera_new_dry_run
  COMMAND tessera-new my_test_proj --template=gemm --dry-run)
add_test(NAME tessera_autotune_grid_dry_run
  COMMAND tessera-autotune "${CMAKE_CURRENT_SOURCE_DIR}/data/sample.mlir"
          --searcher=grid --trials=8 --dry-run
          --out-dir "${CMAKE_BINARY_DIR}/test_out/tune")
add_test(NAME tessera_run_dry_run
  COMMAND tessera-run "${CMAKE_BINARY_DIR}/test_out/compile"
          --shapes=512x512x512 --dry-run)

# ---------------------------------------------------------------------------
# Lit tests (optional — requires llvm-lit in PATH)
# ---------------------------------------------------------------------------
find_program(LIT_EXECUTABLE lit llvm-lit)
if(LIT_EXECUTABLE)
  message(STATUS "Found lit: ${LIT_EXECUTABLE}")
  add_custom_target(check-tessera-cli-lit
    COMMAND "${LIT_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/lit"
            -sv --path "${CMAKE_BINARY_DIR}/bin"
    DEPENDS tessera-opt tessera-compile tessera-inspect tessera-run
            tessera-autotune tessera-profiler tessera-new
    COMMENT "Running Tessera CLI lit tests"
  )
else()
  message(STATUS "lit not found — skipping check-tessera-cli-lit target")
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(DIRECTORY data/
  DESTINATION share/tessera/cli/data
  COMPONENT cli_tools
)
install(FILES docs/cli/CLI_Design.md
  DESTINATION share/tessera/cli/docs
  COMPONENT cli_tools
)

# CPack packaging
set(CPACK_PACKAGE_NAME "tessera-cli")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Tessera CLI tools suite")
include(CPack)
