# Tessera test suite
# Driven by the top-level TESSERA_BUILD_TESTS option.
#
# Test categories:
#   tessera-ir/      — lit-based MLIR dialect regression tests
#   kernel_tests/    — system-level kernel + roofline tests (C++/shell)
#   unit/            — flat Python unit tests
#   performance/     — deterministic compiler performance proxy tests
#   integration/     — end-to-end pipeline tests
#
# Run all tests:   cmake --build . --target check-tessera
# Run Python unit: cmake --build . --target check-tessera-unit
# Run perf proxy:  cmake --build . --target check-tessera-performance
# Run lit only:    cmake --build . --target check-tessera-ir

# ------------------------------------------------------------------
# Lit infrastructure
# ------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# One validated runner for every suite (cmake/TesseraLit.cmake): a candidate is
# selected only if `<lit> --version` actually runs.  The earlier bare
# find_program picked the matched LLVM prefix's `lit` wrapper on Tajasarus,
# which exists but cannot `import lit`, so check-tessera-ir ran zero fixtures
# (SMALL-CORRECTNESS-GAPS-2026-09-27).  This suite is required when tests are
# built, so no working runner is a configure error, not a silent skip.
include(${CMAKE_SOURCE_DIR}/cmake/TesseraLit.cmake)
tessera_lit_command(LLVM_LIT)
if(NOT LLVM_LIT)
    get_property(_tessera_lit_rejected GLOBAL PROPERTY TESSERA_LIT_REJECTED)
    message(FATAL_ERROR
        "No working lit runner for check-tessera-ir. Rejected: "
        "${_tessera_lit_rejected}. Install lit (`pip install lit`) or pass "
        "-DTESSERA_LIT=<path>.")
endif()
set(TESSERA_LIT_CFGS ${CMAKE_CURRENT_SOURCE_DIR})

# Core test dependencies (add more as components mature)
set(TESSERA_TEST_DEPENDS
    tessera-opt
    tessera_runtime
    TesseraIR
    TesseraPasses
    TesseraSR
    TesseraFA4
    TesseraTPPDialect
    TesseraTPPPasses
    TesseraTPPInit
    TesseraPM
)

# ------------------------------------------------------------------
# Lit: Tessera IR dialect regression tests
# ------------------------------------------------------------------
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tessera-ir)
    set(TESSERA_LIT_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tessera-ir)

    add_custom_target(check-tessera-ir
        COMMAND ${CMAKE_COMMAND} -E env
            "PATH=$<TARGET_FILE_DIR:tessera-opt>:${LLVM_TOOLS_BINARY_DIR}:/usr/local/bin:/usr/bin:/bin"
            "TESSERA_OPT=$<TARGET_FILE:tessera-opt>"
            ${LLVM_LIT} -v ${TESSERA_LIT_TEST_DIR}
        DEPENDS ${TESSERA_TEST_DEPENDS}
        COMMENT "Running Tessera IR lit regression tests"
        USES_TERMINAL
    )
endif()

# ------------------------------------------------------------------
# C++ kernel / system tests
# ------------------------------------------------------------------
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/kernel_tests/CMakeLists.txt)
    add_subdirectory(kernel_tests)
endif()

# ------------------------------------------------------------------
# Python unit tests
# ------------------------------------------------------------------
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/unit)
    add_custom_target(check-tessera-unit
        COMMAND ${Python3_EXECUTABLE} -m pytest -q ${CMAKE_CURRENT_SOURCE_DIR}/unit
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running Tessera Python unit tests"
        USES_TERMINAL
    )
endif()

# ------------------------------------------------------------------
# Python performance tests
# ------------------------------------------------------------------
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/performance)
    add_custom_target(check-tessera-performance
        COMMAND ${Python3_EXECUTABLE} -m pytest -q ${CMAKE_CURRENT_SOURCE_DIR}/performance
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running Tessera Python performance tests"
        USES_TERMINAL
    )
endif()

# ------------------------------------------------------------------
# Integration tests
# ------------------------------------------------------------------
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/integration/CMakeLists.txt)
    add_subdirectory(integration)
endif()

# ------------------------------------------------------------------
# Umbrella target: check-tessera
# ------------------------------------------------------------------
set(_check_deps "")
if(TARGET check-tessera-ir)
    list(APPEND _check_deps check-tessera-ir)
endif()
if(TARGET check-tessera-unit)
    list(APPEND _check_deps check-tessera-unit)
endif()

add_custom_target(check-tessera
    DEPENDS ${_check_deps}
    COMMENT "Running all Tessera tests"
)
