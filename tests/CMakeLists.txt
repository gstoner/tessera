# Tessera test suite
# Driven by the top-level TESSERA_BUILD_TESTS option.
#
# Test categories:
#   tessera-ir/      — lit-based MLIR dialect regression tests
#   kernel_tests/    — system-level kernel + roofline tests (C++/shell)
#   unit/            — C++ unit tests (gtest / catch2)
#   integration/     — end-to-end pipeline tests
#
# Run all tests:   cmake --build . --target check-tessera
# Run lit only:    cmake --build . --target check-tessera-ir

# ------------------------------------------------------------------
# Lit infrastructure
# ------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(LLVM_LIT       "${LLVM_TOOLS_BINARY_DIR}/llvm-lit")
set(TESSERA_LIT_CFGS ${CMAKE_CURRENT_SOURCE_DIR})

# Core test dependencies (add more as components mature)
set(TESSERA_TEST_DEPENDS
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
        COMMAND ${LLVM_LIT} -v ${TESSERA_LIT_TEST_DIR}
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
# C++ unit tests
# ------------------------------------------------------------------
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/unit/CMakeLists.txt)
    add_subdirectory(unit)
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

add_custom_target(check-tessera
    DEPENDS ${_check_deps}
    COMMENT "Running all Tessera tests"
)
