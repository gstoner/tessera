cmake_minimum_required(VERSION 3.20)
project(TesseraVerticalSlice LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_testing()

# Build CPU-only runtime shared library
add_library(tessera_runtime SHARED
  runtime/src/api.cpp
)
target_include_directories(tessera_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
)

# Python golden test (uses ctypes to load the shared lib from build dir)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(
  NAME PythonGoldenTest
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_runtime.py
)
# Ensure the test finds the just-built shared library
set_tests_properties(PythonGoldenTest PROPERTIES
  ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR};PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}"
)
