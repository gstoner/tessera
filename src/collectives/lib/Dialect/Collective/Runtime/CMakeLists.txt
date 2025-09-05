add_library(TesseraCollectiveRuntime
  Execution.cpp
)
target_include_directories(TesseraCollectiveRuntime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../.. ${CMAKE_CURRENT_BINARY_DIR}/../../..)
