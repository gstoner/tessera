add_executable(tessera-collective-runtime-smoke runtime_smoke.cpp)
target_link_libraries(tessera-collective-runtime-smoke PRIVATE TesseraCollectiveRuntime)
target_include_directories(tessera-collective-runtime-smoke PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
  ${CMAKE_CURRENT_BINARY_DIR}/../..
)
