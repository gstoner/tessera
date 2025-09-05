add_executable(tessera-exec-demo exec_demo.cpp)
target_link_libraries(tessera-exec-demo PRIVATE TesseraCollectiveRuntime)
target_include_directories(tessera-exec-demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../.. ${CMAKE_CURRENT_BINARY_DIR}/../..)
