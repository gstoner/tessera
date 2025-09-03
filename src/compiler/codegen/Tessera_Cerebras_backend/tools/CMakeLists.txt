add_executable(tessera-cerebras-codegen tessera-cerebras-codegen.cpp)
target_include_directories(tessera-cerebras-codegen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(tessera-cerebras-codegen PRIVATE tesseraCerebrasPasses)

add_executable(tessera-cerebras-opt tessera-cerebras-opt.cpp)
target_include_directories(tessera-cerebras-opt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(tessera-cerebras-opt PRIVATE tesseraCerebrasPasses)
