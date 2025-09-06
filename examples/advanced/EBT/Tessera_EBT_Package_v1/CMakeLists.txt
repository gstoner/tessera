# Minimal CMake hooks for EBT (placeholder)
add_library(tessera_ebt_passes models/ebt/passes/passes.cc)
target_include_directories(tessera_ebt_passes PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/models/ebt/passes)
