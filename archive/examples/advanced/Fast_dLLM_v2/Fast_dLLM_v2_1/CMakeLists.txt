# Tessera Fast-dLLM passes (skeleton)
add_library(TesseraFastDLLMPasses
  lib/Transforms/KVCacheBlockify.cpp
  lib/Transforms/ParallelDecodeExpand.cpp
)
target_include_directories(TesseraFastDLLMPasses PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# Link against MLIR components as needed
