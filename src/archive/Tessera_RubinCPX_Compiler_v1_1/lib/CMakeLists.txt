
add_library(TesseraCPX
  Target/NVRubinCPX/NVRubinCPX.cpp
  Transforms/Partition/PartitionLongContext.cpp
  Transforms/KVTransport/LowerKVTransport.cpp
  Transforms/Vectorize/NVFP4Vectorize.cpp
  Transforms/Video/VideoIngestFuse.cpp
  Transforms/Passes.cpp
)
add_dependencies(TesseraCPX TesseraCPXIncGen)
target_include_directories(TesseraCPX PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(TesseraCPX PUBLIC MLIRIR MLIRFunc MLIRAsync MLIRLLVMIR MLIRParser MLIRPass MLIRSupport)
install(TARGETS TesseraCPX)
