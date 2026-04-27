add_library(TesseraRubinCPX
  Target/NVRubinCPX/CPXTargetIROps.cpp
  Target/NVRubinCPX/CPXTargetIRTypes.cpp
  Target/NVRubinCPX/NVRubinCPX.cpp
  Transforms/KVTransport/LowerKVTransport.cpp
  Transforms/Partition/PartitionLongContext.cpp
  Transforms/Passes.cpp
  Transforms/Vectorize/NVFP4Vectorize.cpp
  Transforms/Video/VideoIngestFuse.cpp
)

target_include_directories(TesseraRubinCPX PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(TesseraRubinCPX PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
)
