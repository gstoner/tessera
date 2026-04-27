
# ─── Tessera Rubin CPX Backend library ───────────────────────────────────────
# add_mlir_library wires DEPENDS automatically through the tablegen target so
# that generated .h.inc files exist before sources are compiled.

add_mlir_library(TesseraRubinCPX
  # Dialect registration + type/attr/op implementations
  Target/NVRubinCPX/NVRubinCPX.cpp
  Target/NVRubinCPX/CPXTargetIROps.cpp
  Target/NVRubinCPX/CPXTargetIRTypes.cpp

  # Compiler passes
  Transforms/Passes.cpp
  Transforms/KVTransport/LowerKVTransport.cpp
  Transforms/Partition/PartitionLongContext.cpp
  Transforms/Vectorize/NVFP4Vectorize.cpp
  Transforms/Video/VideoIngestFuse.cpp

  DEPENDS
    TesseraRubinCPXTableGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRPassManager
    MLIRSupport
    MLIRFuncDialect
    MLIRTransforms
    MLIRTransformUtils
    MLIRParser
    MLIRRewrite
    LLVMSupport
)

target_include_directories(TesseraRubinCPX PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>
  $<INSTALL_INTERFACE:include>
)
