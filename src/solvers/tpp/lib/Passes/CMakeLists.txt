
add_library(TesseraTPPPasses STATIC
  LegalizeSpaceTime.cpp
  HaloInfer.cpp
  FuseStencilTime.cpp
  AsyncPrefetch.cpp
  DistributeHalo.cpp
  VectorizeTPP.cpp
  LowerTPPToTargetIR.cpp
  PassPipeline.cpp
)
target_link_libraries(TesseraTPPPasses PUBLIC MLIRIR MLIRPass)
target_include_directories(TesseraTPPPasses PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
