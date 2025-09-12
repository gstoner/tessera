
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
target_link_libraries(TesseraTPPPasses PUBLIC MLIRIR)
