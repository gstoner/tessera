
//===- Passes.td -------------------------------------------------*- tablegen -*-===//
include "mlir/Pass/PassBase.td"

def CreateLegalizeSpaceTime : Pass<"tpp-legalize-space-time"> {
  let summary = "Normalize/annotate TPP space–time constructs";
  let constructor = "createLegalizeSpaceTimePass()";
}

def CreateHaloInfer : Pass<"tpp-halo-infer"> {
  let summary = "Infer halos from stencil/BC usage";
  let constructor = "createHaloInferPass()";
}

def CreateFuseStencilTime : Pass<"tpp-fuse-stencil-time"> {
  let summary = "Fuse stencil ops within time.step regions";
  let constructor = "createFuseStencilTimePass()";
}

def CreateAsyncPrefetch : Pass<"tpp-async-prefetch"> {
  let summary = "Insert async prefetch (cp.async/TMA or LDS streams)";
  let constructor = "createAsyncPrefetchPass()";
}

def CreateDistributeHalo : Pass<"tpp-distribute-halo"> {
  let summary = "Create overlapped halo exchanges";
  let constructor = "createDistributeHaloPass()";
}

def CreateVectorize : Pass<"tpp-vectorize"> {
  let summary = "Vectorize/Tile map to WGMMA/MFMA/AMX";
  let constructor = "createVectorizeTPPPass()";
}

def CreateLowerToTargetIR : Pass<"lower-tpp-to-target-ir"> {
  let summary = "Lower to Tessera Target-IR";
  let constructor = "createLowerTPPToTargetIRPass()";
}
