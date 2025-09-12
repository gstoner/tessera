//===- Passes.td -----------------------------------------*- tablegen -*-===//
include "mlir/Pass/PassBase.td"

def TesseraHaloInfer : Pass<"tessera-halo-infer", ""> {
  let summary = "Infer per-axis halo widths from Δ-uses";
}

def TesseraStencilLower : Pass<"tessera-stencil-lower", ""> {
  let summary = "Lower stencil.apply to loops/kernels + halo exchanges";
}

def TesseraPipelineOverlap : Pass<"tessera-pipeline-overlap", ""> {
  let summary = "Insert async pack/transport/unpack and tokens for overlap";
}

def TesseraTopologyDynamic : Pass<"tessera-topology-dynamic", ""> {
  let summary = "Manage dynamic topology updates, fences, and replans";
}
