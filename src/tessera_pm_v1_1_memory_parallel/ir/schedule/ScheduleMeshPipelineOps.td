//===- ScheduleMeshPipelineOps.td - Mesh & pipeline ops ------------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"

def ScheduleDialect : Dialect {
  let name = "schedule";
  let cppNamespace = "::tessera::schedule";
}

def Schedule_MeshDefineOp : Op<ScheduleDialect, "mesh.define"> {
  let summary = "Define a logical mesh";
  let arguments = (ins ArrayAttr:$dims, ArrayAttr:$axis_names);
  let hasVerifier = 1;
}

def Schedule_MeshRegionOp : Op<ScheduleDialect, "mesh.region"> {
  let summary = "Enter a mesh axis region";
  let arguments = (ins SymbolRefAttr:$mesh, StrAttr:$axis);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def Schedule_PipelineRegionOp : Op<ScheduleDialect, "pipeline.region"> {
  let summary = "Pipeline-parallel region with schedule";
  let arguments = (ins StrAttr:$schedule, I32Attr:$micro_batches);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def Schedule_StageOp : Op<ScheduleDialect, "stage"> {
  let summary = "A pipeline stage";
  let arguments = (ins ArrayAttr:$devices);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}
