
//===- CPXTargetIR.td -------------------------------------------*- tablegen -*-===//
// ODS fragment to extend the (hypothetical) Tessera Target IR with CPX bits.
// This file can be included by the core TargetIR .td to add CPX types/ops.
//---------------------------------------------------------------------------------
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

// Re-export the same content as NVRubinCPX.td if you prefer to include-only model.
// Keeping both to let you choose composition vs separate dialects.
include "tessera/Target/NVRubinCPX/NVRubinCPX.td"
