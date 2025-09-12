//===- TPPAttrs.td ---------------------------------------*- tablegen -*-===//
include "mlir/IR/AttrTypeBase.td"
include "TPP.td"

def TPP_UnitsAttr : StrAttr<"Units", "tpp.units">;
def TPP_BCAttr    : StrAttr<"BC",    "tpp.bc">;
