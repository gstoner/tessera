
//===- TunableAttrs.td - Tessera Tunable Attributes --------------*- tablegen -*-===//
// Lightweight attribute schema to declare tunables on ops/kernels.
// This is a sketch to be included by the Tessera dialect build.
#ifndef TESSERA_TUNABLE_ATTRS
#define TESSERA_TUNABLE_ATTRS

include "mlir/IR/AttrTypeBase.td"

def TunableSetAttr : I32AttrBase<"TunableSet", "tessera.tunable_set">;

def TunableListAttr : StrAttrBase<"TunableList", "tessera.tunable_list">;

// Example named attributes expected on ops (conceptual):
//  - tessera.tunable_list = ["BLOCK_M","BLOCK_N","BLOCK_K","num_warps","num_stages"]
//  - tessera.BLOCK_M = [64,128,256]
//  - tessera.BLOCK_N = [64,128,256]
//  - tessera.BLOCK_K = [32,64,128]
//  - tessera.num_warps = [4,8,16]
//  - tessera.num_stages = [1,2,3]

#endif // TESSERA_TUNABLE_ATTRS
