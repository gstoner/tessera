//===- Schedule.td (v1.2) -------------------------------------------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"
include "mlir/IR/DialectSpecification.td"

def Tessera_Schedule_Dialect : Dialect {
  let name = "tessera.schedule";
  let cppNamespace = "::tessera::schedule";
}

def WarpOp : Op<"warp", [NoSideEffect]> {
  let summary = "Declare one or more warps with a role";
  let arguments = (ins StrAttr:$role, I32Attr:$count);
  let results = (outs AnyType:$warp_group);
  let assemblyFormat = "$role `,` $count attr-dict";
  let hasVerifier = 1; // Enforces role uniqueness and count>0
}

def PipeOp : Op<"pipe", [NoSideEffect]> {
  let summary = "Connect warp groups in a pipeline with optional multi-buffer depths";
  let arguments = (ins Variadic<AnyType>:$stages, DictionaryAttr:$buffering);
  let assemblyFormat = "$stages attr-dict";
  let hasVerifier = 1; // Validates buffering keys subset {K,V,S,O}
}

def PolicyPersistentOp : Op<"policy", [NoSideEffect]> {
  let summary = "Scheduler policy; supports kind='persistent' and grid defaults";
  let arguments = (ins StrAttr:$kind, I32Attr:$max_cta_per_sm, StrAttr:$tile_queue);
  let results = (outs);
  let assemblyFormat = "$kind `,` $max_cta_per_sm `,` $tile_queue attr-dict";
  let hasVerifier = 1; // persistent -> max_cta_per_sm==1, tile_queue non-empty
}
