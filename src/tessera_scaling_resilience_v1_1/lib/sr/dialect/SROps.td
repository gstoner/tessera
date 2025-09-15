//===- SROps.td - Tessera Scaling & Resilience ODS ----------------------===//
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"

def Tessera_SR_Dialect : Dialect {
  let name = "tessera_sr";
  let cppNamespace = "::mlir::tessera::sr";
  let summary = "Scaling & Resilience primitives";
}

def ResTokenType : TypeDef<Tessera_SR_Dialect, "Token"> {
  let mnemonic = "token";
  let summary = "Resilience token for restart/replay ordering";
}

class SR_Op<string mnem, list<Trait> traits = []> :
  Op<Tessera_SR_Dialect, mnem, traits>;

def CheckpointOp : SR_Op<"checkpoint",
  [SingleBlockImplicitTerminator<"mlir::func::ReturnOp">]> {
  let summary = "Marks region recomputable; pass inserts save/recompute";
  let assemblyFormat = "$body attr-dict-with-keyword";
  let regions = (region AnyRegion:$body);
}

def RecomputeHintOp : SR_Op<"recompute_hint", []> {
  let summary = "Advisory hint for recompute placement";
  let assemblyFormat = "attr-dict";
}

def ResilienceRegionOp : SR_Op<"resilience_region", []> {
  let summary = "Critical section with restart/replay semantics";
  let results = (outs ResTokenType:$token);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$body attr-dict `:` type($token)";
}

def ExportManifestOp : SR_Op<"export_manifest", []> {
  let summary = "Emit deployment manifest JSON (mesh, shards, collectives)";
  let assemblyFormat = "attr-dict";
}