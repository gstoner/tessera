//===- tessera_scaling.td - Tessera Scaling & Resilience Dialect ---------===//
// Minimal ODS sketch; extend as needed.
//===----------------------------------------------------------------------===//
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"

def Tessera_Scaling_Dialect : Dialect {
  let name = "tessera_sr";
  let summary = "Scaling and Resilience extensions";
  let cppNamespace = "::mlir::tessera::sr";
}

def ResTokenType : TypeDef<Tessera_Scaling_Dialect, "ResToken"> {
  let mnemonic = "token";
  let summary = "Resilience token used to order replay/restart regions";
}

class SR_Op<string mnem, list<Trait> traits = []> :
  Op<Tessera_Scaling_Dialect, mnem, traits>;

def CheckpointOp : SR_Op<"checkpoint",
  [SingleBlockImplicitTerminator<"mlir::func::ReturnOp">]> {
  let summary = "Marks a region for recomputation";
  let assemblyFormat = "$body attr-dict-with-keyword";
  let regions = (region AnyRegion:$body);
}

def RecomputeHintOp : SR_Op<"recompute_hint", []> {
  let summary = "Advises the recompute insertion pass";
  let assemblyFormat = "attr-dict";
}

def ResilienceRegionOp : SR_Op<"resilience_region", []> {
  let summary = "Critical section with restart/replay semantics";
  let results = (outs ResTokenType:$token);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$body attr-dict `:` type($token)";
}

def ExportManifestOp : SR_Op<"export_manifest", []> {
  let summary = "Emit deployment manifest JSON";
  let assemblyFormat = "attr-dict";
}