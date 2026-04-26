//===- Schedule.td - Tessera Schedule dialect additions ----------*- tablegen -*-===//
// Warp specialization and persistent policies.
include "mlir/IR/OpBase.td"
include "mlir/IR/DialectSpecification.td"

def Tessera_Schedule_Dialect : Dialect {
  let name = "tessera.schedule";
  let cppNamespace = "::tessera::schedule";
  let summary = "Scheduling constructs for pipelines, warps, and policies";
  let description = [{
    Warp specialization roles and pipeline edges with buffering depths.
    Also declares a 'persistent' tile scheduler policy.
  }];
}

def WarpRoleAttr : StrAttr<"WarpRoleAttr"> {
  let summary = "Role string for a warp (e.g., load, mma, softmax, correction, epilogue)";
}

def WarpOp : Op<"warp", [NoSideEffect]> {
  let summary = "Declare one or more warps with a role";
  let arguments = (ins StrAttr:$role, I32Attr:$count);
  let results = (outs AnyType:$warp_group);
  let assemblyFormat = "$role `,` $count attr-dict";
  let hasVerifier = 1;
}

def PipeOp : Op<"pipe", [NoSideEffect]> {
  let summary = "Connect warp groups in a pipeline with optional multi-buffer depths";
  let arguments = (ins Variadic<AnyType>:$stages, DictionaryAttr:$buffering);
  let assemblyFormat = "$stages attr-dict";
  let hasVerifier = 1;
}

def PolicyPersistentOp : Op<"policy", [NoSideEffect]> {
  let summary = "Scheduler policy; supports kind='persistent' and grid defaults";
  let arguments = (ins StrAttr:$kind, I32Attr:$max_cta_per_sm, StrAttr:$tile_queue);
  let results = (outs);
  let assemblyFormat = "$kind `,` $max_cta_per_sm `,` $tile_queue attr-dict";
  let hasVerifier = 1;
}
