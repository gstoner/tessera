//===- MoeOps.td - Expert-parallel planning & throttling -----------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"

def MoeDialect : Dialect {
  let name = "moe";
  let cppNamespace = "::tessera::moe";
}

def Moe_PlanOp : Op<MoeDialect, "plan"> {
  let summary = "A2A planner and packing policy";
  let arguments = (ins I32Attr:$a2a_bucket, StrAttr:$pack_cast);
  let hasVerifier = 1;
}

def Moe_TokenLimiterCreateOp : Op<MoeDialect, "token_limiter.create"> {
  let arguments = (ins I32Attr:$max_tokens_in_flight, I32Attr:$refill);
  let results = (outs AnyType:$lim);
  let hasVerifier = 1;
}

def Moe_DispatchOp : Op<MoeDialect, "dispatch"> {
  let summary = "Dispatch tokens to experts with limiter enforcement";
  let arguments = (ins AnyType:$input, SymbolRefAttr:$router, AnyType:$limiter);
  let results = (outs AnyType:$output);
  let hasVerifier = 1;
}
