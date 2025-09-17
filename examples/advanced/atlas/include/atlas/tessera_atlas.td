
//===- tessera_atlas.td - Tessera Atlas Dialect ----------------*- tablegen -*-===//
// Lightweight ODS stub for Atlas-style long-term memory ops.
// This is a starter; wire real ODS/CPP generation in your repo.
include "mlir/IR/OpBase.td"

def Atlas_Dialect : Dialect {
  let name = "atlas";
  let cppNamespace = "tessera::atlas";
}

def Atlas_MemoryType : TypeDef<Atlas_Dialect, "Memory"> {
  let summary = "Atlas memory handle";
  let description = "Opaque memory state handle for test-time memorization.";
}

def Atlas_CreateMemoryOp : Op<Atlas_Dialect, "memory.create", [NoSideEffect]> {
  let summary = "Create Atlas memory state";
  let arguments = (ins I64Attr:$capacity, I64Attr:$width, I64Attr:$depth);
  let results   = (outs Atlas_MemoryType:$mem);
  let assemblyFormat = "`(` $capacity `,` $width `,` $depth `)` attr-dict `:` type($mem)";
}

def Atlas_ResetOp : Op<Atlas_Dialect, "memory.reset"> {
  let summary = "Reset/clear memory state";
  let arguments = (ins Atlas_MemoryType:$mem);
  let assemblyFormat = "$mem attr-dict";
}

def Atlas_SetOptimizerOp : Op<Atlas_Dialect, "optimizer.set", [NoSideEffect]> {
  let summary = "Choose internal optimizer for memory update (e.g., muon, gd, momentum)";
  let arguments = (ins Atlas_MemoryType:$mem, StrAttr:$name, F32Attr:$lr, OptionalAttr<F32Attr>:$beta1, OptionalAttr<F32Attr>:$beta2);
  let assemblyFormat = "$mem `,` $name (`lr` `=` $lr) attr-dict";
}

def Atlas_FeatureMapOp : Op<Atlas_Dialect, "feature.map", [NoSideEffect]> {
  let summary = "Apply feature mapping (e.g., poly degree=d, exp, random-feature) to keys/queries";
  let arguments = (ins AnyType:$x, StrAttr:$kind, OptionalAttr<I64Attr>:$degree);
  let results   = (outs AnyType:$y);
  let assemblyFormat = "$x `,` $kind attr-dict `:` type($x) `->` type($y)";
}

def Atlas_UpdateOp : Op<Atlas_Dialect, "memory.update"> {
  let summary = "Omega-rule style sliding-window memory update";
  let description = "Updates mem using window W of (K,V) with chosen optimizer; W can be a compile-time attr or SSA value.";
  let arguments = (ins Atlas_MemoryType:$mem, AnyType:$keys, AnyType:$values, I64Attr:$window);
  let results   = (outs Atlas_MemoryType:$mem_out);
  let assemblyFormat = "$mem `,` $keys `,` $values `,` $window attr-dict";
}

def Atlas_ReadOp : Op<Atlas_Dialect, "memory.read", [NoSideEffect]> {
  let summary = "Read/retrieve value given query and current memory";
  let arguments = (ins Atlas_MemoryType:$mem, AnyType:$query);
  let results   = (outs AnyType:$value);
  let assemblyFormat = "$mem `,` $query attr-dict `:` type($query) `->` type($value)";
}
