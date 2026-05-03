//===- TileMemoryOps.td - Tile IR memory ops (v1.1) ---------------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"

def TileDialect : Dialect {
  let name = "tile";
  let cppNamespace = "::tessera::tile";
}

def Tile_AllocSharedOp : Op<TileDialect, "alloc_shared", [MemRefsNormalizable]> {
  let summary = "Allocate shared/LDS memory with optional swizzle and bank padding";
  let arguments = (ins AnyMemRef:$result);
  let assemblyFormat = "$result attr-dict";
  let hasVerifier = 1;
}

def Tile_AsyncCopyOp : Op<TileDialect, "async_copy"> {
  let summary = "Asynchronous copy between memory spaces with stage index";
  let arguments = (ins Variadic<AnyType>:$operands,
                   OptionalAttr<I32Attr>:$stage,
                   OptionalAttr<I32Attr>:$vector,
                   OptionalAttr<I64Attr>:$tile_rows,
                   OptionalAttr<I64Attr>:$tile_cols);
  let results = (outs AnyType:$staged);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def Tile_WaitAsyncOp : Op<TileDialect, "wait_async"> {
  let summary = "Wait for a completed async stage";
  let arguments = (ins OptionalAttr<I32Attr>:$stage);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Tile_MBarrierAllocOp : Op<TileDialect, "mbarrier.alloc"> {
  let summary = "Allocate a Hopper+ asynchronous transaction barrier";
  let arguments = (ins I32Attr:$count, StrAttr:$scope);
  let results = (outs AnyType:$barrier);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Tile_MBarrierArriveExpectTxOp : Op<TileDialect, "mbarrier.arrive_expect_tx"> {
  let summary = "Arrive on an mbarrier and set expected asynchronous transaction bytes";
  let arguments = (ins AnyType:$barrier, I64Attr:$bytes, StrAttr:$semantics, StrAttr:$scope);
  let results = (outs AnyType:$token);
  let assemblyFormat = "$barrier attr-dict";
  let hasVerifier = 1;
}

def Tile_MBarrierTryWaitOp : Op<TileDialect, "mbarrier.try_wait"> {
  let summary = "Wait for a Hopper+ mbarrier token or parity phase";
  let arguments = (ins AnyType:$barrier, AnyType:$token);
  let results = (outs I1:$ready);
  let assemblyFormat = "$barrier `,` $token attr-dict";
  let hasVerifier = 1;
}

def Tile_ReduceOp : Op<TileDialect, "reduce"> {
  let summary = "Deterministic reduction with explicit order";
  let arguments = (ins AnyType:$input, StrAttr:$op, StrAttr:$order);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict";
  let hasVerifier = 1;
}

def Tile_MMAOp : Op<TileDialect, "mma"> {
  let summary = "Tile-level matrix multiply accumulator operation";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs,
                       OptionalAttr<I32Attr>:$sm,
                       OptionalAttr<I64Attr>:$m,
                       OptionalAttr<I64Attr>:$n,
                       OptionalAttr<I64Attr>:$k,
                       OptionalAttr<StrAttr>:$accum);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasVerifier = 1;
}

def Tile_Conv2DOp : Op<TileDialect, "conv2d"> {
  let summary = "Tile-level 2D convolution contract op";
  let arguments = (ins AnyType:$input, AnyType:$filter,
                       OptionalAttr<I64Attr>:$tile_h,
                       OptionalAttr<I64Attr>:$tile_w,
                       OptionalAttr<I64Attr>:$tile_c);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input `,` $filter attr-dict `:` type($input) `,` type($filter) `->` type($result)";
  let hasVerifier = 1;
}

def Tile_KVCacheOp : Op<TileDialect, "kv_cache"> {
  let summary = "Tile-level paged KV-cache contract op";
  let arguments = (ins StrAttr:$source,
                       StrAttr:$result,
                       OptionalAttr<StrAttr>:$storage,
                       OptionalAttr<I64Attr>:$ordinal);
  let assemblyFormat = "attr-dict";
}

def Tile_TMEMAllocOp : Op<TileDialect, "tmem.alloc"> {
  let summary = "Allocate Blackwell TMEM storage for verifier/FileCheck contracts";
  let arguments = (ins AnyMemRef:$result,
                       OptionalAttr<I64Attr>:$bytes,
                       OptionalAttr<I64Attr>:$alignment);
  let assemblyFormat = "$result attr-dict";
  let hasVerifier = 1;
}

def Tile_TMEMLoadOp : Op<TileDialect, "tmem.load"> {
  let summary = "Load from Blackwell TMEM storage";
  let arguments = (ins AnyMemRef:$source);
  let results = (outs AnyType:$value);
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($value)";
}

def Tile_TMEMStoreOp : Op<TileDialect, "tmem.store"> {
  let summary = "Store into Blackwell TMEM storage";
  let arguments = (ins AnyType:$value, AnyMemRef:$dest);
  let assemblyFormat = "$value `,` $dest attr-dict `:` type($value) `,` type($dest)";
}

def Tile_MMATCGen05Op : Op<TileDialect, "mma.tcgen05"> {
  let summary = "SM100 tcgen05 MMA contract op";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, AnyType:$acc,
                       OptionalAttr<I32Attr>:$sm,
                       OptionalAttr<I64Attr>:$cta_group);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs `,` $acc attr-dict `:` type($lhs) `,` type($rhs) `,` type($acc) `->` type($result)";
  let hasVerifier = 1;
}
