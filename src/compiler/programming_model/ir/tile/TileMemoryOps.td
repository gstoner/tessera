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
  let arguments = (ins AnyMemRef:$src, AnyMemRef:$dst,
                   I32Attr:$stage, OptionalAttr<I32Attr>:$vector);
  let assemblyFormat = "$src `,` $dst attr-dict";
  let hasVerifier = 1;
}

def Tile_WaitAsyncOp : Op<TileDialect, "wait_async"> {
  let summary = "Wait for a completed async stage";
  let arguments = (ins I32Attr:$stage);
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
