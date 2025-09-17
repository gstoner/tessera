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

def Tile_ReduceOp : Op<TileDialect, "reduce"> {
  let summary = "Deterministic reduction with explicit order";
  let arguments = (ins AnyType:$input, StrAttr:$op, StrAttr:$order);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict";
  let hasVerifier = 1;
}
