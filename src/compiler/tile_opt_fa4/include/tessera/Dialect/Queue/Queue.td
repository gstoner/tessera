//===- Queue.td (v1.3) -----------------------------------------------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def Tessera_Queue_Dialect : Dialect {
  let name = "tessera.queue";
  let cppNamespace = "::tessera::queue";
}

def TileQueue : TypeDef<Tessera_Queue_Dialect, "TileQueue"> {
  let mnemonic = "tile_queue";
  let summary = "Opaque tile-queue handle";
  let assemblyFormat = "";
}

def Token : TypeDef<Tessera_Queue_Dialect, "Token"> {
  let mnemonic = "token";
  let summary = "Opaque token for queue ordering";
  let assemblyFormat = "";
}

def CreateOp : Op<Tessera_Queue_Dialect, "create", []> {
  let summary = "Create a tile queue";
  let results = (outs TileQueue:$q);
  let assemblyFormat = "attr-dict";
  // Sprint V8 (2026-05-22): minimal verifier — the result type is
  // already constrained by ODS to !tessera.queue.tile_queue, so
  // verify only checks the op has no operands (in case a future
  // TD revision accidentally adds one).
  let hasVerifier = 1;
}

def PushOp : Op<Tessera_Queue_Dialect, "push", []> {
  let summary = "Push a tile into the queue";
  let arguments = (ins TileQueue:$q, AnyType:$tile);
  let results = (outs Token:$t);
  let assemblyFormat = "$q `,` $tile attr-dict `:` type($tile)";
  // Sprint V8 (2026-05-22): producer-end data-flow well-formedness.
  // Queue handle MUST be defined by `tessera.queue.create`; tile
  // operand MUST be a ranked tensor or memref (the TD's `AnyType`
  // is too permissive for the FA-4 pipeline semantics).
  let hasVerifier = 1;
}

def PopOp : Op<Tessera_Queue_Dialect, "pop", []> {
  let summary = "Pop a tile from the queue";
  let arguments = (ins TileQueue:$q, Token:$dep);
  let results = (outs AnyType:$tile);
  let assemblyFormat = "$q `,` $dep attr-dict `:` type($tile)";
  // Sprint V8 (2026-05-22): consumer-end data-flow well-formedness.
  // Queue handle MUST be defined by `tessera.queue.create`; the
  // dep token MUST be defined by `tessera.queue.push` (the token is
  // strictly a push-result, not, e.g., a function arg).  Result
  // tile MUST be a ranked tensor or memref.
  let hasVerifier = 1;
}
