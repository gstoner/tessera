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
}

def PushOp : Op<Tessera_Queue_Dialect, "push", []> {
  let summary = "Push a tile into the queue";
  let arguments = (ins TileQueue:$q, AnyType:$tile);
  let results = (outs Token:$t);
  let assemblyFormat = "$q `,` $tile attr-dict `:` type($tile)";
}

def PopOp : Op<Tessera_Queue_Dialect, "pop", []> {
  let summary = "Pop a tile from the queue";
  let arguments = (ins TileQueue:$q, Token:$dep);
  let results = (outs AnyType:$tile);
  let assemblyFormat = "$q `,` $dep attr-dict `:` type($tile)";
}
