//===- Queue.td (v1.3) -----------------------------------------------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"
include "mlir/IR/DialectSpecification.td"
include "mlir/IR/AttrTypeBase.td"

def Tessera_Queue_Dialect : Dialect {
  let name = "tessera.queue";
  let cppNamespace = "::tessera::queue";
}

def TileQueueType : TypeDef<Tessera_Queue_Dialect, "TileQueueType"> {
  let summary = "Opaque tile-queue handle";
  let assemblyFormat = "";
}

def TokenType : TypeDef<Tessera_Queue_Dialect, "TokenType"> {
  let summary = "Opaque token for queue ordering";
  let assemblyFormat = "";
}

def CreateOp : Op<Tessera_Queue_Dialect, "create", []> {
  let summary = "Create a tile queue";
  let results = (outs TileQueueType:$q);
  let assemblyFormat = "attr-dict";
}

def PushOp : Op<Tessera_Queue_Dialect, "push", []> {
  let summary = "Push a tile into the queue";
  let arguments = (ins TileQueueType:$q, AnyType:$tile);
  let results = (outs TokenType:$t);
  let assemblyFormat = "$q `,` $tile attr-dict";
}

def PopOp : Op<Tessera_Queue_Dialect, "pop", []> {
  let summary = "Pop a tile from the queue";
  let arguments = (ins TileQueueType:$q, TokenType:$dep);
  let results = (outs AnyType:$tile);
  let assemblyFormat = "$q `,` $dep attr-dict";
}
