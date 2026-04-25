//===- Queue.td (v1.3) -----------------------------------------------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"
include "mlir/IR/DialectSpecification.td"

def Tessera_Queue_Dialect : Dialect {
  let name = "tessera.queue";
  let cppNamespace = "::tessera::queue";
}

def TileQueueType : TypeDef<"TileQueueType"> {
  let summary = "Opaque tile-queue handle";
  let assemblyFormat = "";
}

def TokenType : TypeDef<"TokenType"> {
  let summary = "Opaque token for queue ordering";
  let assemblyFormat = "";
}

def CreateOp : Op<"create", []> {
  let summary = "Create a tile queue";
  let results = (outs TileQueueType:$q);
  let assemblyFormat = "attr-dict";
}

def PushOp : Op<"push", []> {
  let summary = "Push a tile into the queue";
  let arguments = (ins TileQueueType:$q, AnyType:$tile);
  let results = (outs TokenType:$t);
  let assemblyFormat = "$q `,` $tile attr-dict";
}

def PopOp : Op<"pop", []> {
  let summary = "Pop a tile from the queue";
  let arguments = (ins TileQueueType:$q, TokenType:$dep);
  let results = (outs AnyType:$tile);
  let assemblyFormat = "$q `,` $dep attr-dict";
}
