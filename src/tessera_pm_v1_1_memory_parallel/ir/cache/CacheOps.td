//===- CacheOps.td - Cache abstractions (KV, page tables, ring) ---------*- tablegen -*-===//
include "mlir/IR/OpBase.td"

def CacheDialect : Dialect {
  let name = "cache";
  let cppNamespace = "::tessera::cache";
}

def Cache_KVType : TypeDef<CacheDialect, "kv"> {
  let mnemonic = "kv";
}

def Cache_PageType : TypeDef<CacheDialect, "page"> {
  let mnemonic = "page";
}

def Cache_RingType : TypeDef<CacheDialect, "ring"> {
  let mnemonic = "ring";
}

def Cache_KVCreateOp : Op<CacheDialect, "kv.create"> {
  let summary = "Create KV cache object with policy";
  let arguments = (ins);
  let results = (outs Cache_KVType:$kv);
  let hasVerifier = 1;
}

def Cache_PageLookupOp : Op<CacheDialect, "page.lookup"> {
  let arguments = (ins Cache_KVType:$kv, I32:$pos);
  let results = (outs Cache_PageType:$page);
  let hasVerifier = 1;
}

def Cache_PageReadOp : Op<CacheDialect, "page.read"> {
  let arguments = (ins Cache_KVType:$kv, Cache_PageType:$page);
  let results = (outs AnyType:$k, AnyType:$v);
}

def Cache_PageWriteOp : Op<CacheDialect, "page.write"> {
  let arguments = (ins Cache_KVType:$kv, Cache_PageType:$page, AnyType:$k, AnyType:$v);
}

def Cache_PTCreateOp : Op<CacheDialect, "pt.create"> {
  let results = (outs AnyType:$pt);
  let hasVerifier = 1;
}

def Cache_RingCreateOp : Op<CacheDialect, "ring.create"> {
  let results = (outs Cache_RingType:$ring);
  let hasVerifier = 1;
}

def Cache_RingPushOp : Op<CacheDialect, "ring.push"> {
  let arguments = (ins Cache_RingType:$ring, AnyType:$item);
}

def Cache_RingPopOp : Op<CacheDialect, "ring.pop"> {
  let arguments = (ins Cache_RingType:$ring);
  let results = (outs AnyType:$item);
}
