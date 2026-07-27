//===- Attn.td (v1.3) ------------------------------------------------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"
include "mlir/IR/DialectSpecification.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Tessera_Attn_Dialect : Dialect {
  let name = "tessera_attn";
  let cppNamespace = "::tessera::attn";
}

def LseSaveOp : Op<"lse.save", [MemoryEffects<[MemWrite]>]> {
  let summary = "Store finalized per-row LSE into a persistent checkpoint";
  let arguments = (ins AnyType:$lse, AnyMemRef:$destination, Index:$row_offset,
                       StrAttr:$identity, StrAttr:$memory_space, StrAttr:$scope,
                       OptionalAttr<StrAttr>:$cache_policy);
  let assemblyFormat = "$lse `into` $destination `at` $row_offset attr-dict";
  let hasVerifier = 1; // shape-aware check in C++
}

def LseLoadOp : Op<"lse.load", [MemoryEffects<[MemRead]>]> {
  let summary = "Load per-row LSE from a persistent checkpoint";
  let arguments = (ins AnyMemRef:$source, Index:$row_offset,
                       StrAttr:$identity, StrAttr:$memory_space, StrAttr:$scope,
                       OptionalAttr<StrAttr>:$cache_policy);
  let results = (outs AnyType:$lse);
  let assemblyFormat = "$source `at` $row_offset attr-dict";
  let hasVerifier = 1;
}
