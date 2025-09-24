include "mlir/IR/OpBase.td"
def Power_Dialect : Dialect { let name = "power"; let cppNamespace = "::tessera::power"; }
def Power_AttnOp : Op<Power_Dialect, "attn", [DeclareOpInterfaceMethods<MemoryEffectOpInterface>]> {
  let summary = "Symmetric power attention (linear-cost)";
  let arguments = (ins AnyTensor:$q, AnyTensor:$k, AnyTensor:$v,
                        I64Attr:$state,
                        OptionalAttr<I64Attr>:$window,
                        UnitAttr:$causal);
  let results = (outs AnyTensor:$o);
  let assemblyFormat = [{ $q `,` $k `,` $v attr-dict `:` type($q) `,` type($k) `,` type($v) `->` type($o) }];
}