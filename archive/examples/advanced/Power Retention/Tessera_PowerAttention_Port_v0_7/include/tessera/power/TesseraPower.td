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

def Power_RetentionOp : Op<Power_Dialect, "retention",
    [DeclareOpInterfaceMethods<MemoryEffectOpInterface>]> {
  let summary = "Power Retention (linear-cost; training/inference variants)";
  let arguments = (ins AnyTensor:$q, AnyTensor:$k, AnyTensor:$v,
                        Optional<AnyTensor>:$log_g,
                        I64Attr:$deg,
                        I64Attr:$chunk,
                        OptionalAttr<I64Attr>:$switch_over,
                        UnitAttr:$causal);
  let results = (outs AnyTensor:$o, AnyTensor:$state, AnyTensor:$sum_of_keys);
  let assemblyFormat = [{ $q `,` $k `,` $v (`,` $log_g^)? attr-dict `:`
    type($q) `,` type($k) `,` type($v) (`,` type($log_g))? `->` type($o) `,` type($state) `,` type($sum_of_keys)}];
}
