include "mlir/IR/OpBase.td"
// Pure is defined here, not in OpBase.td. Without this include tablegen fails
// with "Variable not defined" — reachable only under -DTESSERA_ENABLE_CUDA=ON,
// which is the whole reason it survived.
include "mlir/Interfaces/SideEffectInterfaces.td"
def Power_Dialect : Dialect { let name = "power"; let cppNamespace = "::tessera::power"; }
def Power_AttnOp : Op<Power_Dialect, "attn", [Pure]> {
  let summary = "Symmetric power attention (linear-cost)";
  let arguments = (ins AnyTensor:$q, AnyTensor:$k, AnyTensor:$v,
                        I64Attr:$state,
                        OptionalAttr<I64Attr>:$window,
                        UnitAttr:$causal);
  let results = (outs AnyTensor:$o);
  let assemblyFormat = [{ $q `,` $k `,` $v attr-dict `:` type($q) `,` type($k) `,` type($v) `->` type($o) }];
}

def Power_RetentionOp : Op<Power_Dialect, "retention",
    [Pure]> {
  let summary = "Power Retention (linear-cost; training/inference variants)";
  let arguments = (ins AnyTensor:$q, AnyTensor:$k, AnyTensor:$v,
                        Optional<AnyTensor>:$log_g,
                        I64Attr:$deg,
                        I64Attr:$chunk,
                        OptionalAttr<I64Attr>:$switch_over,
                        UnitAttr:$causal);
  let results = (outs AnyTensor:$o, AnyTensor:$state, AnyTensor:$sum_of_keys);
  // Both optional groups need their own anchor: MLIR resolves each `(...)?`
  // independently, so the type group cannot borrow the operand group's `^`.
  // Without it tablegen fails with "optional group has no anchor element" --
  // reachable only under -DTESSERA_ENABLE_CUDA=ON, which is why it survived.
  let assemblyFormat = [{ $q `,` $k `,` $v (`,` $log_g^)? attr-dict `:`
    type($q) `,` type($k) `,` type($v) (`,` type($log_g)^)? `->` type($o) `,` type($state) `,` type($sum_of_keys)}];
}
