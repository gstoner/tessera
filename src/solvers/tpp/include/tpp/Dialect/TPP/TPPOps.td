//===- TPPOps.td -----------------------------------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "TPP.td"
include "TPPTypes.td"
include "TPPAttrs.td"

def TPP_BCEnforceOp : Op<TPP_Dialect, "bc.enforce", [Pure]> {
  let summary = "Apply boundary condition policy to a field";
  let arguments = (ins AnyType:$field, OptionalAttr<TPP_BCAttr>:$bc);
  let results   = (outs AnyType:$out);
  let assemblyFormat = "$field attr-dict `:` type($field) `->` type($out)";
}

def TPP_GradOp : Op<TPP_Dialect, "grad", [Pure]> {
  let summary = "Gradient of a field";
  let arguments = (ins AnyType:$x);
  let results   = (outs AnyType:$y);
  let assemblyFormat = "$x attr-dict `:` type($x) `->` type($y)";
}

def TPP_StencilApplyOp : Op<TPP_Dialect, "stencil.apply", [Pure]> {
  let summary = "Apply a stencil kernel with a given radius/order";
  let arguments = (ins AnyType:$x, AnyType:$kernel);
  let results   = (outs AnyType:$y);
  let assemblyFormat = "$x `,` $kernel attr-dict `:` type($x) `,` type($kernel) `->` type($y)";
}

def TPP_HaloExchangeOp : Op<TPP_Dialect, "halo.exchange",
    [Pure, SameOperandsAndResultType]> {
  let summary = "Exchange halo/ghost cells of a field with mesh neighbours";
  let description = [{
    Materialised by `-tpp-distribute-halo` in front of every stencil-like
    consumer.  It represents filling the `tpp.halo`-wide ghost region of a
    field from neighbouring mesh ranks (or, single-rank, applying the periodic
    wrap locally).  The `tpp.halo`, `tpp.mesh.axes`, `tpp.dist.overlap` and
    `tpp.dist.local_only` attributes carry the exchange plan for codegen.
  }];
  let arguments = (ins AnyType:$field);
  let results   = (outs AnyType:$exchanged);
  let assemblyFormat = "$field attr-dict `:` type($field)";
}

def TPP_TimeStepOp : Op<TPP_Dialect, "time.step", [Pure, NoTerminator]> {
  let summary = "Temporal step region (e.g., RK schemes)";
  let description = [{
    Holds the per-step stencil computation in its body region.  The region is
    `NoTerminator` so the scaffold parses without a dedicated yield op; the
    `scheme`/`dt` attributes are normalised by `-tpp-legalize-space-time` into
    `tpp.time.stages`/`tpp.time.order`/`tpp.time.dt`.
  }];
  let arguments = (ins Variadic<AnyType>:$state);
  let results   = (outs Variadic<AnyType>:$out);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$state attr-dict-with-keyword $body `:` functional-type($state, $out)";
}
