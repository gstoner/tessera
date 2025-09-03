//===- kan_ops.td - Tessera KAN dialect (starter) ---------------------*- tablegen -*-===//
// Minimal op stubs that lower into Tessera/Linalg building blocks.
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Tessera_KAN_Dialect : Dialect {
  let name = "kan";
  let cppNamespace = "tessera::kan";
}

def KAN_BSplineEvalOp : Op<
  Tessera_KAN_Dialect, "bspline_eval",
  [NoSideEffect]
> {
  let summary = "Evaluate B-spline basis on each input feature";
  let description = [
    "Inputs: x: tensor<[B,I]>, attrs: degree, grid_min, grid_max, grid_size. "
    "Outputs: phi: tensor<[B,I,M]>, with M = grid_size + degree - 1 (open uniform knots)."
  ];
  let arguments = (ins AnyRankedTensor:$x,
                      I64Attr:$degree,
                      F32Attr:$grid_min,
                      F32Attr:$grid_max,
                      I64Attr:$grid_size);
  let results = (outs AnyRankedTensor:$phi);
}

def KAN_LinearMixOp : Op<
  Tessera_KAN_Dialect, "linear_mix",
  [NoSideEffect]
> {
  let summary = "Linear combination (phi × W_spline) + (x × W_base)";
  let arguments = (ins AnyRankedTensor:$phi,
                       AnyRankedTensor:$W_spline,
                       AnyRankedTensor:$x,
                       AnyRankedTensor:$W_base);
  let results = (outs AnyRankedTensor:$y);
}