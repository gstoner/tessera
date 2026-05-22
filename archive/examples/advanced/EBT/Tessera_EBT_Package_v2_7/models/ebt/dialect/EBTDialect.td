//===- EBTDialect.td -------------------------------------*- tablegen -*-===//
// Tessera EBT Dialect (ODS scaffold)
include "mlir/IR/OpBase.td"

def EBT_Dialect : Dialect {
  let name = "tessera.ebt";
  let cppNamespace = "::tessera::ebt";
  let summary = "Energy-Based Transformer ops";
  let description = [{
    Core EBT primitives used by passes to build K×T inner loops, energies, and gradients.
  }];
}

// Types are tensor-based; we keep shapes opaque in ODS scaffold.

def EBT_Energy : Op<EBT_Dialect, "energy", [NoSideEffect]> {
  let summary = "Generic energy op E(h,y) -> scalar or per-token scalar";
  let arguments = (ins AnyType:$h, AnyType:$y);
  let results   = (outs AnyType:$E);
}

def EBT_EnergyBilinear : Op<EBT_Dialect, "energy_bilinear", [NoSideEffect]> {
  let summary = "Bilinear energy y^T W h (+b)";
  let arguments = (ins AnyType:$h, AnyType:$y, AnyType:$W, AnyType:$b);
  let results   = (outs AnyType:$E);
}

def EBT_EnergyMLP : Op<EBT_Dialect, "energy_mlp", [NoSideEffect]> {
  let summary = "MLP head on concat(y,h)";
  let arguments = (ins AnyType:$h, AnyType:$y, AnyType:$W1, AnyType:$b1, AnyType:$W2, AnyType:$b2);
  let results   = (outs AnyType:$E);
}

def EBT_EnergyBilinearJVP : Op<EBT_Dialect, "energy_bilinear_jvp", [NoSideEffect]> {
  let summary = "Directional derivative dE/dy · v (bilinear head)";
  let arguments = (ins AnyType:$h, AnyType:$y, AnyType:$v, AnyType:$W);
  let results   = (outs AnyType:$dE);
}

def EBT_GradY : Op<EBT_Dialect, "grad_y", [NoSideEffect]> {
  let summary = "Gradient of energy wrt y";
  let arguments = (ins AnyType:$h, AnyType:$y);
  let results   = (outs AnyType:$g);
}

def EBT_InnerStep : Op<EBT_Dialect, "inner_step"> {
  let summary = "Update y_{t+1} = step(y_t, grad, eta)";
  let arguments = (ins AnyType:$y, AnyType:$g, AnyType:$eta);
  let results   = (outs AnyType:$y_next);
}

def EBT_DecodeInit : Op<EBT_Dialect, "decode_init"> {
  let summary = "Initializer for candidate y0 from x";
  let arguments = (ins AnyType:$x);
  let results   = (outs AnyType:$y0);
}

def EBT_SelfVerify : Op<EBT_Dialect, "self_verify", [NoSideEffect]> {
  let summary = "Choose argmin over candidate energies";
  let arguments = (ins AnyType:$E_all, AnyType:$Y_all);
  let results   = (outs AnyType:$y_best);
}
