
//===- Solver.td - Tessera Solver Dialect ----------------------*- tablegen -*-===//
// Minimal ODS to define solver ops and precision policy attributes.
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Tessera_Solver_Dialect : Dialect {
  let name = "tessera.solver";
  let cppNamespace = "tessera::solver";
}

def PrecisionPolicyAttr : AttrDef<Tessera_Solver_Dialect, "PrecisionPolicy"> {
  let mnemonic = "precision";
  let parameters = (ins StringRefParameter<"compute dtype">:$compute,
                        StringRefParameter<"accumulator dtype">:$accum,
                        StringRefParameter<"residual dtype">:$residual,
                        StringRefParameter<"scale dtype">:$scale);
  let summary = "Mixed precision policy (compute/accum/residual/scale)";
  let assemblyFormat = "`<` $compute `,` $accum `,` $residual `,` $scale `>`";
}

// Example: LU factorization (getrf)
def GetrfOp : Op<Tessera_Solver_Dialect, "getrf", [Pure]> {
  let summary = "LU factorization with partial pivoting";
  let description = [{
    Accepts dense or batched dense tensors. Mixed precision via 'policy' attribute.
  }];
  let arguments = (ins AnyType:$A);
  let results = (outs AnyType:$LU, AnyType:$Pivots);
  let hasVerifier = 1;
  let assemblyFormat = "$A attr-dict `:` type($A) `->` type($LU) `,` type($Pivots)";
}

// Triangular solve
def TrsmOp : Op<Tessera_Solver_Dialect, "trsm", [Pure]> {
  let summary = "Triangular solve";
  let arguments = (ins AnyType:$Tri, AnyType:$B);
  let results = (outs AnyType:$X);
  let hasVerifier = 0;
  let assemblyFormat = "$Tri `,` $B attr-dict `:` type($Tri) `,` type($B) `->` type($X)";
}

// Cholesky (potrf)
def PotrfOp : Op<Tessera_Solver_Dialect, "potrf", [Pure]> {
  let summary = "Cholesky factorization";
  let arguments = (ins AnyType:$A);
  let results = (outs AnyType:$L);
  let hasVerifier = 1;
  let assemblyFormat = "$A attr-dict `:` type($A) `->` type($L)";
}

// Solve using Cholesky (potrs)
def PotrsOp : Op<Tessera_Solver_Dialect, "potrs", [Pure]> {
  let summary = "Solve using Cholesky factors";
  let arguments = (ins AnyType:$L, AnyType:$B);
  let results = (outs AnyType:$X);
  let hasVerifier = 0;
  let assemblyFormat = "$L `,` $B attr-dict `:` type($L) `,` type($B) `->` type($X)";
}

// Iterative refinement step
def IRStepOp : Op<Tessera_Solver_Dialect, "ir_step", [Pure]> {
  let summary = "Single iterative refinement step (x_{k+1}, r)";
  let arguments = (ins AnyType:$A, AnyType:$xk, AnyType:$b);
  let results = (outs AnyType:$xkp1, AnyType:$r);
  let hasVerifier = 0;
  let assemblyFormat = "$A `,` $xk `,` $b attr-dict `:` type($A) `,` type($xk) `,` type($b) `->` type($xkp1) `,` type($r)";
}
