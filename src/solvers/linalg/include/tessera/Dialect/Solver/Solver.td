
//===- Solver.td - Tessera Solver Dialect ----------------------*- tablegen -*-===//
// Minimal ODS to define solver ops and precision policy attributes.
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Tessera_Solver_Dialect : Dialect {
  // MLIR operation namespaces end at the first dot.  The historical
  // `tessera.solver` spelling therefore registered a dialect that could never
  // own operations named `tessera.solver.*` (those belong to `tessera`).
  let name = "tessera_solver";
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
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::mlir::Type":$LUType, "::mlir::Type":$pivotsType,
                   "::mlir::Value":$A), [{
      $_state.addTypes({LUType, pivotsType});
      $_state.addOperands(A);
    }]>
  ];
  let hasVerifier = 1;
  let assemblyFormat = "$A attr-dict `:` type($A) `->` type($LU) `,` type($Pivots)";
}

// Triangular solve
def TrsmOp : Op<Tessera_Solver_Dialect, "trsm", [Pure]> {
  let summary = "Triangular solve";
  let arguments = (ins AnyType:$Tri, AnyType:$B);
  let results = (outs AnyType:$X);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::mlir::Type":$XType, "::mlir::Value":$Tri,
                   "::mlir::Value":$B), [{
      $_state.addTypes(XType);
      $_state.addOperands({Tri, B});
    }]>
  ];
  let hasVerifier = 0;
  let assemblyFormat = "$Tri `,` $B attr-dict `:` type($Tri) `,` type($B) `->` type($X)";
}

// Cholesky (potrf)
def PotrfOp : Op<Tessera_Solver_Dialect, "potrf", [Pure]> {
  let summary = "Cholesky factorization";
  let arguments = (ins AnyType:$A);
  let results = (outs AnyType:$L);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::mlir::Type":$LType, "::mlir::Value":$A), [{
      $_state.addTypes(LType);
      $_state.addOperands(A);
    }]>
  ];
  let hasVerifier = 1;
  let assemblyFormat = "$A attr-dict `:` type($A) `->` type($L)";
}

// Solve using Cholesky (potrs)
def PotrsOp : Op<Tessera_Solver_Dialect, "potrs", [Pure]> {
  let summary = "Solve using Cholesky factors";
  let arguments = (ins AnyType:$L, AnyType:$B);
  let results = (outs AnyType:$X);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::mlir::Type":$XType, "::mlir::Value":$L,
                   "::mlir::Value":$B), [{
      $_state.addTypes(XType);
      $_state.addOperands({L, B});
    }]>
  ];
  let hasVerifier = 0;
  let assemblyFormat = "$L `,` $B attr-dict `:` type($L) `,` type($B) `->` type($X)";
}

// Iterative refinement step
def IRStepOp : Op<Tessera_Solver_Dialect, "ir_step", [Pure]> {
  let summary = "Single iterative refinement step (x_{k+1}, r)";
  let arguments = (ins AnyType:$A, AnyType:$xk, AnyType:$b);
  let results = (outs AnyType:$xkp1, AnyType:$r);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::mlir::Type":$xkp1Type, "::mlir::Type":$rType,
                   "::mlir::Value":$A, "::mlir::Value":$xk,
                   "::mlir::Value":$b), [{
      $_state.addTypes({xkp1Type, rType});
      $_state.addOperands({A, xk, b});
    }]>
  ];
  let hasVerifier = 0;
  let assemblyFormat = "$A `,` $xk `,` $b attr-dict `:` type($A) `,` type($xk) `,` type($b) `->` type($xkp1) `,` type($r)";
}

// Target-independent implicit-function differentiation contracts.  These ops
// intentionally use generic assembly: the parameter/solution arities are
// variadic and are verified against the referenced residual by
// NewtonAutodiffPass before any of these values are created.
def ImplicitOp : Op<Tessera_Solver_Dialect, "implicit", [Pure]> {
  let summary = "Implicit solution linked to an explicit residual function";
  let arguments = (ins Variadic<AnyType>:$parameters,
                       FlatSymbolRefAttr:$residual);
  let results = (outs Variadic<AnyType>:$solutions);
}

def ResidualOp : Op<Tessera_Solver_Dialect, "residual", [Pure]> {
  let summary = "Evaluate a residual at an explicit parameter/solution point";
  let arguments = (ins Variadic<AnyType>:$primals,
                       FlatSymbolRefAttr:$callee);
  let results = (outs Variadic<AnyType>:$residuals);
}

def LinearSolveOp : Op<Tessera_Solver_Dialect, "linear_solve", [Pure]> {
  let summary = "Matrix-free solve of a residual linearization";
  let arguments = (ins Variadic<AnyType>:$inputs,
                       FlatSymbolRefAttr:$residual,
                       BoolAttr:$transpose,
                       StrAttr:$linearization,
                       OptionalAttr<F64Attr>:$rhs_scale);
  let results = (outs Variadic<AnyType>:$solution);
}

def ResidualAdjointOp
    : Op<Tessera_Solver_Dialect, "residual_adjoint", [Pure]> {
  let summary = "Apply a residual VJP to an adjoint-state value";
  let arguments = (ins Variadic<AnyType>:$inputs,
                       FlatSymbolRefAttr:$residual,
                       StrAttr:$wrt,
                       F64Attr:$scale);
  let results = (outs Variadic<AnyType>:$cotangents);
}

def ResidualJvpOp : Op<Tessera_Solver_Dialect, "residual_jvp", [Pure]> {
  let summary = "Apply a residual JVP to parameter tangent values";
  let arguments = (ins Variadic<AnyType>:$inputs,
                       FlatSymbolRefAttr:$residual,
                       StrAttr:$wrt);
  let results = (outs Variadic<AnyType>:$tangents);
}
