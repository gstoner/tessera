//===- tessera_solver.td -------------------------------------*- tablegen -*-===//
// Tessera Solver dialect: nonlinear roots, trigonometric solvers,
// periodic halos, param sweeps & continuation.
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

def TSL_Dialect : Dialect {
  let name = "tsl";
  let cppNamespace = "::tessera::tsl";
}

def TSL_BrentRootOp : Op<TSL_Dialect, "root.brent"> {
  let summary = "Brent's method for root finding on [a,b]";
  let arguments = (ins AnyType:$f, AnyType:$a, AnyType:$b);
  let results   = (outs AnyType:$x);
  let assemblyFormat = "$f `,` $a `,` $b attr-dict `:` type($f) `,` type($a) `,` type($b) `->` type($x)";
}

def TSL_NewtonOp : Op<TSL_Dialect, "root.newton"> {
  let summary = "Newton's method with autodiff-derived derivative";
  let arguments = (ins AnyType:$f, AnyType:$x0);
  let results   = (outs AnyType:$x);
  let assemblyFormat = "$f `,` $x0 attr-dict `:` type($f) `,` type($x0) `->` type($x)";
}

def TSL_SolveTrigOp : Op<TSL_Dialect, "solve_trig"> {
  let summary = "Solve f(θ)=0 with periodicity awareness (wrap=2π|π)";
  let arguments = (ins AnyType:$f, AnyType:$theta0);
  let results   = (outs AnyType:$theta);
  let assemblyFormat = "$f `,` $theta0 attr-dict `:` type($f) `,` type($theta0) `->` type($theta)";
}

def TSL_ParamGridAttr : AttrDef<TSL_Dialect, "param_grid"> {
  let summary = "Parameter grid specification";
  let parameters = (ins StrAttr:$axes);
}
