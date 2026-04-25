//===- tessera_sparse.td --------------------------------------*- tablegen -*-===//
// Tessera Sparse dialect: types and ops for sparse algebra & solvers.
// This is intentionally small but representative; extend as needed.
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TSS_Dialect : Dialect {
  let name = "tss";
  let cppNamespace = "::tessera::tss";
}

def TSS_SparseTensorType : TypeDef<TSS_Dialect, "SparseTensor"> {
  let summary = "Sparse tensor with explicit layout metadata";
  let description = [{
Represents a sparse tensor with layout and structural attributes. This is a
front-door carrier for CSR/COO/ELL/BSR/Hybrid; backends may lower into
target-specific buffers + index arrays.
  }];
  let parameters = (ins StrAttr:$layout, OptionalAttr<I64Attr>:$rb, OptionalAttr<I64Attr>:$cb);
}

def TSS_SpmvOp : Op<TSS_Dialect, "spmv", [NoMemoryEffect]> {
  let summary = "y = A * x for sparse A";
  let arguments = (ins AnyType:$A, AnyType:$x);
  let results   = (outs AnyType:$y);
  let assemblyFormat = "$A `,` $x attr-dict `:` type($A) `,` type($x) `->` type($y)";
}

def TSS_SpmmOp : Op<TSS_Dialect, "spmm", [NoMemoryEffect]> {
  let summary = "C = A * B for sparse A";
  let arguments = (ins AnyType:$A, AnyType:$B);
  let results   = (outs AnyType:$C);
  let assemblyFormat = "$A `,` $B attr-dict `:` type($A) `,` type($B) `->` type($C)";
}

def TSS_CGOp : Op<TSS_Dialect, "cg", [NoMemoryEffect]> {
  let summary = "Conjugate-Gradient solve for SPD systems";
  let description = [{
Solves A x = b with optional preconditioner region.
Attributes:
  tol: f64
  max_iter: i64
  mixed_precision: i1
  residual_track: i1
  }];
  let arguments = (ins AnyType:$A, AnyType:$b, Variadic<AnyType>:$init);
  let results   = (outs AnyType:$x);
  let regions = (region AnyRegion:$precond);
  let assemblyFormat = "$A `,` $b (`init` $init^)? attr-dict-with-keyword ($precond^)? `:` type($A) `,` type($b) `->` type($x)";
}

def TSS_GMRESOp : Op<TSS_Dialect, "gmres", [NoMemoryEffect]> {
  let summary = "GMRES solve for non-symmetric systems";
  let arguments = (ins AnyType:$A, AnyType:$b);
  let results   = (outs AnyType:$x);
  let assemblyFormat = "$A `,` $b attr-dict `:` type($A) `,` type($b) `->` type($x)";
}
