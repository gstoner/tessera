//===- AdjointInterface.td - Tessera autodiff interface --------*- tablegen -*-===//
//
// Phase F4 of docs/audit/roadmap/ROADMAP_AUDIT.md. ODS for the AdjointInterface
// op trait. Each differentiable Graph IR op declares an `adjoint` method
// that materializes its backward computation as Graph IR ops; the
// AutodiffPass walks the IR and inserts adjoint ops via interface dispatch.
//
// Cross-references:
//   * lib/Transforms/AutodiffPass.cpp — the pass that consumes this interface
//   * docs/spec/AUTODIFF_SPEC.md §Phase F4 — design + non-goals
//   * python/tessera/autodiff/__init__.py — runtime VJP registry that the
//     pass consults for ops carrying `customAdjointName()`
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_INTERFACES_ADJOINTINTERFACE_TD
#define TESSERA_INTERFACES_ADJOINTINTERFACE_TD

include "mlir/IR/OpBase.td"

def Tessera_AdjointInterface : OpInterface<"AdjointInterface"> {
  let description = [{
    Marks an op as differentiable in reverse-mode autodiff.

    Implementations declare how to construct the backward computation given
    the cotangents flowing in to the op's outputs and the op's own forward
    operands / attributes.

    The `buildAdjoint` method emits Graph IR ops that produce the
    `inputCotangents` — one per operand in the same order. Operands that
    are non-differentiable (e.g., dtype attributes, shape integers) get
    a `Value()` placeholder.

    The AutodiffPass walks the forward IR top-down, calling `buildAdjoint`
    on each op in reverse program order. Cotangents are tracked in a
    `DenseMap<Value, Value>` keyed by forward SSA value.
  }];

  let cppNamespace = "::tessera";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Build the adjoint (backward) computation for this op.

        `builder` is positioned just after the forward op (caller responsible
        for moving to the right point during the reverse-walk). `outputCotangents`
        has one entry per `op->getNumResults()`. Returns one `Value` per
        operand (`op->getNumOperands()`), with `Value()` for non-differentiable
        operands.

        Implementations must NOT mutate the forward op or its operands;
        they should only emit new ops via `builder`.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::Value>",
      /*methodName=*/"buildAdjoint",
      /*args=*/(ins
        "::mlir::OpBuilder &":$builder,
        "::mlir::ValueRange":$outputCotangents
      )
    >,

    InterfaceMethod<
      /*desc=*/[{
        Returns true if this op participates in autodiff. Default true; ops
        with no analytical adjoint can override to false to surface a clear
        diagnostic rather than emitting a stale `buildAdjoint` placeholder.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isDifferentiable",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return true;
      }]
    >,

    InterfaceMethod<
      /*desc=*/[{
        For ops whose adjoint is registered from Python (via
        `tessera.autodiff.custom_rule`), this returns the registry key the
        AutodiffPass should consult. Default empty — ops that ship with a
        compiled-in adjoint do not need this hook.
      }],
      /*retTy=*/"::llvm::StringRef",
      /*methodName=*/"customAdjointName",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return ::llvm::StringRef();
      }]
    >
  ];
}

#endif  // TESSERA_INTERFACES_ADJOINTINTERFACE_TD
