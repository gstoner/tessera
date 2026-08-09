//===- LinearTransposeInterface.td - Linear Graph IR transpose -*- tablegen -*-===//
//
// Compiler-owned transposition for linear Graph IR operations.  A linear op
// declares one rule that maps output cotangents to operand cotangents; reverse
// mode consumes that rule directly and forward mode may reuse the primal op.
// This is the production counterpart of python/tessera/autodiff/linear.py,
// which remains a numerical oracle while Graph IR families migrate.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_INTERFACES_LINEARTRANSPOSEINTERFACE_TD
#define TESSERA_INTERFACES_LINEARTRANSPOSEINTERFACE_TD

include "mlir/IR/OpBase.td"

def Tessera_LinearTransposeInterface : OpInterface<"LinearTransposeInterface"> {
  let description = [{
    Marks a Graph IR operation as linear in one or more operands and provides
    its transposed action.  This includes multilinear operations such as
    matmul: each tensor operand is linear while the other operand is held
    fixed.

    Implementations construct operand cotangents from result cotangents.  They
    may use other primal operands as coefficients for a multilinear map, but
    must not differentiate those coefficients as part of the same operand
    action.  The returned vector has one entry per operand; a null Value denotes
    an operand outside the differentiable linear map.
  }];

  let cppNamespace = "::tessera";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{Returns whether the operation is linear in operandIndex.}],
      /*retTy=*/"bool",
      /*methodName=*/"isLinearInOperand",
      /*args=*/(ins "unsigned":$operandIndex),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return operandIndex < $_op->getNumOperands();
      }]
    >,

    InterfaceMethod<
      /*desc=*/[{
        Build the transposed linear map for this operation.  The builder is
        positioned in the backward Graph IR and outputCotangents has one entry
        per result.  Implementations emit new operations only and must not
        mutate the primal operation.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::Value>",
      /*methodName=*/"buildLinearTranspose",
      /*args=*/(ins
        "::mlir::OpBuilder &":$builder,
        "::mlir::ValueRange":$outputCotangents
      )
    >
  ];
}

#endif  // TESSERA_INTERFACES_LINEARTRANSPOSEINTERFACE_TD
