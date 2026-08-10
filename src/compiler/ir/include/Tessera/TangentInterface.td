//===- TangentInterface.td - Tessera forward-mode interface ----*- tablegen -*-===//
//
// Compiler-owned JVP construction for Graph IR operations.  This is distinct
// from LinearTransposeInterface: nonlinear operations need their primal
// operands/results to construct tangents, while linear transposition remains a
// reverse-mode contract.
//
//===----------------------------------------------------------------------===//

#ifndef TESSERA_INTERFACES_TANGENTINTERFACE_TD
#define TESSERA_INTERFACES_TANGENTINTERFACE_TD

include "mlir/IR/OpBase.td"

def Tessera_TangentInterface : OpInterface<"TangentInterface"> {
  let description = [{
    Marks a Graph IR operation as supporting compiler-owned forward-mode
    differentiation. Implementations construct one result tangent per primal
    result from one input tangent per primal operand. A null input tangent
    denotes a statically inactive operand.
  }];

  let cppNamespace = "::tessera";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Build this operation's JVP in Graph IR. The builder is positioned after
        the cloned primal operation. Implementations emit new operations only
        and return exactly one tangent per primal result.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::Value>",
      /*methodName=*/"buildTangent",
      /*args=*/(ins
        "::mlir::OpBuilder &":$builder,
        "::mlir::ValueRange":$inputTangents
      )
    >
  ];
}

#endif  // TESSERA_INTERFACES_TANGENTINTERFACE_TD
