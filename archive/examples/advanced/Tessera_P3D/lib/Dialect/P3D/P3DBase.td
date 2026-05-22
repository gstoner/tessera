//===- P3DBase.td - Tessera P3D Dialect base -------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def P3D_Dialect : Dialect {
  let name = "p3d";
  let summary = "Tessera 3D-physics surrogate (P3D) dialect";
  let description = [{
    High-level ops for scalable 3D surrogates: Conv3D pyramid, global context,
    multires down/up-sampling, and boundary/halo policies.
  }];
  let cppNamespace = "::mlir::tessera::p3d";
}
