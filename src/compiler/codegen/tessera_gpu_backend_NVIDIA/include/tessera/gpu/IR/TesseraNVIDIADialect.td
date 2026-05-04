include "mlir/IR/OpBase.td"

def TesseraNVIDIADialect : Dialect {
  let name = "tessera_nvidia";
  let cppNamespace = "tessera::nvidia";
  let summary = "Tessera NVIDIA Target IR contract dialect";
  let description = [{
    Hardware-free NVIDIA Target IR contracts for Hopper and Blackwell lowering.
    The v1 contract uses generic operations in this dialect so tests and
    frontend artifacts can stabilize before native NVGPU/NVVM op builders are
    fully typed.
  }];
  let isExtensible = 1;
}
