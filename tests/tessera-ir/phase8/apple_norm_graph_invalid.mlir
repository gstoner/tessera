// APPLE-NTBI-2 slice 9: the isolated apple7 f16 norm is claimed by the
// Graph -> Schedule collector; one beyond the synthesized kernel's int32
// extent fails closed with the stable semantic-kernel diagnostic rather than
// reaching a launch that cannot carry it (Decision #21).
// RUN: not tessera-opt --tessera-graph-to-schedule %s 2>&1 | FileCheck %s

// CHECK: error: E2E-REAL-6 Graph->Schedule requires a supported static semantic-kernel contract for the selected target
module attributes {tessera.target = "apple_gpu", tessera.arch = "apple7"} {
  func.func @apple_norm_int32_extent(%x: tensor<65536x32768xf16>) -> tensor<65536x32768xf16> {
    %0 = tessera.rmsnorm %x : (tensor<65536x32768xf16>) -> tensor<65536x32768xf16>
    return %0 : tensor<65536x32768xf16>
  }
}
