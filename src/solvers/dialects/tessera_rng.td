//===- tessera_rng.td ----------------------------------------*- tablegen -*-===//
// Tessera RNG dialect: counter-based RNG + distributions + QMC sequences.
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TRNG_Dialect : Dialect {
  let name = "trng";
  let cppNamespace = "::tessera::trng";
}

def TRNG_CreateStateOp : Op<TRNG_Dialect, "create_state", [NoMemoryEffect]> {
  let summary = "Create a RNG state (algorithm, seed, stream, substream)";
  let arguments = (ins AnyType:$seed);
  let results   = (outs AnyType:$state);
  let assemblyFormat = "$seed attr-dict `:` type($seed) `->` type($state)";
}

def TRNG_UniformOp : Op<TRNG_Dialect, "uniform", [NoMemoryEffect]> {
  let summary = "Uniform [0,1) draws";
  let arguments = (ins AnyType:$state, AnyType:$shape);
  let results   = (outs AnyType:$out);
  let assemblyFormat = "$state `,` $shape attr-dict `:` type($state) `,` type($shape) `->` type($out)";
}

def TRNG_NormalOp : Op<TRNG_Dialect, "normal", [NoMemoryEffect]> {
  let summary = "Normal(μ, σ) draws";
  let arguments = (ins AnyType:$state, AnyType:$mu, AnyType:$sigma);
  let results   = (outs AnyType:$out);
  let assemblyFormat = "$state `,` $mu `,` $sigma attr-dict `:` type($state) `,` type($mu) `,` type($sigma) `->` type($out)";
}
