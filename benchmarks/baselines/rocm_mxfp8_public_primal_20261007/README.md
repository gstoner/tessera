# Ordinary typed MXFP8 primal JIT, gfx1201

Owning RX 9070 XT / gfx1201; matching LLVM/MLIR 23.1.1 compiler.
Four MXFP8 KN/NK cases cover M17/N19/K256 and M200/N129/K1536.
Independent float64 K32 block oracle, changed E8M0 scales and warm
compiler-subprocess refusal pass. Combined FP8/MXFP8/scale-JVP device lane:
21 passed. Shared frontend/dtype/diagnostic gates: 472 passed.
Three native unsigned/signless/invalid scale fixtures pass.

The explicit Dtype("uint8", allow_planned_gated=True) Tensor declaration
serializes planned_gated status into actual tracer Graph arguments. Bare uint8
annotations remain refused; no general unsigned arithmetic promotion.
Original Graph -> native Schedule -> Tile -> ROCm Target -> LLVM -> HSACO
reaches the checked descriptor runtime. Source/tool hashes bind this packet.

timings.json separates public calls with descriptor staging/readback from
diagnostic native-owner event windows of the identical image. Event windows
include enqueue gaps; neither isolated kernel speedup nor production native
primal program projection is claimed.

Open: generic batching, transpose-left, composed AD, E8M0 eager differential
reference, native primal owner integration and sibling physical parity.
