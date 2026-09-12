# Apple arena f32 denormal policy

Owner: [NUMPOL-CARRIER-1](../audit/compiler/INTEGRATED_COMPILER_PLAN.md#numpol-carrier-1).
This is the bounded compiler-owned Apple arena consumer, not a promise about
all Metal, MPSGraph, scalar/vector dtypes or arbitrary Tessera operations.

The source module may carry `tessera.denormal_mode` with one of these values:

| Value | Consumed behavior |
|---|---|
| `unspecified` (also absence) | Existing native arithmetic; no gradual-underflow claim. |
| `gradual` | Integer-significand f32 add/subtract/multiply/divide, one round-to-nearest, ties-to-even step, preserved subnormal inputs/results. |
| `flush_to_zero` | Replace subnormal inputs with sign-preserving zero; compute with the same IEEE rounding core; replace subnormal results with sign-preserving zero. |

The emitter records `tessera.apple.denormal_mode` beside the generated MSL and
sizing companion. The package binds the complete artifact and shader identity;
its explicit policy marker must agree with that native record. Function/op-local
overrides, unknown policy values, fast-math flags and unimplemented floating
operations (including transcendental functions and comparisons) fail closed.
The CUDA/HIP generic storage builder refuses this currently Apple-only carrier
rather than silently discarding it. Other backend consumers require their own
implementation and exact-device proof.

Multiplication forms the exact product of two at-most-24-bit significands in
64 bits. Division first normalizes both significands, then obtains at least
32 quotient bits and a remainder sticky bit. Addition/subtraction aligns
normalized significands with 32 guard bits and a sticky shift; large exponent
gaps cannot cause cancellation of significant leading bits. Packing chooses
normal or subnormal units before rounding and carries a rounded significand
into the next exponent, including the minimum-normal and overflow boundaries.
No floating arithmetic occurs inside these helpers.

Signed zero and infinities follow IEEE arithmetic. Invalid zero/infinity
combinations return a quiet NaN; NaN operands are quieted. NaN payload selection
is not a portable cross-backend guarantee. No floating exception flags, dynamic
rounding mode, f64/vector arithmetic or performance promotion is implied.

The emitted helper source is tested as C++ against host IEEE arithmetic and
compiled unchanged as MSL on the owning Mac. The recorder uses boundary pairs
and random bit patterns, and checks signed zero as well as finite results and
NaN/infinity classification. Both policies require independent numerical gates;
fast-math-off and native hardware FTZ alone are insufficient evidence.
