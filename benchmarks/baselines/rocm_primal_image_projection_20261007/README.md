# Native primal physical projection A/B, gfx1201

Owning RX 9070 XT / gfx1201, matching LLVM/MLIR 23.1.1.
Native member packaging now runs the existing verified Target image-identity
projection, while codegen exports launch dimensions from the original typed
SSA function contract. The complete program witness remains on the package;
launch-only SSA metadata is excluded from physical kernel-name hashing.

Twenty-one ordinary FP8/MXFP8/JVP tests pass. Native core: 492 pass, 66
unsupported. Shared frontend/registry gates: 483 pass. Twelve package
projection tests pass, including byte-identical images for M17/M18 with
different launch scalars and output capacities.

Eight paired rows cover FP8/MXFP8, KN/NK, and M17/N19/K256 or M200/N129/K1536.
Every arm passes an independent float64 block oracle before timing.
Projected arms also pass bitwise public-owner parity. Each arm uses the same
native C++ owner and geometry; 21 alternating-order windows contain 100
native invocations each. Duplicate projected/static image controls stay
within approximately 1.4% of unity. Event windows include native enqueue
gaps; these are not isolated-kernel measurements or public speedups.

Projection has profile-specific effects:
- Large MXFP8 NK: static 0.507 ms -> projected 0.043 ms (ratio 0.085).
- Small MXFP8 NK: ratio 0.489.
- Large FP8 KN: static 0.032 ms -> projected 0.523 ms (ratio 16.526).
- Small FP8 KN: ratio 2.288.
- Other four measured profiles: ratios approximately 0.999-1.013.

A uniform projection policy is not performance closure. Next: native
profile-specific policy, a wider measured shape envelope, public-call timing,
and FP8/MXFP8/MXFP4 evaluation before promotion. General cache families,
batching/transpose closure, sibling physical proof and aggregate publication
remain open. Historical regression packets remain unchanged.
