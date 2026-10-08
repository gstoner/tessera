# Native scaled-primal image policy, gfx1201

Owning RX 9070 XT / gfx1201; matching LLVM/MLIR 23.1.1.
The native Target pass selects static ragged FP8 KN and static aligned FP8/
MXFP8 NK. Other profiles retain verified runtime projection and image reuse.
The actual member manifest records image_policy; typed SSA remains the source
of launch scalars, output capacities and ownership. Python orchestrates
compiler invocations and marshals the ABI without choosing physical policy.

refined-timings.json covers sixteen rows: four shapes, two formats and KN/NK.
Policy, forced-static and forced-projected arms plus identical-image controls
use native ownership and alternating order over 21 x 100 invocation windows.
Independent float64 block numerics pass before timing; policy arms also match
public outputs bitwise. Event windows include enqueue gaps. Controls are
generally near unity; the noisiest refined policy-control ratio is 0.980.

Named refined results:
- Ragged FP8 KN M200/N129/K1536: 0.031 ms, 0.060x projected arm.
- Ragged MXFP8 NK same shape: 0.043 ms, 0.085x static arm.
- Aligned FP8 NK M256/N256/K2048: 0.0085 ms, 0.464x projected arm.
- Short aligned FP8 KN M64/N64/K512 retains runtime image reuse at about
  5% kernel-window cost versus the static arm. This tradeoff is retained.

Twenty-one owning ordinary primal/JVP tests pass. Native core: 492 passed,
66 unsupported. Shared frontend/package/registry gates: 485 passed.
Public warm calls in public-fp8.json/public-mxfp8.json remain 0.87-4.54 ms:
native execution is not the whole public cost. Six FP8/MXFP8/folded-MXFP4
diagnostic staging rows pass numerical/bitwise/changed-scale checks.
Folded MXFP4 is a diagnostic existing-image route, not ordinary typed AD/JIT
coverage or Radiance attribution closure. Pinned staging remains opt-in.

refined-source-tools.sha256 binds the refined source, compiler and runtime.
timings.json is the earlier uniform-FP8-KN policy exploration; it predates the
refinement and is retained as historical evidence rather than current-source
proof. No universal policy, general cache-family closure, isolated-kernel
speedup, sibling parity or full-suite/publication completion is claimed.
