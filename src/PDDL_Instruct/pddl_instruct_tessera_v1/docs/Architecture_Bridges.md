<!-- MERGE-START: Architecture_Bridges.md -->
# Architecture Bridges: WMMA/WGMMA Feasibility & cp.async/TMA Heuristics

This note operationalizes the architecture-specific predicates in your Domain Modeling and CoT docs.

## WMMA / WGMMA Feasibility
- **WMMA** (Ampere): prefers tiles like m16n16k16, m32n8k16. Require divisibility of M,N,K by native tile sizes.
- **WGMMA** (Hopper): prefers m64n256k32, m128n256k32. Check `K % 32 == 0`; map `N` to 256 multiples.
- **Policy**: If problem dims are near-feasible (< 10% padding), allow pad-and-trim and echo this in CoT.

## cp.async (Ampere) vs TMA (Hopper)
- **cp.async**: Best for 16–64KB stages; deeper pipelines improve hiding but increase register pressure.
- **TMA**: Best for large, regular tiles (>=64KB transfers). Prefer when shared memory ≥ 96KB available and layout is contiguous.
- **Heuristic**:
  - If shared_mem_stage_bytes in [16KB, 96KB) and regs/thread < 192 → cp.async
  - If shared_mem_stage_bytes ≥ 96KB or multi-dimensional swizzle → TMA
  - Always echo decision + expected overlap (0.2–0.4) into CoT `estimates`.

## Cluster Mode (Hopper)
- Use 2×2 clusters when S or M dims exceed single-SM residency; verify cross-SM shared memory availability.
<!-- MERGE-END: Architecture_Bridges.md -->
