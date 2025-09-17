<!-- MERGE-START: P3D_Overview -->
# P3D in Tessera — Overview (Science & Engineering)

**Scope**: Scalable neural surrogates for high‑resolution 3D physics (turbulence, weather, multiphysics).

## Why P3D
- Replace costly PDE solvers at inference time with accurate surrogates.
- Capture **local details** (3D conv pyramid) + **global dependencies** (attention-like global context).
- Scale to **O(1000^3)** grids with domain decomposition + halo exchange.

## Where it maps in practice
- **CFD**: LES/DNS coarse→fine emulation, wall‑bounded turbulence, jet mixing.
- **Climate/Weather**: 3D downscaling, microphysics surrogates, assimilation proposals.
- **Subsurface**: Porous media flow, CO₂ plume modeling.
- **Medical/Imaging**: Volumetric reconstruction priors.

## Tessera advantages
- Multi‑level IR lowers to Hopper/Blackwell/MI300 tensor cores.
- Mesh + halo support for 3D domain decomposition.
- Built‑in numerics policies (bf16 compute / f32 accumulate) and deterministic modes.
<!-- MERGE-END: P3D_Overview -->
