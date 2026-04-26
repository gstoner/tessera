# Neighbors & Halo (User Guide)

This guide shows how to use **neighbors**, **@halo**, **stencils**, **pipeline**, and **dynamic** features.

## Quick Start

```tessera
@neighbors 2d_mesh(dims=(1024,1024), default_neighbors=von_neumann) as grid

stencil lap2d() {
  taps   = von_neumann
  coeffs = { Δ(0,0):-4, Δ(+1,0):1, Δ(-1,0):1, Δ(0,+1):1, Δ(0,-1):1 }
  bc = dirichlet(0.0)
}

@pipeline(stages=3, double_buffer=true, reuse=lines, overlap=eager)
@apply stencil lap2d to u -> du @grid
```
- Halo width auto-inferred as `[1,1]`
- Async exchange overlaps with compute
- Line reuse reduces N/S transfers in a sweep

## Hex Lattices
```tessera
@neighbors hex_2d(dims=(Q,R), orientation=pointy, default_neighbors=hex_axial) as hex
stencil diffusion_hex() {
  taps = hex_axial
  coeffs = { Δ(0,0):-6, Δ(+1,0):1, Δ(-1,0):1, Δ(0,+1):1, Δ(0,-1):1, Δ(+1,-1):1, Δ(-1,+1):1 }
  bc = reflect
}
@apply stencil diffusion_hex to conc -> conc2 @hex
```

## Dynamic Topology
```tessera
@dynamic(topology_update=allowed, replan=auto, mask=refine_mask)
if refine_needed:
  grid = grid.refine(blocks=new_blocks)  # fences inserted by compiler
```
