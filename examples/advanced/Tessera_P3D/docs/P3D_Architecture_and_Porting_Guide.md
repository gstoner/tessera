<!-- MERGE-START: P3D_Architecture -->
# P3D Architecture & Porting Guide → Tessera

## Model Blocks
- **Conv3D Pyramid**: `p3d.pyramid.build` + `p3d.downsample3d` + `p3d.upsample3d`
- **Global Context (3D)**: `p3d.global_context` (block‑sparse attention/GEMM path)
- **Boundary Conditions**: `p3d.conv3d` `bc` attr (`periodic|dirichlet|neumann`)

## IR Mapping
- Graph IR: tensor ops with shapes `[B,C,D,H,W]`.
- Schedule IR: fusion (conv→norm→act), tile‑sizes, pipeline stages.
- Tile IR: 3D tiles with halos; warp reductions; async copies.
- Target IR: WGMMA/WMMA, TMA/async, LDS/DS ops.

## Passes & Pipelines
- `-tessera-autotune-p3d`: attach search spaces (tile, threads, pipeline).
- `-tessera-lower-p3d`: legalize to Tile→Target IR (tensor core paths).

**Example**:
```bash
tessera-opt model.mlir -tessera-autotune-p3d -tessera-lower-p3d -o lowered.mlir
```

## Porting Tips
- Use channels‑last memory for conv3d when possible.
- Prefer power‑of‑two tiles; align D/H/W to vector width.
- Attach `bc` early for halo‑infer to be effective.
<!-- MERGE-END: P3D_Architecture -->
