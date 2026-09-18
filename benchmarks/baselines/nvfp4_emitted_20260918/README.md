# Emitted sm_120a NVFP4 block-scale warp tile — exact-operation packet (2026-09-18)

Recorded by `benchmarks/record_nvfp4_emitted.py` on The-Super-Bear (RTX 5070,
sm_120, WSL2, CUDA 13.4 toolkit, driver 610.88). Sync `GFX1201-PARITY-2026-09-17`,
NVIDIA owed item "NVFP4 emitted kernel as a consumer".

The compiler-emitted `tessera_nvfp4_mma_m16n8k64` kernel
(`ptx_emit.emit_nvfp4_block_scale_mma_ptx`, one warp, `mma.sync ... m16n8k64
... kind::mxf4nvf4.block_scale.scale_vec::4X`) registered through the launch
bridge under its own entry (`invokeNvfp4Emitted`) with operands laid out per
lane by `compiler/nvfp4_fragments.py`, compared bit-for-bit against
`nvfp4_tile_reference` (e2m1 and ue4m3 decoded as the on-silicon spike decodes
them, one scale per 16-wide K block) for the spike's unit, uniform 0.5 and 2.0
and mapped non-uniform scale modes plus a random-scale row. Every row is
exact.

`driver_jit_ptx_version` is the `.version` the driver JIT was handed after
`ptx_for_driver_jit` lowered the toolkit's stamp (driver 610.88 accepts PTX
<= 9.3). Correctness only: one fixed tile per launch with host transfers, no
timer, no general-shape dispatch, therefore no arbiter candidate.
