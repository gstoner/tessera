# ROCm batch-offset foundation

Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.

The native gfx1201 fixture carries dynamic rank-one memref offsets through
tile.view into E4M3 typed fragment registers and accumulator stores.
The unbounded path retains offset-backed vector loads; the ragged path
retains offset-backed guarded scalar loads. Target-to-ROCDL lowering passes.

This is host WSL compiler evidence only. It establishes that the existing
materializer can consume per-batch offset views. Independent-RHS/shared-LHS
Graph/Schedule/codegen projection, matrix/scale batch strides, image identity,
checked package ABI, owning-device numerics and benchmarking remain open.
The fixture does not establish allocation capacity or launch lifetime.

Source/compiler fingerprints and the raw lit results are recorded here.

Validation: 84 ROCm backend lit fixtures pass and 11 audit-document checks pass in host WSL.
