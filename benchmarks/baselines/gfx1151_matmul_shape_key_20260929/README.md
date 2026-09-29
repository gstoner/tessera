# gfx1151 scheduled matmul image key, 2026-09-29

Owner: E2E-REAL-6. Sync: E2E-REAL-6-ROCM-MATMUL-IMAGE-2026-09-29.

The packet was recorded on Princess-Luna, live/configured gfx1151, from clean source commit `683b9c410c4cb8a208a8e8be6c6026dd9bfeba76` with a freshly rebuilt full `tessera-opt` using LLVM/MLIR 23.1.1. The JSON records the compiler SHA256, distinct Schedule/Tile digests, checked image identity, 31 synchronized WSL host-wall launch samples per shape, and NumPy error. No device-kernel timing or promotion authority is claimed.

| f16 MxKxN | New image state | Package ms | Tile-text control | Control compile ms | Max abs error |
| --- | --- | ---: | --- | ---: | ---: |
| 16x16x16 | cold | 627.9 | cold | 454.1 | 5.96e-8 |
| 32x16x16 | warm cache | 78.6 | cold | 458.6 | 8.94e-8 |
| 48x16x16 | warm cache | 78.1 | cold | 454.3 | 5.96e-8 |

All three new packages bind the same `714609af...d8cb` image digest and symbol. The historical Tile-text compiler path compiles all three shapes, though its payload hashes are identical. The first new package pays an extra Tile-to-Target and directive projection step; later shapes still spend about 78 ms in Schedule/Tile replay. The control is compile-only, so these figures do not compare end-to-end execution performance.

The image identity drops only shape-bound host scaffolding and the checked Schedule digest from the Target directive. Physical panel, dtype, architecture, numeric policy and compiler/toolchain inputs remain in the key. This optimization is admitted only for static, unfused, unsplit gfx1151 f16/bf16 register matmul with K unroll 1. Exact-device tests cover three f16 and three bf16 shapes, with an independent NumPy oracle. gfx1201, fused, split-K, dynamic shapes and other physical routes remain open.

Reproduce on Princess-Luna with a source-matched full compiler and HIP environment:

```bash
source scripts/_rocm_env.sh
PYTHONPATH=python:. TESSERA_OPT=/path/to/rebuilt/tessera-opt TESSERA_ROCM_CHIP=gfx1151 \
  python benchmarks/rocm/record_gfx1151_matmul_shape_key.py
```

Post-review raw-launch validation: the E2E-REAL-4 benchmark resolves the scheduled entry from its package descriptor. Princess-Luna gfx1151 passed NumPy checks for 64x64x64 (maximum absolute error 1.94e-7) and ragged 65x67x31 (8.94e-8). Its 64-cube timing run returned a reject verdict against the canonical throughput floor; that small shape is correctness evidence only.
