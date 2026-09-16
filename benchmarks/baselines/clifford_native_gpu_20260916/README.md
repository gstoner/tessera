# Clifford product family through the native GPU storage route

Owner: W6.4 (sync `GA-NATIVE-GPU-2026-09-16`). `record_clifford_native_gpu.py`
executed independently on three owning devices; `source-hashes.json` binds the
implementation. Packets are correctness evidence; no performance is measured
or claimed, and gfx1151 / gfx1201 / sm_120 proofs never transfer.

| Packet | Host | Device | Rows | Worst abs error |
|---|---|---|---|---|
| `rocm_gfx1151.json` | Princess-Luna (WSL2, ROCm 10.0) | gfx1151 | 30 | 0 |
| `rocm_gfx1201.json` | Tajasarus (WSL2, ROCm 10.0, `TESSERA_ROCM_CHIP=gfx1201`) | gfx1201 | 30 | 0 |
| `nvidia_sm120.json` | The-Super-Bear (WSL2, CUDA 13.4 / driver 610.88) | sm_120 (RTX 5070) | 30 | 2.4e-07 |

Each row is one op of the family (geometric product, wedge, left contraction,
inner, norm, reverse, grade involution, conjugate, Hodge star, rotor sandwich)
at one shape (`8`, `37x8`, `3x5x8`, Cl(3,0), f32) compared with the standalone
GA reference. `emitted_products` counts the `arith.mulf` ops in the arena IR
of the geometric product: 64 for the full product and 24 under a grade-2
restriction — the pruning is in the device code the compiler emitted, and the
packet also checks the pruned product equals the grade-2 projection with the
other coefficients written as zero.

How the kernel is produced (no Python-emitted arithmetic): the recorder writes
only a kernel skeleton (one thread per multivector; loads, a rank-1
`tessera_clifford` op on tensors, stores). `ts-clifford-opt` expands the op
through the same GradeFusion + ExpandProductTable lowering the CPU JIT runs,
the arena pipeline's canonicalization folds the tensors away, and
`build_native_gpu_storage` packages the scalar kernel for the device. The
Python-emitted `rocm_clifford_compiled` / `x86_clifford_compiled` / Apple
kernels are untouched and remain their lanes until measured against this
route.

Reproduce on an owning host (Clifford backend configured ON, toolkit env sourced):

```bash
PYTHONPATH=python:. python benchmarks/record_clifford_native_gpu.py --backend rocm --chip gfx1151 \
  --compiler build/tools/tessera-opt/tessera-opt --output <dir>/rocm_gfx1151.json
```
