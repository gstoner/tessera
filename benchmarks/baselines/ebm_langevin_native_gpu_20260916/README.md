# The EBM Langevin loop as one cooperative kernel, on the owning devices

Owner: W4-PRODUCT-1 / AD-SOLVER-IFT-1 (sync `EBM-NATIVE-GPU-2026-09-16`).
`record_ebm_langevin_native_gpu.py` executed independently on three owning
devices; `source-hashes.json` binds the implementation. Packets are
correctness evidence; no performance is measured or claimed, and gfx1151 /
gfx1201 / sm_120 proofs never transfer.

| Packet | Host | Device | Rows | Worst abs error |
|---|---|---|---|---|
| `rocm_gfx1151.json` | Princess-Luna (WSL2, ROCm 10.0) | gfx1151 | 6 | 0 |
| `rocm_gfx1201.json` | Tajasarus (WSL2, ROCm 10.0, `TESSERA_ROCM_CHIP=gfx1201`, assertions-ON driver) | gfx1201 | 6 | 0 |
| `nvidia_sm120.json` | The-Super-Bear (WSL2, CUDA 13.4 / driver 610.88) | sm_120 (RTX 5070) | 6 | 0 |

Each row is one K-step loop over the quadratic energy `0.5·Σ(x − y)²` at one
`(shape, K, T)` — `(4×8, 1, 0.7)`, `(6×8, 5, 0.7)`, `(3×5, 12, 0.7)`,
`(9×100, 4, 0.3)`, `(16×33, 8, 0)`, `(2×1024, 3, 0.5)` — compared with
`native_langevin.reference_langevin_loop`, the numpy statement of the declared
Philox-4x32-10 / Box–Muller policy, and the returned key. `kernel_structure`
is read from the arena IR of the packaged kernel: one `gpu.func`, one
`scf.for` carrying `(state f32, key i64)` in registers (the loop-invariant key
word is hoisted by canonicalization), 18 `arith.mului_extended` Philox rounds
inside the loop, no barrier (the quadratic gradient is elementwise), and no
`linalg`/`tensor` op left.

How the kernel is produced (no Python-emitted arithmetic): one `tessera-opt`
invocation runs `--tessera-autodiff-paired` (the gradient),
`--tessera-ebm-canonicalize --tessera-ebm-lower-langevin` (the step with the
noise), `--tessera-to-linalg --inline --convert-elementwise-to-linalg
--canonicalize --cse` (a `[rows, features]` row program in linalg) and
`--tessera-row-program-to-gpu` (one block per row, one lane per feature,
the K-step loop in registers, ordered shared-memory reductions); the tensor
contract is attached, `build_native_gpu_storage` packages it and the native
storage binding launches it. The emitter's reduction path is exercised by the
row-normalization device test in `tests/unit/test_ebm_native_langevin_gpu.py`
(bit-exact with the sequential f32 fold on the same three devices), not by
these rows.

Reproduce on an owning host (EBM backend configured ON, toolkit env sourced):

```bash
PYTHONPATH=python:. python benchmarks/record_ebm_langevin_native_gpu.py --backend rocm --chip gfx1151 \
  --compiler build/tools/tessera-opt/tessera-opt --output <dir>/rocm_gfx1151.json
```
