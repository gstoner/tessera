# What every `math.*` op in a row-program kernel costs in accuracy

Owner: W4-PRODUCT-1 / AD-SOLVER-IFT-1 (sync `ROW-PROGRAM-MATH-AUDIT-2026-09-16`).
`record_row_program_math_precision.py` run on each owning device.

## Why this packet exists

On 2026-09-16 `math.sqrt` was found to reach libdevice's *approximate*
`__nv_sqrtf` on the NVVM route — the precise branch is gated on a reflect value
MLIR never sets — and one row of the row-normalization proof was 1 ulp off on
sm_120. That op was pinned to the rounding-explicit `__nv_fsqrt_rn`. **Every
other `math.*` op on both device routes is subject to the same vendor default**,
and the emitter passed them all through, so a result could be called exact that
nobody had measured. The emitter now admits a closed set, and this packet is the
measurement behind the `measured` plan in it.

## Measured — 16384 points per input domain, device against numpy f32

| op | plan | gfx1151 | gfx1201 | sm_120 |
|---|---|---|---|---|
| `math.sqrt` | rounding_explicit | **0 ulp** | **0 ulp** | **0 ulp** |
| `math.absf` | bit_exact | **0 ulp** | **0 ulp** | **0 ulp** |
| `math.cos` | measured | 1 ulp | 1 ulp | 1 ulp |
| `math.exp` | measured | 2 ulp | 2 ulp | 3 ulp |
| `math.log` | measured | 3 ulp | 3 ulp | 3 ulp |

`sqrt` at 0 ulp on sm_120 **is** the pin working: the same sweep against the
unpinned `__nv_sqrtf` is what produced the original 1-ulp row. The two RDNA
parts agree to the bit on every op (same ROCm 10.0 / HIP 7.15 toolchain);
sm_120's `exp` is one ulp looser than theirs.

The worst input is recorded per row, not just the bound — e.g. `log` on gfx1151
at 0.7044497 returns -0.3503383696 against the host's -0.3503382802.

## Two ops left the table, and why

| op | what the sweep found |
|---|---|
| `math.tanh` | Lowers to `__ocml_tanh_f32`, and the packaged image's kernel body was **one `s_endpgm`** — the launch succeeded, wrote nothing, and every element read back as zero, which looks like an answer. The identical module serialized with `format=isa` contains the correct 40-instruction implementation, so the body is lost in the *binary* path, not the lowering. |
| `math.log1p` | Lowers to `__ocml_log1p_f32` and the device computes `log(1 + x)`: `log1p(-1e-6)` returned -1.0132795e-06 against the host's -1.0000005e-06. Accuracy near zero is the only reason log1p exists, so admitting it would admit a trap. |

The first is why `build_native_gpu_storage` now disassembles the image it is
about to ship and refuses one whose kernel stores nothing. That guard covers the
AMDGPU image only: the NVIDIA cubin needs `nvdisasm`, not a matched-LLVM tool, so
an equivalent silent loss on the NVVM route would still ship — named in
`docs/audit/backend/nvidia/todo.md`, not papered over.

## Not claimed

* **No performance claim.** No timing is taken; `promotion_eligible` and
  `measured_performance` are false in every file.
* **Accuracy is per device and per toolchain.** A row says what *this* chip and
  *this* LLVM/ROCm/CUDA produced. A toolkit upgrade can move a ulp, and the
  recorder is what shows it (Decision #11).
* **The bounds do not retroactively weaken the EBM packets.** Those rows recorded
  `max_abs_error = 0` for their own inputs, which remains true at that scope;
  this packet is the wider statement the earlier rows never made.
* `f32` unary ops only. f64 `log`/`cos` in the Philox draw were measured
  bit-exact separately (the EBM native GPU packet) and are not re-swept here.

## Reproduce on an owning host

```bash
PYTHONPATH=python:. python benchmarks/record_row_program_math_precision.py --backend rocm --chip gfx1151 \
  --compiler build/tools/tessera-opt/tessera-opt --output <dir>/rocm_gfx1151.json
```
