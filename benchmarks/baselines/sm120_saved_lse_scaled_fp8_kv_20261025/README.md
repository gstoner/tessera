# SM120 saved-LSE forward: scaled FP8 K/V (dequantize + fusion)

Owner: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Sync: SM120-SAVED-LSE-SCALED-FP8-KV-20261025, which extends SM120-SAVED-LSE-FP8-KV-20261025.
Contract: `docs/spec/SM120_SAVED_LSE_STORAGE_CONTRACT.md` §"Scaled FP8 K/V" and
`docs/spec/PYTHON_API_SPEC.md` §dequantize_scaled.
Recorded by `benchmarks/nvidia/diagnose_saved_lse_scaled_kv_pairing.py`.

## What it is

The public form is a composition:

```python
k = ts.ops.dequantize_scaled(k_codes, k_scale)
v = ts.ops.dequantize_scaled(v_codes, v_scale)
o, lse = ts.ops.flash_attn(q, k, v, lse_checkpoint="saved")
```

**`tessera.dequantize_scaled`** is a new Graph op with ODS, a verifier, an eager
reference and a declared shape rule. It computes f32(code) · scale, rounded once
to f32. Codes are FP8; the scale is f32, and each scale axis is 1 or the codes'
extent.

**Fusion on SM120.** The checkpoint importer fuses the chain into the new
`tessera_attn.checkpoint_forward_scaled_kv`. The kernel decodes each FP8 code
exactly with the sm_120 f8x2 conversion, then multiplies it by its broadcast
scale in registers. No dequantized K/V is ever stored.

**Ownership.** Everything runs through the checked descriptor launch under new
`scaled_kv` ABI ids. AD, JVP, resident FP8 roots and dynamic sequences are not
admitted.

## Host

- Device: The-Super-Bear, RTX 5070 (sm_120), WSL2, CUDA 13.4, `build-nvidia-cuda`.
- Branch `claude/saved-lse-scaled-fp8-kv`, clean tree; final kernel `539b9cafc`.

## Correctness

### `tests/device/nvidia/test_saved_lse_scaled_kv.py`: 182 passed

**180 parametrized cases.** The grid is:
- Q storage: f16, bf16, f32;
- K/V format: E4M3FN, E5M2;
- scale layout: per-tensor, per-head, per-token, per-channel, per-element;
- key length Sk: 5 or 129;
- variant: causal, noncausal, broadcast bias.

Each case checks:
- the public JIT call against an fp64 oracle computed from the
  `dequantize_scaled` reference's f32 K/V, the declared semantics;
- a native receipt and a `_scaled_` entry;
- a serialized `rt.launch` replay with guard bytes, bit-identical to the public
  result.

**2 exact cases.** Every finite code of each format is the only key of its own
batch element, so its softmax weight is exactly 1. The test draws a random f32
scale per element and requires:
- O to equal f32(v_code) · v_scale bit for bit;
- row-LSE to equal 0.5 · f32(k_code) · k_scale bit for bit.

### Other tests

- `tests/device/nvidia/test_saved_lse_storage.py`: 182 passed, unchanged.
- `tests/unit/test_saved_lse_scaled_kv_contract.py`: 69 passed. It covers
  fusion for every combination, refusals, and descriptor order and scalars.
  The 69 plus 135 storage contract tests give 204 that pass on Tajasarus's
  assertions-ON LLVM.

## Timing (diagnostic)

Every arm is compiled by the public JIT and checked against the fp64 oracle.
All arms are then timed through the same launcher benchmark entry: CUDA events
around 10 launches after 2 warmup. The run uses 30 ABAB/BABA rounds and reports
the median per launch. Shape 1×8×2×128×512×64×64:

| Arm (Q / K/V) | µs | vs f16 homogeneous | vs unscaled FP8 |
|---|---:|---:|---:|
| f16 / f16 | 6143.7 | 1.00 | — |
| f16 / e4m3 unscaled | 6413.6 | 1.04 | 1.00 |
| f16 / e4m3 per-tensor | 6700.6 | 1.09 | 1.04 |
| f16 / e4m3 per-token | 6889.3 | 1.12 | 1.07 |
| f16 / e4m3 per-element | 8280.2 | 1.35 | 1.29 |
| f16 / e5m2 unscaled | 6413.7 | 1.04 | 1.00 |
| f16 / e5m2 per-token | 6889.5 | 1.12 | 1.07 |

Per-element scaling also reads a full f32 scale per element, four times the
bytes of the FP8 codes. That extra traffic is where its additional cost comes
from.

At 1×2×1×3×5×4×3 every arm lands between 9 and 12 µs; launch latency
dominates, and no scaling effect resolves.

**A defect found and fixed by this measurement.** The first kernel
recovered each element's (b, h, s, d) from its flat index with 64-bit
divide/remainder by runtime extents. That made per-token and per-element
scaling about 6.4× slower: 40994 µs and 42720 µs against 6414 µs unscaled.
Per-tensor stayed cheap at 6700 µs, because its unused coordinates were
eliminated.

The fix, `539b9cafc`, indexes scales from the coordinates the call sites
already hold, so no integer division runs per element. Those pre-fix numbers
come from that session's output; its JSON was overwritten before it was copied
here.

The kernel is the scalar correctness-first materializer, so these numbers are
not a performance claim. CUDA events alone are diagnostic on sm_120.

## Other checks on this host

**Full `tests/device/nvidia` sweep (`-n 8`)** compared with the FP8 K/V
branch run:
- 1 new failure and 2 healed. All three are parametrizations of
  `test_source_reverse_waits_pending_writes_and_recovers_after_rejection`, the
  known GPU-contention case.
- Run serially, all 12 of its parametrizations pass.
- No other difference.

**Lit:**
- IR in `build-nvidia-cuda`: 441 passed, 124 unsupported, 0 failed. This run
  used a shell without the device-lane env.
- `check-tessera-nvidia`: 72/72.
- Mac IR: 501 passed, 64 unsupported, 0 failed.

**Mac unit subset** (attention / checkpoint / quant / catalog / registry /
coverage / manifest / capability / audit / route): no failures outside the Mac
`main` baseline.

## Registries this op touches

- **Op definition and verifier:** ODS (`TesseraOps.td`), the verifier
  (`TesseraOps.cpp`), and the Attn sibling op.
- **Op registration:** the op catalog with its shape rule `dequantize_scaled`,
  the public op and eager reference (`tessera.__init__`), and `PYTHON_API_SPEC`.
- **Capability and manifest:** the capability table (`artifact_only` on SM120,
  `unsupported` elsewhere), and the backend manifest. Its SM120
  `device_verified_jit` row names this device test as the execute-compare
  fixture.
- **Generated:** the primitive route map family `attention_scaled_kv` and the
  regenerated dashboards.

## Claim limits

- RTX 5070 only.
- Host roots only.
- Static sequences only.
- Forward only.

No ROCm, Apple or x86 result follows. Those backends refuse the chain or run
the eager reference.
