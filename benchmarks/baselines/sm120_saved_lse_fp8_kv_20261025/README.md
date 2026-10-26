# SM120 saved-LSE forward: unscaled FP8 K/V storage

Owner: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Sync: SM120-SAVED-LSE-FP8-KV-20261025. This extends SM120-SAVED-LSE-MIXED-STORAGE-20261025.
Contract: `docs/spec/SM120_SAVED_LSE_STORAGE_CONTRACT.md` §"Unscaled FP8 K/V storage".
Recorded by `benchmarks/nvidia/diagnose_saved_lse_mixed_pairing.py --fp8`.

## What changed

Before this change, a saved-LSE `flash_attn` on sm_120 with `fp8_e4m3` or
`fp8_e5m2` K/V ran eager Python, just as mixed storage had before #1012. Its
Graph also lacked `head_dim`, because the render rule only inferred it when
every operand was f16/bf16/f32/f64.

That call now lowers natively. K and V share one FP8 storage, and Q is f16,
bf16 or f32. FP8 is unscaled: the stored values are the operands.

**Decode.** The kernel loads one byte per element and decodes it with sm_120's
`cvt.rn.f16x2.{e4m3,e5m2}x2`, through `nvvm.convert.f8x2.to.f16x2`. It then
widens to the declared f32 accumulation. Every value of both formats is exact
in f16, so the result is exact.

**Storage roles.**
- O is Q's storage or f32. Row-LSE is f32.
- Backward, paired AD and JVP stay homogeneous.
- Resident FP8 roots are refused by the shared resident-tensor contract.
- A scaled KV cache (K = k_fp8 · k_scale) is a different, unadmitted contract.

## Host

- Device: The-Super-Bear, RTX 5070 (sm_120), WSL2, CUDA 13.4, `build-nvidia-cuda`.
- Branch `claude/saved-lse-fp8-kv`, clean tree.
- Final kernel: `2373cacc3`.

## Correctness

**Exhaustive decode.** `tests/device/nvidia/test_saved_lse_storage.py::test_fp8_kv_decodes_every_finite_code_exactly`
gives every finite code point of each format its own batch element with a
single key, so its softmax weight is exactly 1.
- O must equal the decoded V code, and row-LSE must equal 0.5·k.
- Both are checked bit-exactly against ml_dtypes' f32 value.
- This covers every subnormal, both signed zeros, and ±448 (E4M3FN) or
  ±57344 (E5M2).
- It passes with the software decode and again with the hardware decode.

**Random cases.** 72 FP8 cases cover Q ∈ {f16, bf16, f32}, both formats,
Sk ∈ {5, 129}, no/broadcast/dense bias, and causal/noncausal.
- The public host call and a serialized `rt.launch` with guard bytes are both
  held to the fp64 oracle.
- A resident FP8 call must raise.

**Totals on this host.**
- The saved-LSE device file passes 182: 108 existing, 72 FP8 and 2 exhaustive.
- `tests/unit/test_saved_lse_storage_contract.py` passes 135 here and on the Mac.

## Timing (diagnostic: CUDA events, paired and interleaved, after an fp64 check)

`paired-interleaved.json` (`2373cacc3`, the final kernel) at
1×8×2×128×512×64×64. Medians in µs, with the ratio to the homogeneous-Q control:

| Q | homogeneous | Q / fp8_e4m3 | Q / fp8_e5m2 |
|---|---:|---:|---:|
| f16 | 6157.1 | 6426.0 (1.044) | 6412.2 (1.041) |
| bf16 | 6196.0 | 6464.1 (1.043) | 6463.7 (1.043) |
| f32 | 5856.8 | 6085.9 (1.039) | 6087.2 (1.039) |

At 1×2×1×3×5×4×3 every arm lands between 48 and 56 µs, so launch latency
dominates and no storage effect is resolvable.

### How the decode was chosen

Each step was measured the same way, with each run's own homogeneous control:

| Decode | Run | E4M3FN / control | E5M2 / control |
|---|---|---:|---:|
| Software E4M3FN (≈15 integer/select ops), E5M2 by `<<8` into f16 | `eee867226`, file overwritten on the box | 1.59 (9853 vs 6193 µs, f16 Q) | 1.09 |
| Hardware E4M3FN, E5M2 still by shift | `5ceb23b0e`, `paired-interleaved-e5m2-shift-decode.json` | 1.04 | 1.08 |
| Hardware for both formats | `2373cacc3`, `paired-interleaved.json` | 1.04 | 1.04 |

The first row's numbers come from that session's output. Its JSON was
overwritten by the next run, before it was copied into this packet.

The kernel is the scalar correctness-first materializer, about 6 ms at this
shape, so these numbers are not a performance claim. CUDA events alone are
diagnostic on sm_120. The remaining ~4% has not been attributed.

## Other checks on this host

- **Full `tests/device/nvidia` sweep (`-n 8`)** against the #1012 branch run:
  - 3 new failures, all parametrizations of
    `test_source_reverse_waits_pending_writes_and_recovers_after_rejection`.
    Run serially, all 12 of its parametrizations pass, twice. They are
    eight-way GPU contention failures.
  - 0 healed.
- **`tests/device/nvidia/test_lse_cotangent_native.py::test_lse_cotangent_tile_contract[...requires f32]`**
  fails here and on `main`. It is in the pre-existing set from #1012.
- **IR lit in `build-nvidia-cuda`:** 441 passed, 124 unsupported, 0 failed.
  - This needs a shell without the device-lane env. With
    `TESSERA_OPT`/`TESSERA_NVIDIA_OPT` exported, lit detects fewer features and
    reports 414 passed and 151 unsupported, still with 0 failed.
- **`check-tessera-nvidia`:** 72/72.
- **Mac:** `tests/unit -k "attention or attn or checkpoint or lse or graph_ir or flash"` shows no failures outside the Mac `main` baseline.

## Claim limits

- RTX 5070 only, host roots only, unscaled FP8 only.
- No ROCm, Apple or x86 result follows. gfx1201 FP8 attention is a separate
  backend with its own owners.
