# SM120 saved-LSE forward: mixed Q vs K/V storage

Owner: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Sync: SM120-SAVED-LSE-MIXED-STORAGE-20261025 (extends SM120-SAVED-LSE-STORAGE-20261016).
Contract: `docs/spec/SM120_SAVED_LSE_STORAGE_CONTRACT.md` §"Mixed Q vs K/V storage".

## What changed

Before this change, `ts.jit(target="nvidia_sm120")` handed a saved-LSE
`flash_attn` whose K/V storage differs from Q to eager Python. It recorded no
fallback reason. Measured on The-Super-Bear at `93c9e55e2`: f16 Q with bf16 K/V,
and f32 Q with f16 K/V, both returned results with no native receipt.

Those calls now lower natively. The route is Graph → native checkpoint →
Schedule → Tile → NVIDIA Target → PTX → the retained C++ owner. The storage
contract is:

| Role | Storage |
|---|---|
| Q | f16, bf16 or f32 |
| K and V | one shared storage: f16, bf16 or f32, distinct from Q |
| O | Q's storage, which is the frontend result type, or f32 |
| Accumulation | f32, declared |
| Row-LSE | f32 |

Every operand widens exactly to f32.

The new field and its effects:
- The contract carries `kv_storage` only when K/V storage differs from Q.
- Homogeneous contracts, checkpoint identities, entry names and Tile kernels are
  byte-unchanged.
- Mixed entries are named `tessera_tile_attention_lse_<q>_kv_<kv>_[out_<o>_]<digest>`.
- The retained owner is reached through the new
  `tessera_nvidia_attention_forward_prepare_kv`. The original C entry is now a
  homogeneous wrapper with an unchanged ABI.

Still homogeneous-only, refusing a mixed forward: the backward checkpoint,
paired AD, original-source reverse, and the JVP owners.

## Host and protocol

- Device: The-Super-Bear, RTX 5070 (sm_120, cc 12.0), WSL2, CUDA 13.4, using the
  `build-nvidia-cuda` tree.
- Recording: branch `claude/saved-lse-mixed-storage`, clean tree. `packet.json`
  is from `1f8cdbd65`; `paired-interleaved.json` is from `95b35714b`. The
  native source is identical between the two.
- Correctness: every row is checked against an independent fp64 oracle, from
  the exact stored operands, before any timing. O is held to one rounding of
  its storage plus 2e-6, and LSE to rtol/atol 3e-6.

## Results

- **Correctness: `packet.json`, 72 rows.** Nine storage pairs (three
  homogeneous controls and six mixed), times two shapes, causal and noncausal,
  with and without broadcast f32 bias. Every row passes the oracle on both host
  and resident inputs. Every mixed row runs a `_kv_` entry, and no homogeneous
  row does.
- **Compiler-free replay: `replay.json`.** All 72 serialized packages relaunch
  with the compiler and subprocesses forbidden, and pass the same oracle.
- **Timing: `paired-interleaved.json`, diagnostic.** All eight arms alternate
  ABAB/BABA for 40 rounds in one process after an fp64 check. Median CUDA
  events, in µs:

| Q / K/V | 1×2×1×3×5×4×3 | 1×8×2×128×512×64×64 | vs homogeneous-Q control (large) |
|---|---:|---:|---:|
| f16 / f16 | 47.1 | 6164.8 | — |
| f16 / bf16 | 46.5 | 6163.1 | −0.0% |
| f16 / f32 | 44.1 | 6101.5 | −1.0% |
| bf16 / bf16 | 43.6 | 6205.3 | — |
| bf16 / f16 | 47.2 | 6219.3 | +0.2% |
| f32 / f32 | 44.8 | 5882.1 | — |
| f32 / f16 | 46.7 | 5925.8 | +0.7% |
| f32 / bf16 | 41.5 | 5944.1 | +1.1% |

Mixed storage costs nothing measurable against its homogeneous-Q control.

**Recorded so it is not misread.** `packet.json`'s own event medians show
homogeneous half rows about 3× faster than every mixed and f32 row at the tiny
shape. That gap is an artifact of recording order: the recorder times its rows
one after another, and the homogeneous-half rows ran first. Two checks rule out
the code:
- The generated PTX has the same structure across all nine pairs: 184–190
  instructions, with the same global loads, branches, divides and `ex2`. The
  pairs differ only by up to six `cvt` widenings.
- Under paired interleaving the gap disappears, as the table above shows.

`packet.json`'s completed-call medians are wall-clock figures and carry the
same ordering effect.

## Claim limits

- This is a correctness and capability slice. The kernel is the scalar
  correctness-first materializer, about 6 ms at the larger shape, and there is
  no performance promotion.
- CUDA events alone are diagnostic on sm_120 and are not an admitted timing
  witness.
- RTX 5070 only. No ROCm, Apple or x86 result follows. Those backends have no
  saved-LSE checkpoint consumer for this contract.

## Tests on this host

- **`tests/device/nvidia/test_saved_lse_storage.py`:** 108 passed, zero skips.
  That is 36 homogeneous cases plus 72 mixed cases, each covering the public,
  resident and serialized byte boundaries.
- **`tests/unit/test_saved_lse_storage_contract.py`:** 116 passed. It also
  passes 116/116 on the Mac.
- **Full `tests/device/nvidia/` directory:** 239 failures, 5500 passed. It was
  run in parallel (`-n 8`), and the 2026-10-24 baseline came from the release
  gate's device layer, not this directory.
  - Against that 111-test baseline, 135 failures are extra and 7 baseline
    failures now pass.
  - The 135 extras are not new. Rerun one file per process, they fail with
    identical counts on the branch and on `main` (`93c9e55e2`, rebuilt in the
    same tree). That includes the same segfault in `test_bounded_lhs_jit.py`.
  - They are recorded here as the full-directory pre-existing set.
