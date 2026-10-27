# Bounded-row native quantized MLP on gfx1201 (2026-10-27)

**Sync key:** `GFX1201-NVFP4-BOUNDED-MLP-20261027`. **Owner:** ROCM-NVFP4-INGEST-1.

**Host:** Tajasarus, an RX 9070 XT (gfx1201) on Ubuntu 26.04 under WSL2, with ROCm 10.0 / HIP 7.15. The compiler is the assertions-ON LLVM/MLIR 23.1.1 build (`build-assertions/`). Everything ran unsandboxed on the device box.

The seven-member NVFP4 gate/up → SiLU-mul → down program is compiled once at capacity M with schema `tessera.native.quantized_product.v5`. One native owner then serves any active M up to that capacity. Contract: `docs/spec/NATIVE_FP8_BLOCK_QUANT_FOUNDATION.md` §"Bounded-row v5 MLP".

## Correctness (device)

**`tests/device/rocm/test_native_quantized_mlp_bounded_rows.py`: 14 passed.**

The H/K/N shapes tested were:

| H/K/N | Capacity | Active M |
|---|---|---|
| 32/48/19 | 8 | 3, 8, 1, 5 |
| 16/16/1 | 1 | 1 |
| 128/12288/256 | 256 | 37, 256, 1 |

- **Bitwise equality.** Each bounded output equals the first rows of the static capacity program run over padded roots: same schedule, no M guard.
- **Independent oracle.** For M ≤ 64 each output also passes the independent staged chain (`assert_compiled_chain`) at the active shape. Caller roots are never mutated.
- **Weight residency.** `update_activations` changes M while every weight root stays the same. Every private row step then refuses with status 10. Running steps 0, 2–4 and 6, skipping both weight expansions, reproduces the static rows bitwise.
- **Refusals.** The owner refuses:
  - an active M above capacity, in both Python and native;
  - a row change that omits the activation root;
  - an activation span for different rows.
- **Derived frame admission.** Seven tampered frames are refused before HIP:
  - a missing row step;
  - an invariant step declared as a row step;
  - a missing row slot;
  - a wrong scalar scale, grid tile, grid axis or scalar index.

  The untampered frame prepares.

**Unit tests:** `tests/unit/test_native_quantized_mlp_bounded_rows.py` gives 12 passed (Mac and Tajasarus). They cover the v5 frame, the replay of `original_graph_ir`, malformed bounds, out-of-set operations, a bound on a smaller program, and Python frame re-derivation.

## Regression

- **Owner-related gfx1201 device files:** 204 passed. The quantized, NVFP4, program, owner, gate/up, block-quant and parent-view files include both v1 bounded-row files. The 12 skips were `test_source_reverse_program.py` behind its opt-in; run with `TESSERA_ROCM_SOURCE_REVERSE_DEVICE_PROOF=1`, it gives 12 passed.
- **Quantized MLP/product/gate-up/cache/checkpoint unit set on Tajasarus:** 59 passed.
- **`check-tessera-ir` on the assertions build:** 493 passed, 72 unsupported, 0 failed.

## Timing (diagnostic)

`paired.json` was produced by `benchmarks/rocm/diagnose_bounded_quantized_mlp.py` at commit `3d8af93c9` on a clean worktree.

One bounded owner was prepared at capacity 256 (H/K/N = 128/12288/256). Each static owner was compiled and prepared at its own M. The arms alternated ABAB/BABA for 12 rounds, and each value is the owner's native HIP event window averaged over 50 repeats.

| Active M | Bounded µs | Static µs | Bounded / static | Bitwise equal | Activation update ms | Static compile s |
|---|---|---|---|---|---|---|
| 1 | 362.74 | 363.78 | 0.997 | yes | 0.194 | 0.91 |
| 16 | 148.70 | 145.03 | 1.025 | yes | 0.234 | 0.92 |
| 64 | 158.80 | 153.93 | 1.032 | yes | 0.269 | 0.91 |
| 256 | 189.64 | 186.71 | 1.016 | yes | 0.488 | 0.72 |

- **Launch cost.** The bounded launch costs at most 3.2% against a program compiled for that exact M. In exchange it replaces a fresh compile and preparation per M with a 0.2–0.5 ms activation update. The single owner holds 18 device buffers and 24.7 MB at capacity, and its one compile took 1.3 s.
- **Bitwise equality with static-at-M.** The bounded outputs are bitwise equal to static programs compiled at each M. That is stronger than the capacity oracle the tests require, but it was observed here and is not asserted in general.
- **Unexplained M = 1 result.** M = 1 is slower than M = 16 in both arms equally. The cause is not identified; it is not a bounded-owner effect.
- **Not a promotion witness.** HIP events on WSL2 have no admitted kernel-side witness, and `rocprofv3` returns no records without `/dev/kfd`.

## Not covered

- Public `@jit` admission of a quantized row bound.
- A portable bounded record; `to_manifest` refuses one.
- Dynamic K/N.
- gfx1151, which has no proof here and none transfers.
- Sibling backends.
