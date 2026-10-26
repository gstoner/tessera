# SM120 typed attention reverse: once-per-row correction prepass

Owner: E2E-REAL-6 / AD-RESIDUAL-EVAL-1.
Sync: SM120-TYPED-REVERSE-ROW-PREPASS-20261026, which closes the defect recorded by
SM120-SAVED-LSE-MIXED-BACKWARD-20261026.
Contract: `docs/spec/SM120_SAVED_LSE_STORAGE_CONTRACT.md` §"Typed reverse row correction".
Recorded by `benchmarks/nvidia/diagnose_saved_lse_mixed_backward_pairing.py`.

## What changed

When the saved output is half, the native reverse recomputes the row
correction sum(P·dP) in f32 instead of using the rounded O. The inline kernel
(`materializeSm120AttentionBackwardKernel` / `rowStats`) recomputed it inside
every (head, query) step of every dQ/dK/dV element, which made the typed
reverse ~220× slower than the f32 kernel at a mid-size shape.

The compiler already had the split form: a `row_correction` product that
computes the correction once per row into an f32 [B, Hq, Sq] buffer, and a
`backward_rows` consumer that reads it. It was reachable only through
`TESSERA_NVIDIA_ATTENTION_ROW_CORRECTION=prepass` and admitted homogeneous f16/bf16 Q/K/V only.

- **Selection:** the public reverse now selects the prepass automatically
  (`auto`, the new default) whenever Q or the shared K/V storage is f16/bf16.
  The selection is read from the traced Graph IR's `flash_attn` operand types.
  Homogeneous f32 keeps its existing inline kernel, whose correction is the
  O(Dv) dot(dO, O). `inline` and `prepass` remain explicit overrides; an
  explicit prepass the compiler cannot admit still fails at the export.
- **Mixed K/V on the row path:** the row-correction product, its Tile kernels,
  the rows owner and the Python rows program accept a K/V storage distinct
  from Q (as the inline path has since #1015). `kv_storage` enters the row
  contract, identity and entry names (`..._<q>_kv_<kv>_out_<o>_...`) only when it
  differs from Q, so homogeneous row contracts and entries are unchanged. The
  owner is reached through the new `tessera_nvidia_attention_vjp_rows_prepare_kv`.

## Host

- Device: The-Super-Bear, RTX 5070 (sm_120), WSL2, CUDA 13.4, `build-nvidia-cuda`.
- Branch `claude/typed-reverse-row-correction`, clean tree; timing at `5febdf0ef`.
- Contract tests on Tajasarus with the assertions-ON LLVM/MLIR 23.1.1.

## Correctness

- `tests/device/nvidia/test_public_typed_attention_vjp.py`: all homogeneous and
  mixed public reverse cases pass on the row program (they now assert schema
  v3 for every typed case and v1 for f32/f32). Each case covers the public
  call, the warm compiler-free call, serialized replay, resident replay and a
  delayed resident producer against an fp64 oracle.
- With `tests/device/nvidia/test_prepared_attention_row_owner.py`,
  `test_saved_lse_mixed_backward_kernel.py` and
  `test_saved_lse_half_backward_kernel.py`: **601 passed**.
- `tests/device/nvidia/test_public_bounded_attention_rows_program.py::test_static_mixed_kv_row_program_replay_and_resident_capture`:
  10 static mixed row programs (five storage pairs, with and without bias)
  pass compiler-free replay and the resident capture frame. Before `928930fea` all
  10 failed at capture, because the resident tape typed K/V snapshots and dK/dV
  from Q's storage (review finding on #1016). Together with the resident, JVP
  and bounded attention device files, 712 tests pass.
- The bounded-sequence forward projection refuses mixed storage, so there is
  no mixed v4 row program.
- Tajasarus, assertions-ON 23.1.1: 276 row-correction, backward-rows, saved-LSE
  and policy contract tests pass.
- Full device sweep: see the PR (compared against the #1015 failure set).

## Timing (diagnostic only)

`paired-interleaved.json`. Each arm is compiled by the public JIT, checked
against the fp64 oracle, then executed through its prepared reverse owner in
ABAB/BABA order. Values are the owner's CUDA-event windows. CUDA events alone
are not a promotion witness on sm_120. The two `inline` arms force the old
per-element correction in the same interleaved rounds.

At 1×8×2×128×512×64×64 (B, Hq, Hkv, Sq, Sk, D, Dv), causal, median reverse
device time (row producer + backward):

| Arm | Row producer | Backward | Reverse total |
|---|---:|---:|---:|
| f16 homogeneous, **inline** (before) | — | 3565.0 ms | 3565.0 ms |
| f16 homogeneous | 2.21 ms | 15.43 ms | **17.64 ms** (202× faster) |
| bf16 homogeneous | 2.21 ms | 15.57 ms | 17.78 ms |
| f16 Q / bf16 K/V | 2.21 ms | 15.55 ms | 17.76 ms |
| bf16 Q / f16 K/V | 2.21 ms | 15.52 ms | 17.73 ms |
| f32 Q / f16 K/V, **inline** (before) | — | 3434.3 ms | 3434.3 ms |
| f32 Q / f16 K/V | 2.16 ms | 14.80 ms | **16.96 ms** (202× faster) |
| f32 Q / bf16 K/V | 2.17 ms | 14.94 ms | 17.11 ms |
| f16 Q / f32 K/V | 2.24 ms | 14.56 ms | 16.80 ms |
| f32 homogeneous (unchanged v1 kernel) | — | 16.44 ms | 16.44 ms |

Every typed arm is now within 1.03–1.08× of the f32 kernel.

**Small shapes pay one launch.** At 1×2×1×3×5×4×3 the prepass arms take
17.8–20.2 µs against 18.1–18.2 µs inline, up to about 1.1× slower, because the
producer is a second kernel. No shape threshold is applied: the selection is
storage-only, and the absolute cost at that size is about 2 µs.

No route or default promotion beyond the row-correction selection follows.

## Not covered

Bounded/dynamic-sequence mixed K/V (the bounded forward projection refuses it); FP8 or scaled K/V reverse; original-source reverse and JVP over
mixed storage; any non-SM120 device.
