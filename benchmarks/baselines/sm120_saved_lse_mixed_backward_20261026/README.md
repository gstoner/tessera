# SM120 saved-LSE reverse: mixed Q vs K/V storage

Owner: E2E-REAL-6 / AD-RESIDUAL-EVAL-1.
Sync: SM120-SAVED-LSE-MIXED-BACKWARD-20261026, which extends SM120-SAVED-LSE-MIXED-STORAGE-20261025.
Contract: `docs/spec/SM120_SAVED_LSE_STORAGE_CONTRACT.md` §"Mixed Q vs K/V storage (backward)".
Recorded by `benchmarks/nvidia/diagnose_saved_lse_mixed_backward_pairing.py`.

## What it is

The paired native reverse of `flash_attn` and the saved-LSE backward checkpoint
admit K/V (shared by K and V) in an f16/bf16/f32 storage distinct from Q's.
dQ takes Q's storage; dK and dV take the K/V storage. LSE, its cotangent and
bias stay f32, and accumulation is f32. Before this slice the public call
failed at Graph AD with zero cotangents.

`kv_storage` enters the native contract, the checkpoint identity, the entry
name (`..._<q>_kv_<kv>_out_<o>_<digest>`) and the Tile kernel only when it
differs from Q, so homogeneous reverse contracts are unchanged. A mixed reverse
uses the typed backward ABI and VJP program schema v2 even when Q is f32, and
reaches the retained owner through the new
`tessera_nvidia_attention_vjp_prepare_typed_kv`.

## Host

- Device: The-Super-Bear, RTX 5070 (sm_120), WSL2, CUDA 13.4, `build-nvidia-cuda`.
- Branch `claude/saved-lse-mixed-backward`, clean tree; timing at `147617b43`.

## Correctness

- `tests/device/nvidia/test_public_typed_attention_vjp.py::test_public_mixed_kv_reverse`:
  **192 passed.** Grid: six (Q, K/V) storage pairs × LSE-cotangent seed or not
  × bias or not × causal or not × two shapes (incl. Sk=129 GQA) × full or subset
  `wrt`. Each case checks the public call, the warm call with compilers and the
  concrete frontend forbidden, compiler-free serialized replay, resident replay
  and a delayed resident producer against an fp64 reverse oracle, and the
  result dtype per role.
- `tests/device/nvidia/test_saved_lse_mixed_backward_kernel.py`: **96 passed.**
  The direct backward kernel with pruned, non-compact gradient outputs, so an
  inactive dQ/dK/dV buffer is still present and zero-filled. The grid is six
  mixed pairs × four `wrt` subsets × seed or not × two shapes, with guard bytes
  around every buffer. Before `ce9a0adf9` the 48 cases where an inactive dK/dV's
  width differs from Q's failed: the zero-fill used Q's storage (review finding
  on #1015).
- Attention VJP and saved-LSE device files: **923 passed, 10 skipped**. The 10
  are the proof-lane prepared-VJP tests, which pass 10/10 with
  `TESSERA_NVIDIA_DEVICE_PROOF=1`.
- Full `tests/device/nvidia -n 8`: 240 failed / 5947 passed / 59 skipped. The
  failure set matches the one recorded on the scaled-K/V branch (#1014) with
  no new entries; one known flaky source-reverse case passed this time.
  Lit on this host: `tests/tessera-ir` 441 passed / 124 unsupported / 0 failed;
  `check-tessera-nvidia` 72 passed.
- `tests/unit/test_saved_lse_mixed_backward_contract.py`: 21 contract tests,
  passing on the Mac and on Tajasarus's assertions-ON LLVM/MLIR 23.1.1
  (226 tests in the saved-LSE contract set there).

## Timing (diagnostic only)

`paired-interleaved.json`. Each arm is compiled by the public JIT, checked
against the fp64 oracle, then executed through the same prepared reverse owner
in ABAB/BABA order. Values are the owner's CUDA-event windows: the saved-LSE
forward and the backward kernel. CUDA events alone are not a promotion
witness on sm_120.

Backward medians at 1×8×2×128×512×64×64 (B, Hq, Hkv, Sq, Sk, D, Dv), causal:

| Arm | Backward | vs its homogeneous control |
|---|---:|---:|
| f16 homogeneous | 3.572 s | — |
| f16 Q / bf16 K/V | 3.582 s | 1.003× f16 |
| bf16 homogeneous | 3.581 s | — |
| bf16 Q / f16 K/V | 3.588 s | 1.002× bf16 |
| f32 homogeneous (v1 kernel) | 16.18 ms | — |
| f32 Q / f16 K/V | 3.427 s | **212× f32** |
| f32 Q / bf16 K/V | 3.405 s | **210× f32** |
| f16 Q / f32 K/V | 3.423 s | 0.96× f16 |

**Mixed storage costs nothing against the typed path it runs on, but the typed
path itself is slow.** It is pre-existing: on main (`da64aea4e`) the same shape
measured f16 3.560–3.570 s, bf16 3.583–3.599 s and f32 16.18 ms with the same
owner. The cause is in `materializeSm120AttentionBackwardKernel`. When the
saved output is half, `rowStats` recomputes the row correction sum(P·dP) with
`emitProbabilityRowCorrection` (a full Sk loop of D- and Dv-length dots). It
does that inside every (head, query) step of every dK/dV element, instead of
computing it once per row; the f32 path uses the O(Dv) dot(dO, O).
Consequence for this slice: a mixed reverse with f32 Q and half K/V lands on
that slow path, about 210× its f32 homogeneous control at this shape. At the
small shape (1×2×1×3×5×4×3) all typed arms are about 18 µs and f32 homogeneous
is 8.9 µs.

No route or default promotion follows.

**Update 2026-10-26:** SM120-TYPED-REVERSE-ROW-PREPASS-20261026 closes this defect: the reverse now computes the correction once per row by default (`benchmarks/baselines/sm120_typed_reverse_row_prepass_20261026/`).

## Not covered

FP8 or scaled K/V reverse; original-source reverse, JVP and projection over a
mixed forward; resident mixed reverse beyond the public owner; any non-SM120
device.
