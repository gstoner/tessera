# ROCM-SPLIT-K-1 — gfx1201 router-gate split-K, 2026-09-26

Host **Tajasarus** (RX 9070 XT, gfx1201, WSL2 `/dev/dxg`, ROCm 10.0 / HIP 7.15),
`tessera-opt` from the branch's assertions-ON LLVM/MLIR 23.1.1 build
(`build/`, `-fno-rtti -UNDEBUG`). Source revision in `gfx1201.json`
(`git_head`, clean tree). No other GPU job was running (`pgrep -af record_ssd`
empty before the run).

```bash
source ~/.config/tessera/env.sh   # then TESSERA_OPT / PYTHONPATH pointed at the branch worktree
export TESSERA_ROCM_CHIP=gfx1201 TESSERA_GFX1201_DEVICE_PROOF=1
python benchmarks/rocm/record_split_k_router_gate.py --shapes 16x256x2048 \
    --dtypes fp16,bf16 --runs 3 --rounds 15 --iters 200 --extra-slices 4,8 \
    --output benchmarks/baselines/rocm_split_k_20260926/gfx1201.json
```

| dtype | variant | route | us / iter | vs unsplit (median) | rounds faster |
|---|---|---|---|---|---|
| fp16 | unsplit | measurement-only control | 16.06 | 1.00x | — |
| fp16 | **split:2** | **production** `gfx1201_register_wmma_1x1_splitk2_ordered` | 7.84 | **2.05x** | 45/45 |
| fp16 | split:4 | measurement-only sweep | 6.29 | 2.58x | 45/45 |
| fp16 | split:8 | measurement-only sweep | 5.72 | 2.79x | 45/45 |
| bf16 | unsplit | measurement-only control | 16.04 | 1.00x | — |
| bf16 | **split:2** | **production** | 8.02 | **2.01x** | 45/45 |
| bf16 | split:4 | measurement-only sweep | 6.42 | 2.52x | 45/45 |
| bf16 | split:8 | measurement-only sweep | 5.75 | 2.80x | 45/45 |

(us / iter and the ratio are medians over the three fresh-process runs; each
run's ratio is the median of 15 paired, interleaved rounds.)

What the numbers are and are not:

- A split iteration is **both** launches (partial + ordered reduce); the
  workspace is allocated once outside the timed loop, as are all buffers.
  `runtime.launch` instead `hipMalloc`s/`hipFree`s the S*M*N*4-byte workspace
  on every call; that per-call cost is **excluded** from every row here.
- The unsplit control is the production Tile IR with the `tessera.split_k`
  pair removed, compiled by the same `_compile_native_tile_ir` call. It is
  what this shape ran before ROCM-SPLIT-K-1; it is not a production route now.
- Timing is the synchronized host wall clock over 200-launch batches. It is
  **not** a device-clock measurement and `performance_eligible` is False.
- Correctness before timing: relative error vs an f64 reference 1.6e-6 (fp16)
  and 5e-7 (bf16) for the selected split; split vs unsplit max relative
  difference 1.7e-6 / 7.4e-7.
- The S=4/S=8 rows say the selection rule (fill each WGP once) is
  conservative for this shape. Why is not measured: one hypothesis is that one
  wave per WGP leaves three of its four SIMDs idle, but no counters exist on this
  WSL2 host to confirm it. The rule was **not** retuned from one shape.
- gfx1151 evidence: none. Split-K is never selected there.

## Device-test and sweep logs (Tajasarus, 2026-09-26)

Each log starts with host, device, commit (with a tracked-changes count),
`tessera-opt` path, env and the exact command.

- `device_tests_gfx1201.txt` -- `tests/unit/test_rocm_split_k.py` at
  `1387cd1b` (review-fix branch): **41 passed, 0 skipped**. That covers the
  host-free cases, the router/ragged split cases (fp16/bf16, none / bias+gelu /
  bias+relu, bit-identical reruns), the unsplit control, and the forged-descriptor
  refusals (typed workspace, provenance copy, reduce workgroup).
- `spectral_sweep_{head,base_054fa7c3}.txt` -- the same `-k "rocm or gfx1201 or
  gfx1151"` unit sweep at the review-fix HEAD and at `054fa7c3` (before any
  split-K code). Both fail exactly one test,
  `test_spectral_streaming.py::test_physical_streaming_broadcast_strides_and_artifact_lineage[rocm-True]`
  (`rc=246`). It was already failing, and split-K did not introduce it.
- `spectral_isolated_{head,base_054fa7c3}.txt` -- that test file's parameters
  run alone: they pass on both commits. So the failure depends on test order.
  Separately, the passing run asserts `architecture_identity == "gfx1151"`
  on a gfx1201 device (see `docs/audit/backend/rocm/todo.md`).

