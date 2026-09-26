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
- The unsplit control is the production Tile IR with the `tessera.split_k`
  pair removed, compiled by the same `_compile_native_tile_ir` call. It is
  what this shape ran before ROCM-SPLIT-K-1; it is not a production route now.
- Timing is the synchronized host wall clock over 200-launch batches. It is
  **not** a device-clock measurement and `performance_eligible` is False.
- Correctness before timing: relative error vs an f64 reference 1.6e-6 (fp16)
  and 5e-7 (bf16) for the selected split; split vs unsplit max relative
  difference 1.7e-6 / 7.4e-7.
- The S=4/S=8 rows say the selection rule (fill each WGP once) is
  conservative for this shape. The rule was **not** retuned from one shape.
- gfx1151 evidence: none. Split-K is never selected there.
