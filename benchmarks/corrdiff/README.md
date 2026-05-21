# CorrDiff-core benchmark

A small regional-weather diffusion benchmark exercising the
compiler-pipeline pieces shipped in Phase 7:

| Piece                          | Source                                                 |
|--------------------------------|--------------------------------------------------------|
| NHWC conv2d                    | `tessera.ops.conv2d` — Graph IR backbone               |
| 2D local-window attention      | `tessera.ops.attn_local_window_2d` (Sub-1 Graph IR op) |
| Deterministic diffusion noise  | `tessera.rng.RNGKey` (Philox)                          |
| Activation checkpointing       | `tessera.autodiff.checkpoint`                          |
| Tiled fields                   | `tile_field` — row-major partition with halo width    |

## Forward path

```
x  : (B, H, W, C_in)
h1 : conv2d(x,  W1)             — 3×3 conv stem
h2 : checkpoint(conv2d)(h1, W2) — second conv, recomputed in backward
attended : attn_local_window_2d(reshape(h2, heads), window=cfg.window)
flat     : reshape back to NHWC
noised   : diffusion_noise_step(flat, sigma)
y        : conv2d(noised, W_out)
```

Each forward call is **deterministic** given the same ``(seed, step)``
— the `RNGKey.fold_in(step)` derivation guarantees reproducible noise.

## Run

```
PYTHONPATH=.:python python benchmarks/corrdiff/benchmark_corrdiff.py \
    --reps 5 --warmup 2 --output /tmp/corrdiff.json
```

Sample (laptop CPU, fp32):

```
shape                                              latency_ms   thr_msps    bw_gb/s det
B=1 H=16 W=16 C_hid=8 heads=2 window=[1, 1]              3.96      0.065      0.006 ok
B=2 H=32 W=32 C_hid=16 heads=2 window=[1, 1]            18.13      0.113      0.018 ok
B=2 H=32 W=32 C_hid=16 heads=4 window=[2, 2]            22.81      0.090      0.014 ok
```

## Status

| Axis                    | Status              | Notes                          |
|-------------------------|---------------------|--------------------------------|
| Numerical contract      | locked, deterministic | bit-identical across runs    |
| Backend                 | reference (CPU)     | Apple GPU / NVIDIA paths land via the kernel manifest as the underlying ops ship native lowering |
| Halo / distributed      | scaffold ready      | `HaloMeshIntegrationPass` consumes the `halo_aware` metadata on `attn_local_window_2d` when sharded |
| JSON schema             | Architecture Decision #12 | ingestible by `tools/roofline_tools/` |

## Why CorrDiff?

CorrDiff is NVIDIA's regional-weather diffusion model: it combines
spatial convolutions, local-window attention for terrain bias, and
diffusion sampling for refinement.  Each ingredient is one of the
pieces the Phase 7 asks delivered, so this benchmark gives us an
end-to-end smoke target every time a related compiler pass changes.
