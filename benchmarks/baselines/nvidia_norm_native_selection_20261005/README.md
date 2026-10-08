# Native SM120 normalization selection

Owner: W1.1; sibling FRONTEND-IR-MEDIUM-1. Sync: NVIDIA-NORM-NATIVE-SELECTION-2026-10-05.

C++ Graph-to-Schedule selects cooperative_128 for SM120 norm columns >= 256 and serial for shorter rows. Explicit policies override selection. Python reads the native serialized decision; Schedule/Tile hashes retain it. This is normalization scheduling for fp16/BF16/fp32, with no quantized-format or matmul policy promotion.

## Validation

- 126 RTX 5070 device tests passed, including 255/256/257 boundaries, cached ordinary frontend calls, portable replay and fused consumers.
- 424 shared tests passed, 17 skipped.
- 18 RX 9070 XT / gfx1201 norm and fused-epilogue regressions passed after matching shared compiler rebuild.
- 36 composed benchmark cases passed the original independent float64 oracle before timing. Selected/serial arms use identical source/RHS/epilogue values and consumer image.

## Paired timing

Five alternating arm trials; CUDA event windows include resident dispatch, while checked host wall includes staging, allocation, dispatch, readback and cleanup. These are separate measurements.

| Storage | Producer | M/K/N | Serial producer event ms | Selected producer event ms | Serial wall ms | Selected wall ms |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| fp16 | rmsnorm | 128/1024/64 | 0.207394 | 0.008716 | 3.365919 | 3.491607 |
| fp16 | layernorm | 128/1024/64 | 0.254577 | 0.009190 | 4.286372 | 4.561184 |
| bf16 | rmsnorm | 128/1024/64 | 0.207543 | 0.008444 | 3.518867 | 3.646242 |
| bf16 | layernorm | 128/1024/64 | 0.254331 | 0.009222 | 3.750439 | 3.611473 |

The producer event window improves strongly at K1024; checked wall time does not improve consistently. Consumer image identity is preserved, and host overhead remains open.

## Retained numerical failure

BF16 M128/K4096/N64 RMSNorm→matmul exceeds the original composed oracle tolerance for both schedules (one output each). Serial max output error is 0.02533446; cooperative is 0.02263976. Stored norm differs from the float64-rounded oracle at 139 versus 35 values, with max difference 0.015625. Matmul matches an independent oracle using its actual stored intermediate to 0.00032467. The original failure log and bf16-long-attribution.json are retained. All BF16 K4096 benchmark timing claims are excluded pending a justified composed numerical contract; the original tolerance has not been loosened.

General producer/AD/dynamic integration, FP8/MXFP8/MXFP4 evaluation and broader program closure remain open. Automatic selection is an implementation under evaluation, not a declaration of compiler-wide completion.
