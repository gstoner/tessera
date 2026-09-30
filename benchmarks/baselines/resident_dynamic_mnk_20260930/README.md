# Paired resident RMSNorm to matmul: bounded dynamic M/N/K

This evidence packet covers the paired gfx1201 and NVIDIA sm_120 resident Graph to Schedule to Tile package contract in PR #891. The public producer and consumer are RMSNorm and matmul. Host views with row padding and a padded column-major RHS are packed to the checked compact device ABI. Producer and consumer event timings are measured separately on their owning devices; host packing and upload are excluded.

The fp16 packets were captured from clean source revision `7b42974c40c2d86b00c95d4a9c061dd1ecb2e6b8`: gfx1201 used 100 HIP-event samples per stage and shape, and sm_120 used 31 CUDA-event samples. The separate bf16 packets were captured from clean revisions `5d680dc7b728b7e8aa38505a1fb6c33c0379e6ba` (gfx1201) and `2af201674c3580a47c1f882fca0a53937780df42` (sm_120); sample counts match each device's fp16 packet. Both storage types passed exact-device numerical checks and confirmed intermediate allocation reuse and stable package images.

## fp16

| Device | Package bound M/N/K | Active M/N/K | Producer median / CV | Consumer median / CV |
| --- | ---: | ---: | ---: | ---: |
| RX 9070 XT, gfx1201 | 128/256/256 | 64/128/128 | 14.86 us / 93.3% | 10.28 us / 86.0% |
| RX 9070 XT, gfx1201 | 128/256/256 | 96/192/192 | 11.00 us / 6.2% | 14.44 us / 9.2% |
| RX 9070 XT, gfx1201 | 128/256/256 | 128/256/256 | 11.18 us / 25.2% | 15.62 us / 59.6% |
| RTX 5070, sm_120 | 512/512/256 | 256/256/128 | 30.92 us / 9.0% | 14.60 us / 3.7% |
| RTX 5070, sm_120 | 512/512/256 | 384/384/192 | 47.13 us / 0.1% | 20.61 us / 1.5% |
| RTX 5070, sm_120 | 512/512/256 | 512/512/256 | 59.66 us / 0.2% | 30.69 us / 0.2% |

## bf16

| Device | Package bound M/N/K | Active M/N/K | Producer median / CV | Consumer median / CV |
| --- | ---: | ---: | ---: | ---: |
| RX 9070 XT, gfx1201 | 128/256/256 | 64/128/128 | 14.50 us / 48.6% | 12.72 us / 164.9% |
| RX 9070 XT, gfx1201 | 128/256/256 | 96/192/192 | 10.84 us / 12.7% | 14.20 us / 1.0% |
| RX 9070 XT, gfx1201 | 128/256/256 | 128/256/256 | 10.96 us / 1.7% | 15.36 us / 108.4% |
| RTX 5070, sm_120 | 512/512/256 | 256/256/128 | 30.88 us / 0.1% | 14.70 us / 2.6% |
| RTX 5070, sm_120 | 512/512/256 | 384/384/192 | 47.10 us / 0.3% | 20.67 us / 1.3% |
| RTX 5070, sm_120 | 512/512/256 | 512/512/256 | 59.62 us / 1.1% | 30.70 us / 0.3% |

These measurements support stage attribution and package-reuse evidence only. gfx1201 variation remains high on several rows, especially bf16; no speedup, cross-architecture ranking, or route-promotion claim is supported.

Reproduce on Tajasaurus after setting its gfx1201 build and runtime environment:

    python benchmarks/rocm/benchmark_gfx1201_resident_norm_matmul.py --dtype bf16 --dynamic-mnk --warmup 25 --iterations 100

Reproduce on Super-Bear after setting its sm_120 build and runtime environment:

    python benchmarks/nvidia/benchmark_scheduled_rmsnorm_matmul_edge.py --dtype bf16 --dynamic-mnk --warmup 25 --reps 100 --samples 31

Raw packets: [gfx1201 fp16](gfx1201_dynamic_mnk.json), [sm_120 fp16](sm120_dynamic_mnk.json), [gfx1201 bf16](gfx1201_dynamic_mnk_bf16.json), and [sm_120 bf16](nvidia_dynamic_mnk_bf16.json).
