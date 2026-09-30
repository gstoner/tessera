# Paired resident RMSNorm to matmul: bounded dynamic M/N/K

This evidence packet closes the combined bounded M/N/K slice for the paired gfx1201 and NVIDIA sm_120 resident Graph to Schedule to Tile package contract in PR #891. The public producer and consumer are RMSNorm and matmul. Host views with row padding and a column-major padded RHS are packed to the checked compact device ABI. Producer and consumer event timings are measured separately on their owning devices; host packing and upload are excluded.

Both raw packets use the clean source revision 7b42974c40c2d86b00c95d4a9c061dd1ecb2e6b8 and check correctness before timing. The NVIDIA packet has 31 CUDA-event samples per stage and shape. The gfx1201 packet has 100 HIP-event samples per stage and shape.

| Device | Package bound M/N/K | Active M/N/K | Storage | Producer median / CV | Consumer median / CV |
| --- | ---: | ---: | --- | ---: | ---: |
| RX 9070 XT, gfx1201 | 128/256/256 | 64/128/128 | fp16 | 14.86 us / 93.8% | 10.28 us / 86.4% |
| RX 9070 XT, gfx1201 | 128/256/256 | 96/192/192 | fp16 | 11.00 us / 6.3% | 14.44 us / 9.3% |
| RX 9070 XT, gfx1201 | 128/256/256 | 128/256/256 | fp16 | 11.18 us / 25.3% | 15.62 us / 59.9% |
| RTX 5070, sm_120 | 512/512/256 | 256/256/128 | fp16 | 30.92 us / 9.0% | 14.60 us / 3.7% |
| RTX 5070, sm_120 | 512/512/256 | 384/384/192 | fp16 | 47.13 us / 0.1% | 20.61 us / 1.5% |
| RTX 5070, sm_120 | 512/512/256 | 512/512/256 | fp16 | 59.66 us / 0.2% | 30.69 us / 0.2% |

The fp16 timings demonstrate stage attribution and package reuse only. gfx1201 variation is high, especially on the smallest and largest rows; no speedup or route-promotion claim is supported. The NVIDIA timings were measured on a different shape envelope and are not directly comparable to gfx1201.

Separate exact-device M/N/K tests also passed for fp16 and bf16 on both devices, including numerical comparison, stable image digests, and reuse of the intermediate allocation. The timing packets intentionally contain fp16 data only.

Reproduce on Tajasaurus after setting its gfx1201 build and runtime environment:

    python benchmarks/rocm/benchmark_gfx1201_resident_norm_matmul.py --dtype fp16 --dynamic-mnk --warmup 25 --iterations 100

Reproduce on Super-Bear after setting its sm_120 build and runtime environment:

    python benchmarks/nvidia/benchmark_scheduled_rmsnorm_matmul_edge.py --dynamic-mnk --warmup 25 --reps 100 --samples 31

Raw packets: gfx1201_dynamic_mnk.json and sm120_dynamic_mnk.json.
