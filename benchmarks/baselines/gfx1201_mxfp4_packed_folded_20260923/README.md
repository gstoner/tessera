# GFX1201 packed-folded prefill proof and matched timing

Owner ROCM-MXFP4-W4A8-1 / IKF-1; sync
`GFX1201-PACKED-FOLDED-DECODE-2026-09-23`.

The recorder is
`benchmarks/rocm/benchmark_gfx1201_mxfp4_packed_folded.py`. It compares
the exact K32, expanded-folded, packed table-decode, packed integer-decode,
and pinned Radiance routes on the same logical quantized inputs. Each shape
must pass the independent sampled exact oracle and bitwise BF16 agreement
before alternating HIP-event measurements. Package/model-load conversion and
HSACO compilation are outside the timed launch. `RADIANCE_MXFP4_WPERM=1`
selects the same fragment-order weight layout. The evidence packet records
timed payload SHA-256, selected-symbol ISA, toolchain, source, and benchmark
identities. It does not claim an AMD profiler phase fraction.

The packed integer route is manually executable and receipt-bound, but it
is not automatically selected. Exact K32 remains the correctness default;
the expanded-folded route remains faster for the measured prefill shapes.

The clean Tajasarus packet comes from `eeeabc3379d4bcbb88b6e3503fca27995d671b53`
on RX 9070 XT/gfx1201 with LLVM 23.1.1 and HIP 7.15.26333. Median kernel
milliseconds (11 interleaved trials, 12 iterations each):

| Shape M×N×K | Expanded folded | Packed table | Packed integer | Radiance |
|---|---:|---:|---:|---:|
| 256×5120×8704 | 0.1690 | 0.2651 | 0.2295 | 0.1374 |
| 1024×17408×5120 | 1.1253 | 1.5340 | 1.4120 | 0.8684 |

Packed integer is 1.67× and 1.63× Radiance, respectively, and 1.36× and
1.25× the expanded folded route. All five routes produced identical BF16
outputs for the matched lossless inputs. The selected timed packed-integer
ISA has 32 FP8 WMMA instructions, 2 barrier signal/wait pairs, 88 static
`s_wait_loadcnt` instructions, 117 VGPRs, and no spills. Expanded folded
has 71 static `s_wait_loadcnt` instructions and 109 VGPRs. These are static
counts, not dynamic wait or memory-stall measurements. The packed byte
reduction is not yet a win because register decoding and scheduling costs
currently dominate; test a batched packed-word decode/LDS producer change
before selector admission.
