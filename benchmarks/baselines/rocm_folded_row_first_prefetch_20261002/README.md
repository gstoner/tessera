# Row-first A/B prefetch retunes on gfx1201

Owner ROCM-MXFP4-W4A8-1; sync ROCM-FOLDED-PREFETCH-RETUNE-2026-10-02.

All variants retain row-first WMMA, ordered full-K accumulation, the ABI,
and workgroup barriers. Native scheduling fences control overlap. Each passed
89 focused contract/device tests on Tajasaurus RX 9070 XT. Seven alternating
graph trials compare against a preserved native reference and frozen matched
HIP control. All three resident outputs are poisoned before validation and
checked again after timing. These are GPU graph windows, not isolated kernel
profiling; public launch wall time is recorded separately.

Lower virtual pressure did not reduce physical allocation: all variants use
177 VGPRs, 29 SGPRs, 25,600 LDS bytes, no scratch/spills. Static split-barrier
counts remain three signals and three waits. All variants were removed from
active source and preserved as patches. No selector promotion or sibling
device evidence follows. Native exact-per-K32, wider coverage and performance
closure remain open.

## late-a: virtual peak 154

| M x N x K | Candidate us | Reference us | Ratio |
|---|---:|---:|---:|
| 256 x 4096 x 1024 | 24.515 | 23.062 | 1.0630 |
| 256 x 4096 x 2048 | 38.176 | 35.286 | 1.0819 |
| 256 x 4096 x 5120 | 80.746 | 77.511 | 1.0417 |
| 256 x 8192 x 5120 | 149.765 | 150.857 | 0.9928 |

## late-b: virtual peak 159

| M x N x K | Candidate us | Reference us | Ratio |
|---|---:|---:|---:|
| 256 x 4096 x 1024 | 24.031 | 23.019 | 1.0439 |
| 256 x 4096 x 2048 | 37.950 | 35.314 | 1.0746 |
| 256 x 4096 x 5120 | 79.010 | 78.423 | 1.0075 |
| 256 x 8192 x 5120 | 150.545 | 150.692 | 0.9990 |

## split-a: virtual peak 155

| M x N x K | Candidate us | Reference us | Ratio |
|---|---:|---:|---:|
| 256 x 4096 x 1024 | 24.360 | 23.042 | 1.0572 |
| 256 x 4096 x 2048 | 37.993 | 35.107 | 1.0822 |
| 256 x 4096 x 5120 | 80.229 | 77.694 | 1.0326 |
| 256 x 8192 x 5120 | 149.637 | 151.341 | 0.9887 |

