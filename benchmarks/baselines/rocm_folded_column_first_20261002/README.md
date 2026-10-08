# gfx1201 folded column-first consumption experiment

Owner ROCM-MXFP4-W4A8-1; sync ROCM-FOLDED-COLUMN-FIRST-2026-10-02.

## Outcome

Rejected. Consuming each B fragment across the four row fragments before
advancing to the next column changes native instructions but does not shorten
the measured live-register peak. Seven alternating trials show no consistent
gain; the 256 x 4096 x 5120 median is 1.95% slower. The candidate is removed
from active source. Its patch and revision-bound measurements remain here.

## Exact-device proof

Tajasaurus RX 9070 XT, live gfx1201. The matching LLVM/MLIR 23.1.1 compiler
passed 89 focused contract and device tests. Device coverage includes ragged
M/N, static and runtime K, package reuse, and exceptional scale-product
overflow/underflow/zero recovery. Before timing, outputs from all three
resident copies were poisoned and checked against the frozen matched HIP
control and an independent numerical oracle. Graph/plain results agree
bitwise; outputs were rechecked after the last timing operation.

## Timing

Microseconds per captured launch, median of seven alternating graph-window
trials. Ratios above one mean the candidate was slower.

| M x N x K | Candidate us | Reference us | Candidate / reference |
|---|---:|---:|---:|
| 256 x 4096 x 1024 | 23.069 | 23.083 | 0.9994 |
| 256 x 4096 x 2048 | 34.817 | 34.841 | 0.9993 |
| 256 x 4096 x 5120 | 78.770 | 77.261 | 1.0195 |
| 256 x 8192 x 5120 | 150.278 | 150.786 | 0.9966 |

These windows include GPU graph dispatch and markers, exclude capture and
instantiation, and are not isolated kernel profiling. Each marked window is
at least 5 ms, with a device-clock/event witness check within 5%. Public
runtime-launch wall samples are recorded separately in gfx1201.json. Startup
and clock variation are visible in the raw samples. No speedup, occupancy,
Radiance comparison, selector promotion, or sibling-device claim follows.

## Compiler attribution and restoration

Candidate and reference: 165 virtual peak VGPRs, 177 physical VGPRs,
29 SGPRs, 25,600 LDS bytes, no scratch or spills. At the peak, the listed
live set has 64 WMMA accumulator registers, 48 LDS-fragment registers,
20 global-prefetch registers, and 25 other registers; eight additional
registers are unattributed at the early-clobber instruction. Native HSACO
instructions match the diagnostic LLVM register-pressure stream.

The rebuilt restored compiler reproduces the pre-candidate instruction-stream
SHA256 and resources. See pressure/pressure.json and
restored-pressure/pressure.json; the rejected source diff is
rejected-column-first.patch. The candidate temporarily touched the shared
FP8 body during this isolated experiment; no source change is retained.

## Remaining work

Native versus matched HIP graph gaps remain; changing fragment consumption
order alone does not explain or close them. Next investigation should control
prefetch overlap and LDS fragment lifetime separately. Exact-per-K32 native
migration, wider K/layout coverage, and model-quality obligations stay open.
Apple, NVIDIA, x86, and gfx1151 have no changed lowering or execution proof.
