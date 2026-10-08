# GFX1201 packed MXFP4 vector-scale epilogue

Owner: ROCM-MXFP4-W4A8-1; sibling ROCM-NVFP4-INGEST-1.
Sync: GFX1201-PACKED-VECTOR-SCALES-2026-10-06. Publication pending.

## Native change and envelope

Graph → Schedule → Tile → ROCm Target → MLIR/LLVM native image remains
the package route. The native materializer selects the existing RDNA4 Tile
vector-scale epilogue for static packed M=256, N>=1024, K>=1024.
Runtime-projected shapes, short K, small N and other M retain the scalar seed.
Vector loads check complete lane rows and 16-byte alignment; fallback loads
retain masked row/column semantics. Full-K accumulation, row-reference order,
BF16 rounding, packed decode, scale semantics and checked ABI are unchanged.
This changes no format selector and adds no Python production construction.

## Validation

Live RX 9070 XT / gfx1201 on Tajasaurus; matching LLVM/MLIR 23.1.1 build.
72 numerical/contract tests pass, including all E2M1 codes, scale deltas,
zero scales, ragged shapes, poison outputs, resident launch and runtime projection.
344 selector, diagnostic and pass metadata checks pass. Six new selector
cases protect scalar boundaries and projected image identity.
Four packets have 60 correctness-checked arms, each with three resident
device-clock windows bracketed by HIP events and separate checked host walls.
All final packet source hashes match the current measured source.
FP8, standard MXFP8, expanded MXFP4 and the M200 short-K packed image are
byte-identical to their preserved controls; original f32 inputs match by hash.

Recorder: benchmarks/rocm/benchmark_gfx1201_three_formats.py.
Options: --include-native-packed --windows 3
--shapes "256,1024,1024;256,4096,5120;200,256,128";
reverse packets additionally use --reverse-arms. The reference-reverse
run uses the preserved compiler and --reference-source-dir. Native diff
and complete build/test receipts are adjacent; compiler binaries stay in scratch.

## Resident device execution plus dispatch (microseconds)

| Order | M,N,K | Scalar reference | Final selector | Final/reference |
| --- | --- | ---: | ---: | ---: |
| forward | [256, 1024, 1024] | 22.993 | 20.431 | 0.8886 |
| forward | [256, 4096, 5120] | 89.344 | 87.168 | 0.9756 |
| forward | [200, 256, 128] | 15.306 | 18.249 | 1.1923 |
| reverse | [256, 1024, 1024] | 23.031 | 22.627 | 0.9825 |
| reverse | [256, 4096, 5120] | 91.390 | 89.554 | 0.9799 |
| reverse | [200, 256, 128] | 18.367 | 18.380 | 1.0007 |

At N1024/K1024, the final forward run also speeds up unchanged controls
(e.g. expanded MXFP4 21.166→19.059 us). Its raw 11% packed improvement
must not be attributed solely to this epilogue. Reverse order gives a 1.75%
packed improvement with the expanded control essentially unchanged.
At N4096/K5120, both orders improve about 2.0–2.4%. The unmodified M200
image varies substantially across forward runs, demonstrating timing drift.
Host-wall samples are reported separately and imply no kernel speedup.

ISA adds sixteen static global_load_b128 sites, retaining scalar fallback
sites. The two split-barrier signal/wait sites and twelve ds_load_2addr_b64
sites are unchanged. This is an epilogue experiment, not LDS staging closure.

## Remaining work

The measured performance envelope contains these two M256 shapes only;
intermediate/larger N/K need owning-device characterization. This does not
explain or close the Radiance per-column gap. Wider staging, deep fragment
buffering, C-through-LDS transpose and persistent scheduling remain open.
No gfx1151, NVIDIA, Apple or x86 physical evidence transfers.
Full-suite scaled-matmul AD closure and aggregate publication remain open.
