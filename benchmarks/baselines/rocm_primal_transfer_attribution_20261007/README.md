# gfx1201 public primal transfer and idle-cache attribution

Owner: FRONTEND-IR-MEDIUM-1 / E2E-REAL-6 / ROCM-FP8-BLOCKSCALE-1.
Synchronization: ROCM-PRIMAL-TRANSFER-EVICTION-2026-10-07.
Owning device: RX 9070 XT, gfx1201, GPU-28d9e7efbf2ef716.

## Current implementation and proof

Native HIP chooses pinned staging automatically only for gfx1201 inputs
between 256 KiB and 8 MiB, with output capacity at most 8 MiB. Explicit
TESSERA_ROCM_PROGRAM_PINNED=0/1 remains an attribution override. Transfer
mode belongs to exact program cache identity. gfx1151 remains pageable by
default; these measurements establish no sibling execution or performance.

Completed idle owners now use bounded least-recently-used eviction rather
than permanently admitting the first four programs. The four-owner/128 MiB
bound remains. Active owners are untouched. Foreign-context and poisoned
owners cannot be selected for eviction. Cleanup waits for stream completion;
failed cleanup retains quarantined resources and their budget accounting.

eviction-device-tests.log records 24 public primal/JVP tests passing after
the matching native runtime rebuild. Six distinct programs exercise cache
capacity, recent-owner hits, oldest-owner misses and changed-scale numerics.
The selected automatic transfer mode reuses the equivalent explicit mode.
eviction-source-tools.json binds current native source, test, recorder,
compiler and runtime bytes. Shared diagnostics/pass metadata gates pass 296.

## Paired public timing

paired-public-eviction.json records eight format/layout/shape rows, with
float64 block numerics and changed scales checked before timing. Compiler
subprocesses are forbidden during warm windows. Each row uses 21 alternating
windows of 10 ordinary JIT calls across pageable, pinned, automatic and
identical-policy controls. Owner HSACO hashes and live rocminfo are retained.

At M200/N129/K1536, automatic public medians are 1.22-1.74 ms versus
pageable 4.17-4.72 ms. Paired automatic/pageable ratios are 0.293-0.371.
Small M17/N19/K256 paired ratios are 1.000-1.009. Identical-policy control
ratios range 0.980-1.005 in the large rows. These are public wall-clock costs
including frontend/validation/transfers/launch/readback, not isolated kernel
measurements and not comparisons against AITER or Radiance.

## Historical exploration retained

paired-public.json is the original pre-eviction run: after four cached owners
the later rows repeatedly allocated and destroyed resources, reaching 6-9 ms.
It is retained unchanged to expose the admission bug, not as current timing.
profile.json and pinned/automatic format packets preceded eviction; cProfile
is intrusive attribution, not a speedup measurement. three-format-runtime-
regression.json covers six diagnostic FP8/MXFP8/MXFP4 staging rows; it does
not establish a public MXFP4 compiler route.

## Remaining work

Generic scaled_matmul batching/linear transpose, eager E8M0 differential
reference, composed/dynamic AD, broader layout/cache families, W8A8/MXFP4
performance programs, sibling physical parity, full-suite closure and PR
delivery remain open.
