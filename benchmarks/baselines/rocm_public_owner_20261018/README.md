# Native ROCm scale admission and program ownership

Owner: ROCM-NVFP4-INGEST-1 / E2E-REAL-6.
Synchronization key: ROCM-PUBLIC-OWNER-OVERHEAD-20261018.

## Result and scope

The four final ABBA runs have identical candidate source snapshots and all
seven identical compiler-generated numerical images. For synthetic
M,H,K,N=128,12288,4096,4096 on RX9070XT/gfx1201:

| Arm | Warm completed public median, ms | Native HIP-event median, ms |
| --- | ---: | ---: |
| 128 MiB control, first | 75.2348 | 2.47189 |
| Integrated owner, first | 37.0271 | 2.46220 |
| Integrated owner, second | 36.5552 | 2.44378 |
| 128 MiB control, last | 71.3925 | 2.46171 |

Both arms use the same checked native scale scanner. This measures the combined
native ownership/staging improvement, including the byte quota. The pool
retains a 374,926,568-byte frame; each selected run reports seven hits and five
misses, versus zero hits and twelve misses for controls. Stage-oracle and
explicit retained-owner sessions contribute to those counters.

Warm timing excludes compilation, device priming and independent oracle
checks. Prepare, fresh-root update, invoke/read and HIP-event intervals have
separate fields. Every timed result is compared bitwise with the prechecked
result. These are synthetic roots, not representative-model quality acceptance.
The unchanged numerical images are generated through Graph/Schedule/Tile/ROCm
Target/LLVM/HSACO and consumed by the checked native program owner.

## Admission and ownership contracts

Positive E4M3FN codes 1..126 are monotone. The native scanner validates every
byte, rejecting negative nonzero/NaN encodings and accepting both signed zeros.
It returns the smallest/largest positive code per projection. Positive finite
FP64 global multiplication followed by FP32 rounding is monotone, so those
endpoints prove the original all-element overflow/positive-underflow predicate.
All-zero projections use an explicit sentinel after global validation.
Bounds commit only after the complete plane passes. Legacy providers retain
the elementwise admission path. Tensor output arithmetic and tolerances stay
under the existing compiler contract.

The native pool retains PID/context/device checks, fresh handles, generations,
image/ABI/geometry identity, completion-before-reuse, LRU eviction and failed
cleanup quarantine. Every acquire/update uploads current roots, including
same-pointer mutations. Partial updates retain capacity-sized snapshots;
readback reserves the declared maximum. Closed owners reset private readiness,
and returned host arrays remain independent of future leases.

gfx1201 uses max(128 MiB, device capacity/32), capped at 512 MiB. Other/unknown
architectures retain 128 MiB. An overflow-safe fit predicate handles contexts
with different quotas. The policy budgets declared device/host payloads and
image-key storage, not total process RSS or driver-internal memory.

## Validation

| Lane | Result |
| --- | --- |
| Full non-slow CPU selection, host WSL | 28,700 passed; 5,697 skipped; zero failed |
| Portable production compiler, all six backends | 1,754 passed; zero skips; executed-proof certificate |
| ABI/header/diagnostic/pass/audit/admission/policy/recorder gates | 2,425 passed |
| gfx1201 prepared execution plus real HIP fault recovery | 38 passed |
| gfx1201 bounded-row selections | 27 passed |
| gfx1201 adjacent primal/JVP/VJP/allocation | 162 passed |
| Generated documents / typing | 33 in sync; mypy errors=0, rc=0 |

Selections overlap and are not summed. The fault cases cover fresh/cached
preparation, full updates and partial updates. Only failed API statuses are
injected; successful work uses the real HIP runtime and compiler images.
Recovery, immutable earlier outputs and cleanup are checked.

Radeon8060S/gfx1151 independently builds all 17 identical runtime source files
and executes the supplied compiler-generated FP32 add. Three leases/two cache
hits, full/partial root uploads, caller mutation after prepare, invalidated
generations, retired handles, immutable outputs and cleanup pass. Its receipt
binds the actual provider/image and every runtime source file. This proves the
named shared ownership contract; gfx1201 format/performance evidence is separate.
Apple, NVIDIA and x86 plan assessments identify the HIP-specific boundary.

## Receipts and reproduction

validation_summary.json indexes the final receipts. CPU/compiler JUnit reports
are losslessly compressed; their uncompressed SHA256 values are recorded.
Intermediate duplicate receipts were pruned. The two scan-attribution packets
retain the selected scalar scanner and rejected slower branchless experiment.
The selected admission comparison also records tracked Python/NumPy temporary
allocation separately from GPU execution.

Use benchmarks/rocm/record_native_program_owner.py on owning gfx1201 with
--expected-revision HEAD_SHA --shape 128x12288x4096x4096 --output OUTPUT_JSON.
The snapshots bind the measured dirty tree to its base revision and hash the
source files and loaded binaries. Restore the base runtime CPP from revision
01906b028c1882b2141576c77180b49737996549 into a scratch copy of the current HIP
runtime sources to build the control, retaining the common scale scanner.
Build selected/control providers into separate directories. provider_variants.json
and owner_v4.patch identify the runtime source difference. Use matching compiler
tools for both arms, run each in a fresh process, reverse arm order, and compare
all seven image digests before interpreting timing.

The gfx1151_check_staging.py recorder accepts --root, --provider and
--source-root. Its scratch root needs package/package.json and package/add.hsaco.
gfx1151_package.json supplies the full compiler artifact; decode its image
payload to reconstruct add.hsaco. Build the native movement target from the
current source/hip, then run the recorder with host Python 3 after an owning
gfx1151 probe. It requires no Python tensor backend.

Current-head public CI/review remains a delivery gate. All five aggregate
programs, representative-model quality, broader composition/layout/AD and the
remaining W8A8/MXFP4 performance obligations remain open.
