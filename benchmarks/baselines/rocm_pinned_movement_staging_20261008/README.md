# Native pinned movement staging: paired exact-device evidence

Owner E2E-REAL-6. Synchronization ROCM-PINNED-MOVEMENT-STAGING-20261008.

The native host movement service now uses reusable private pinned staging by
default for paged gathers and the existing gfx1151 MoE gather route. Every call
copies the full checked source span and index data into its private scratch,
then copies completed output into the caller's separate array. Public Python
storage validation, MLIR Graph/Schedule/Tile/Target, LLVM/HSACO, image identity
and checked descriptor ABI remain unchanged. Python emits no kernel or physical
schedule. The gathered output remains compact.

This is a host-transfer optimization, not a GPU kernel speedup. Prior native
cost attribution identified H2D/D2H transfers as the dominant pageable cost;
this packet measures the actual candidate rather than inferring a gain from
those diagnostic profiles.

## Ownership and policy

`TESSERA_ROCM_MOVEMENT_PINNED_STAGING` unset selects best-effort pinned staging.
A pinned allocation failure falls back to the existing native pageable path;
partial private allocations stay owned and are cleaned during explicit clear.
`0` selects the pageable control; `1` explicitly requires pinned staging and
returns the existing allocation status on allocation failure. Other values are
rejected. Failed free/completion does not authorize releasing a still-owned
buffer or image lease; quarantined owners require completion and cleanup.

The 128 MiB retained arena limit accounts for both device and host capacities.
Pinned reuse requires the request's combined host/device scratch to fit the
limit. Larger requests can use transient scratch; the limit is retained memory,
not a bound on peak memory during growth. Shared native math includes retained
host capacities even though its own transfers keep their previous path.
Cleanup remains tied to the owning process/context and synchronous completion.

## Validation

- Final default-policy build, gfx1151 Radeon 8060S: 52 owning movement cases
  pass; four other-architecture cases skip. Six additional f32/f16/BF16 native
  sqrt/add interleaving and repeated teardown cases pass with no skips.
- Final default-policy build, gfx1201 RX 9070 XT: 48 owning movement cases
  pass; six other-architecture/family cases skip. Six native math interleaving
  cases pass with no skips. The existing unregistered pytest timeout warning
  is retained in the receipt.
- Seven compiled native fault-injection/binding tests pass in host WSL.
  Scenarios include warm scratch reuse, changed pages/table, policy switching,
  failure of each of three pinned allocations, copy failure, failed completion,
  failed pinned free/growth, quarantine/recovery, reuse disabled, process/context
  isolation and shared math ownership. These controlled tests are not AMD device
  proof. The initial stale HIP graph-header failure is retained; the repair adds
  rejecting graph API stubs. Real graph execution is proved on the owning GPUs.

Core/ROCm tools are the unchanged assertion-enabled LLVM/MLIR 23.1.1 binaries
used by the preceding strided packet. The candidate movement runtime was built
fresh on each GPU host; image-cache runtime ABI was retained. The recorded
explicit candidate-source path names the C++ bytes compiled into that binary,
while unchanged Python/IR sources come from the synchronized owning snapshot.
All 20 source hashes match this delivered candidate. Actual compiler/target/
runtime SHA-256 values are recorded. All sixteen image digests exactly match
the preceding strided execution packet; there is no shader/image change.

## Same-image paired timings

Recorder: `benchmarks/rocm/benchmark_pinned_staging.py`.
Seven rotating rounds alternate arm order at identical input/table addresses,
using the same compiler images and device arenas in one candidate runtime.
Each arm measures 128 completed ordinary JIT calls; compilation is forbidden
while warm. Numerical checks precede timing and follow every window, then
changed data/table and retained output are checked again. All completed windows
are longer than 20 ms. Raw windows and paired ratios are retained.

Small pages `(4,256,3,8)`, four logical pages; large `(32,16,8,128)`, 64 logical
pages. Both gather 1,024 output tokens. Within each shape all layouts have the
same logical page values. Public timing includes host copies/transfers and
completion. Separate resident HIP events exclude transfers and are diagnostic
kernel measurements, not the denominator for the host gain.

| Architecture | Profile | Layout | Pageable public (ms) | Pinned public (ms) | Paired pinned/pageable median |
| --- | --- | --- | ---: | ---: | ---: |
| gfx1151 | small | compact | 1.9012 | 0.4907 | 0.255 |
| gfx1151 | small | padded | 2.0007 | 0.5911 | 0.294 |
| gfx1151 | small | permuted | 2.0041 | 0.5858 | 0.293 |
| gfx1151 | small | fortran | 1.9917 | 0.5771 | 0.290 |
| gfx1151 | large | compact | 2.4381 | 0.7378 | 0.309 |
| gfx1151 | large | padded | 2.6696 | 0.9656 | 0.361 |
| gfx1151 | large | permuted | 2.5890 | 0.9005 | 0.348 |
| gfx1151 | large | fortran | 2.6298 | 0.8988 | 0.344 |
| gfx1201 | small | compact | 2.4829 | 0.5444 | 0.224 |
| gfx1201 | small | padded | 2.5460 | 0.6690 | 0.266 |
| gfx1201 | small | permuted | 2.5963 | 0.6692 | 0.256 |
| gfx1201 | small | fortran | 2.5710 | 0.6646 | 0.258 |
| gfx1201 | large | compact | 2.6091 | 0.8849 | 0.339 |
| gfx1201 | large | padded | 2.8306 | 1.1066 | 0.390 |
| gfx1201 | large | permuted | 2.7441 | 1.0160 | 0.370 |
| gfx1201 | large | fortran | 2.7066 | 1.0246 | 0.367 |

The paged-KV paired medians are 0.224–0.390 of the pageable public-call cost.
This justifies best-effort native staging for the measured route; it does not
close host cost for all ROCm families or establish FP8/MXFP8/MXFP4 kernel gains.
No physical schedule or quantized-kernel selector is changed.

## Reproduction and remaining gates

Use the exact owning device with matching compiler/runtime paths and
`TESSERA_ROCM_MOVEMENT_DEVICE_PROOF=1`, then run:

```sh
python -m pytest -q tests/unit/test_strided_paged_kv_runtime.py \
  tests/unit/test_public_movement_frontend.py \
  tests/unit/test_native_graph_rocm_movement.py \
  tests/unit/test_pinned_rocm_staging.py
python benchmarks/rocm/benchmark_pinned_staging.py --architecture gfx1151 \
  --candidate-source src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_movement_runtime.cpp \
  --output "$HOME/scratch/pinned-movement-device.json" --rounds 7 --repeats 128
```

Use gfx1201 on its own host. Source/runtime proof and timing scopes must remain
separate for each architecture. Broader image families, symbolic/general paged
layouts, W8A8/MXFP4 coverage, generic scaled-product AD/batching and full-suite
closure remain open. This is a focused contribution to the fifth slice.

## Existing gfx1151 MoE staging profile

The same final runtime also passes a separate seven-round / 128-call
counterbalanced compiler-forbidden MoE comparison. Small input (7,13) and
nine gathered slots measure pageable 0.5002 ms / pinned 0.4932 ms, paired ratio
0.983: the effect is minor and is not a robust speedup claim. Large input
(64,256) and 128 slots measure 1.2058 / 0.4566 ms, paired ratio 0.378.
Changed source/index and retained output checks pass. All windows exceed 20 ms.
The gfx1151/moe.json and moe-recorder.txt files retain exact recorder/source/runtime
fingerprints and raw data. This does not add a gfx1201 MoE package route.

Final host native ownership/math/ABI/audit/diagnostic/pass gates pass 445 tests,
with 17 hardware/tool cases skipped. All 32 generated documents and compiler
plan ownership checks pass. These do not replace the red aggregate
scaled-matmul batching/transpose closure assertions.
