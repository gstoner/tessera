# Ordinary SM120 JIT native tensor program evidence

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: NATIVE-TENSOR-JIT-20261020.
Contract: docs/spec/NATIVE_TENSOR_JIT.md.

## Exact source and device

RTX5070 SM120, driver610.88, GPU-cba12639-821a-7a10-4cd3-f918f9c0a545.
Compiler ff8bad652b1a2c0f75d58c5589e37433e4774e5032664e3d202fa112cd59e070:
LLVM/MLIR23.1.1, optimized with assertions.
Provider72aaf57b4819f6aaf7931e59bbeb4cb68b8bd794e68cb2bb9108924eb6547e45.
Per-arm receipts bind GPU, compiler, provider and actual Python file hashes.
The compiler receipt records the private checkpoint and modified C++ hashes;
it does not call an uncommitted build an exact Git head.

The route preserves actual Graph/Schedule/Tile/Target stages, uses compiler
admission and retains a checked C++ owner. Python transports metadata and
arguments; numerical kernels and operation-family selection stay native.

## Validation

- Historical ordinary-JIT red control:8 failures. Six legacy artifacts lacked
  the Target controller; explicit-FP32 RMSNorm refused twice.
- Matching compiler:13 ordinary/artifact/boundary cases,8 explicit packages
  and 19 host capacity/transport cases pass. These are the 40 successful cases
  in jit-matching-device-final.txt.gz.
- That mixed log also has45 owner failures caused exclusively by absent
  committed image fixtures. After supplying unchanged fixtures, all 45 owner
  tests pass in jit-matching-owner.txt. The failed log remains available.
- Resident JIT:32 cases pass across FP16/BF16, four source roles, and separate,
  shared, legacy and per-thread producer streams. Host coercion and warm
  tracing/tools are forbidden; delayed producers complete before consumption.
- Eight additional LayerNorm/softmax and half-output cases pass (6 initially,
  followed by 2 LayerNorm cases after correcting an unsupported test keyword).
- The final combined device run passes all53 new ordinary/artifact/boundary/
  resident cases; its28 failures are the temporary legacy selector returning
  None instead of the dispatch sentinel. The corrected legacy/adjacent suite
  passes all118 cases. The earlier adjacent run's12 failures were predecessor
  manifest assertions intercepted by the migrated route. All logs are retained.
  Five ABI-specific test functions explicitly select their predecessor;
  the new53-case suite exercises unmodified ordinary dispatch.
- Final production compiler route:2594 passed, zero skips. Resident admission:
  19 passed. Ruff and Mypy pass with zero errors/baseline. Full CPU run:
  30783 passed,3710 skipped, with one new-recorder naming failure; the tracked
  benchmark index now names it and the final focused naming/citation/audit/diagnostic/pass gate passes319 tests.
  Counts from overlapping runs are never added together.
- Capacity red:5 failed assertions,3 valid boundaries. Four exposed admitted
  ABI mismatches; the oversized source also checks diagnostic priority.
  Admission, typed verification and export now enforce64 stages/128 buffers,
  CUDA geometry and a256-byte-aligned64-bit arena. Extreme metadata tests
  allocate no described memory. The full64-stage program executes numerically.
- All8 original source/Target/image programs remain byte-identical.
  Owned-result pinning has a host transport test; it does not establish
  FP32 owned results as supported half-storage roots.

## Paired measurements

Recorder: benchmarks/nvidia/record_native_tensor_jit.py.
Each arm uses21 samples of 16 completed public calls, seeded M17/N11/K32 inputs,
and independent float64/half-rounding oracles before/after timing and profiling.
Decoration and cold compile/bind costs are separate. Warm public calls forbid
subprocess tools; the candidate also forbids tracing.

Order: native A1, legacy B1, legacy B2, native A2. Compiler/provider are common.
All4 legacy source controls match merged main fac21f5bc byte-for-byte.
Candidate restoration and image stability across orders are checked.
The2 unsupported legacy explicit-policy RMSNorm cases have no timing denominator.

Six comparable cases yield completed-public legacy/candidate ratios
1.8853-2.4551x across both orders. Candidate cold calls cost 1900.25-3205.02 ms;
decoration costs 88.96-131.70 ms separately. These are tiny-program latency
observations, not throughput or universal selector promotions.

Captured/enqueued program and stage windows exclude host copies and are
independent/nonadditive. Two-sided captured windows are slower:
candidate0.0064245-0.0073845 ms, legacy0.0046815-0.0047615 ms. Both producer images
match exactly. Consumer instructions/directives match after entry renaming
and comment normalization. Runtime/capture/device attribution remains open;
no pure-kernel speedup or physical schedule promotion is claimed.

consumer-attribution.json references existing native images and two compressed
unique legacy controls. Original bytes and hashes remain intact.

## Reproduce and remaining work

On the owning configured host:

    python -m pytest tests/device/nvidia/test_native_tensor_program_jit.py -q
    python -m benchmarks.nvidia.record_native_tensor_jit --route native --label candidate --output /path/to/candidate.json

run-jit-abba.sh records the actual scratch source swaps with restoration on exit.
Its absolute paths are historical; use an isolated validation checkout.
--route validates implementation choice; labels never suppress correctness.

Final CPU/docs/audit and adjacent-route gates, cold compilation and device-window
attribution remain open. Wider producers/layouts/dynamic/AD/encoded storage,
compiler-free restoration and all five aggregate programs remain open.

After measurement, the native execution row adds named proof annotations.
execution-proof-annotations.json verifies all remaining module AST is unchanged;
raw benchmark identities retain the actual measured source hashes.


## Final source reconciliation

final-source-identity.json binds the final Python/device-test/recorder files to
the owning-device overlay and confirms native compiler-source hashes still
match its build receipt. Measured JSON files are unchanged.
execution-proof-annotations.json proves the execution-matrix delta is limited
to named evidence fields. resident-admission-host-equivalence.json proves the
later resident preflight correction leaves the measured host helper path and
all other helper code unchanged. Unsupported resident storage/pitches decline
before the compact-half parser; supported half metadata remains fully checked.

Raw pytest failure logs containing original trailing whitespace are stored losslessly as gzip; raw-log-encoding.json records uncompressed hashes. No log content or measured JSON was normalized.
