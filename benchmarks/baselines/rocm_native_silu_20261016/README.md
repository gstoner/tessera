# Native gfx1201 SiLU activation

Owner: ROCM-NVFP4-INGEST-1 / E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Sync: ROCM-NATIVE-SWIGLU-ACTIVATION-20261016.

RX9070XT/gfx1201, LLVM/MLIR23.1.1 assertions, ROCm10.0. This is an isolated
activation dependency, not full NVFP4 MLP or model-quality closure.

45 owning numerical/JIT/portable replay cases pass. The unit file now has 60
cases: 47 compiler-dependent proofs and 13 pure frontend/capability cases.
Fresh-process replay forbids compiler access and eager/legacy arithmetic;
FP16/BF16 matching-store and widened FP32 outputs are checked independently.
Adjacent native math/widening and owner lifetime tests pass 91 cases. Focused
registry/compiler gates pass 603 cases; ABI/document gates pass 32 cases.
Counts have different overlapping scopes; they are not summed as coverage.

Two ordered trials each contain 15 correctness-gated rows and all raw samples.
Both compare legacy/native device windows with matching input/output storage;
explicit widened outputs have no mismatched legacy comparator. HIP events
exclude allocation/transfers/readback, but may include host submission gaps.
Public-call samples include checked dispatch, staging, transfers and completed
readback; cold calls also include compilation/cache lookup. Domains are separate.

At 256x12288 native device medians are 25.47–27.56 us, and same-store
legacy/native ratios are 0.993–1.018 across arm orders. No speedup is claimed.
Completed public calls at that shape take 4.84–6.13 ms. Smaller rows vary with
trial order; no performance promotion or host-overhead closure follows.
The earlier exploratory packet/logs remain in owned scratch, not final evidence.

Compiler archive and installed-tool receipts bind the unchanged compiler core
built at 05481f153. Packets additionally hash current changed sources, the
compiler and candidate HIP provider. The native owner is rebuilt separately;
source timestamps were never changed to manufacture compiler freshness.
Original runtime providers are preserved. gfx1151 and sibling backends receive
no physical execution or timing claim. Full resident views/lifetimes,
activation quantization/downprojection and original-model quality remain open.

Reproduce on Tajasaurus after sourcing the candidate environment:

```sh
python -m benchmarks.rocm.benchmark_native_silu_schedule --output native-first.json
python -m benchmarks.rocm.benchmark_native_silu_schedule --legacy-first --output legacy-first.json
```

CI lane classification is explicit: 47 compiler-dependent proofs execute in the
production compiler lane, while 13 pure frontend/capability tests stay in the
CPU lane. Existing production compiler-route coverage passes 1,265 tests with
zero skips. The generated-document gate confirms 33 documents are synchronized;
plan ownership and public claim lint pass. The broad host unit lane recorded 26,109 passes and 5,902 skips, with one
recorder-citation failure. The missing citation is fixed without changing its
test; 146 follow-up tests and 556 final registry/ABI/compiler/recorder gates
pass. Final CI remains a required publication/merge gate.

Final timing packets bind committed source a6ae03792190f30a6c39eee5642afe488c27ca98.
The earlier uncommitted-source trial is retained in owned scratch/history; these
packets replace it as the final measurement evidence. No arithmetic changed.

Recorder: [`benchmarks/rocm/benchmark_native_silu_schedule.py`](../../rocm/benchmark_native_silu_schedule.py).

Final public selector review keeps unproved gfx1151 SiLU on its existing route
selection; low-level compiler artifact recognition remains shared. All 45
gfx1201 device cases pass again after that selector correction. Timings are
explicitly from a6ae037921, before this metadata-only selector follow-through;
the compiled arithmetic, native ABI and HIP provider are unchanged.

Final local gates: corrected broad CPU lane 26,067 passed/5,902 skipped;
production compiler lane 1,312 passed with zero skips and executed-proof gate.
The complete shared provider preserves all 47 baseline exports and adds only
`tessera_rocm_math_launch_v2`. Its 87 owning gfx1201 cases include all 45 SiLU
cases plus existing bounded/public/prepared NVFP4 programs. The earlier
math-only scratch provider was a build-isolation aid; final timing packets
will bind the complete provider before publication.
