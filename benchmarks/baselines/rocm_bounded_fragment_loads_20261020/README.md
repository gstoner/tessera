# Guarded bounded WMMA fragment loads — 2026-10-20

Owner: ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6.
Sync: ROCM-BOUNDED-CONTIGUOUS-PACK-20261020.
Base: merged PR995, 71564e7679bee9efeda5c0b18ac2a6430a9a7936.

## Change and boundaries

The native MLIR ROCm materializer keeps complete contiguous K fragments as
vector loads after proving the K range and a nonempty slow dimension. Invalid
slow lanes load a safe address and select a zero vector. Partial K and strided
K retain the existing scalar guarded sequence; packed MXFP4 B retains its
packing path. This preserves generic Tile zero-fill independently of a consumer
store mask. No new Graph/Schedule/Tile operation, dtype, numerical policy,
backend fast path, pass or runtime ABI is introduced.

The evaluated one-wave GPU annotation was removed. Although native launches
already use 32 threads, that annotation changed generated machine code and
regressed FP32-scaled K32 at M208/N1024/K4096 by 3.79x and M200 by 1.66x versus
vector loads alone. `rejected_geometry.json` records that rejected experiment;
its 6f2 compiler is not the retained candidate. Its reference-source CLI
argument initially named the annotated source; the checked-in record corrects
that source hash with explicit original-record/hash errata. Timing values and
frozen compiler identities are unchanged.

## Compilers and numerical proof

- Control: db3a4e86bb42c937d4da70fb98a45266e8e758241c7c773115906d9a6ab871ec.
- Retained vector candidate: 276e09dfe344a910b345a71de4d39eeaf430a3a18a290fc052b1489cc648adb2.
- Rejected vector-plus-geometry compiler: 6f2a995a7f6e0eadb4f1cf6262969eef8b082852c55b40d097074fdf87ad7280.

`native_load_change.patch` and `frozen_vector_identity.json` retain the native
build change. The maintained fixture tests FP16, BF16, both OCP FP8 types and
both mixed FP8 pairs, signed INT8 and signed INT4, four memory-order pairs,
twelve short/partial/full-K and ragged M/N shapes, tight physical capacities,
unmasked full output stores, input immutability and an output canary.

The exact retained compiler passes all 32 gfx1201 cases (384 inner shapes)
and 16 gfx1151 cases (192 inner shapes). The 16 gfx1151 FP8 cases are explicit
ISA exclusions, not executed proof. gfx942 has compiler-only structural proof.
No sibling architecture inherits these measurements.

On gfx1151, strict equality with integer-valued floating references initially
failed by at most 1.90735e-6 on BOTH old and new compilers. All 96 float cases
match bitwise between those compilers, with exact zero-filled invalid outputs.
An independent hipcc implementation using AMD WMMA builtins reproduces all
24 distinct FP16/BF16 shape outputs bitwise, without Tessera compiler/runtime.
The maintained floating check therefore uses gamma16(FP32) times sum(abs(A*B)),
with no absolute tolerance floor. Integer results, zero-contribution elements,
canaries and input bytes remain exact. This corrects the test oracle and does
not relax a production numerical policy. Sources, arrays, receipts and original
failed assertions are in `gfx1151/`; the final retained-vector gate is separate.

## Timing domains

All FP8 format runs identify the live RX 9070 XT/gfx1201, verify independent
quantized references before/after each admitted window, and alternate A/B and
B/A. The first and counterbalanced format packets contain all 24 rows, including
explicitly approximate folded MXFP4. Every admitted resident-graph device
window is at least 10 ms. These are device plus dispatch measurements, not
ordinary frontend/JIT latency and not a new comparison against AITER.

Repeated M200/N1024/K4096 results:

| Format | Control | Vector candidate | Candidate/control |
| --- | ---: | ---: | ---: |
| FP8 K128/N128 | 205.6 us | 86.3 us | 0.420 |
| FP32-scaled FP8 K32/N1 | 1655.5 us | 108.3 us | 0.0652 |
| MXFP8 K32/N1 | 60.53 us | 60.61 us | 1.005 |
| Approximate folded MXFP4 | 49.68 us | 49.71 us | 1.002 |

Small differences are not promoted. The first MXFP8 +2.1% observation did not
repeat at that magnitude. Both historical M200 short-K regression shapes and
the K1536/M197 ragged row are retained in the packets.

`completed_public_calls.json` separately counterbalances completed runtime
calls, including staging, copies and completion, excluding compilation and
validation. For M200/N1024/K4096 FP32 output, control/candidate are 4.557/4.335 ms;
BF16 output is 4.497/4.498 ms. A large kernel gain is not a comparable public-call
gain. This is the public descriptor runtime call domain, not full cold @jit.

## Exact packed MXFP4 at M256

`exact_mxfp4_member_domains.json` executes the actual compiler-owned exact
packed Graph/Schedule/Tile primal, gate-JVP and reverse programs, with independent
FP64/group scaling, the required BF16 store and gate-only derivatives. Four
shapes include K128/N80,N160 and K4096/N1024,N2048. All numerical errors are zero.
Captured member, ordinary interleaved native-program and completed public-call
times stay separate; captured grouping is a different diagnostic schedule.
Some small member windows are below 10 ms and do not support fine performance
promotion. The grouped matmul medians at K4096 are 1.148/2.216 ms versus only
0.0082/0.0118 ms combined cast/gate work. The observed column cost is inside
matmul, but this does not identify an A-fetch/LDS cause or resolve Radiance.

`record_paired_exact_mxfp4.py` additionally compares control and candidate primal
programs in alternating windows using the SAME native program owner API. It
reuses duration admission, keeps allocations live with byte-identical operands,
checks exact results before/after every window, records native-image identities,
and requires at least 10 ms per complete window. HIP spans include host launch
gaps; they are neither kernel-only nor completed public-call time. The runtime
returns per-invocation milliseconds; the harness converts to total window once.
`exact_control_identity.json` confirms control correctness and records differing
matmul images with unchanged logical program digests. No blanket MXFP4 speedup
or selector promotion is claimed.

The reversed-shape 16-window repeat gives candidate/reference program ratios
1.006 at N1024 and 1.014 at N2048. The first N1024 ratio of 0.711 does not
repeat; the N2048 ~1.4% slower span repeats. Those spans include host launch
gaps and do not identify a kernel-only regression. Exact MXFP4 retuning and
controlled A-fetch/LDS attribution remain follow-up work.

## Reproduction

Use a matching LLVM/MLIR 23.1.1 compiler build on the owning host. Freeze the
control, vector-only patch and rejected geometry builds separately; never rebuild
an executable used by a running measurement. Set TESSERA_OPT, MLIR_OPT, PYTHONPATH
and the checked native ROCm runtime libraries to those builds. Verify the live
architecture before device tests; gfx1151 uses Princess-Luna and gfx1201 uses
Tajasaurus. All project tests here run in host WSL.

```sh
python -m pytest tests/unit/test_rocm_bounded_fragment_pack.py tests/unit/test_paired_exact_mxfp4_recorder.py tests/unit/test_interleaved_graph_duration.py -q
python -m pytest tests/device/rocm/test_bounded_fragment_pack.py -q -rs
python -m benchmarks.rocm.record_paired_exact_mxfp4 --compiler "$CANDIDATE" --reference "$CONTROL" --output exact-paired.json
python -m benchmarks.rocm.record_native_encoded_scaled_gate --formats mxfp4 --mxfp4-shapes 256,128,80 256,128,160 256,4096,1024 256,4096,2048 --output exact-members.json
```

For the independent gfx1151 hardware diagnostic, copy compare_float_control.py,
check_direct_wmma.py, original_bounded_fixture.py and direct_wmma.hip from
`gfx1151/` into an owning scratch source snapshot root. Supply the frozen
control as tools/tessera-opt-control and the retained candidate as
tools/tessera-opt-vector. The original fixture is imported as an isolated
module; the maintained final device fixture stays unchanged. Build the HIP
shared diagnostic with `hipcc -O2 --offload-arch=gfx1151 -fPIC -shared
 direct_wmma.hip -o direct_wmma.so`; run compare_float_control.py then
check_direct_wmma.py. Both the 96-case control comparison and the independent
24-case builtin comparison were rerun with the exact retained vector compiler.
Their source/binary hashes are recorded independently of the final test gate.
Timing JSON keeps individual windows on one line; `timing_formatting.json`
records byte hashes and verified unchanged data for those formatting changes.

`validate_packet.py` checks frozen compiler identities, paired inputs, window
admission and diagnostic receipts without requiring a GPU. Run
`python benchmarks/baselines/rocm_bounded_fragment_loads_20261020/validate_packet.py`
to reproduce those data checks.

## Local delivery gates

- Full CPU lane: 30,813 passed, 3,721 declared skips (four workers).
- Full production compiler lane: 2,692 passed, no skips; executed-case gate passed.
- Final focused diagnostics/pass/lifecycle/recorder gate: 322 passed.
- Fragment/recorder/duration gate: 39 passed; tracked naming/lifecycle gate: 19 passed.
- Ruff passes; mypy ratchet has zero errors against baseline zero.
- Generated documents and compiler-plan transition checks pass in host WSL.

The initial full CPU run had one naming-gate failure because the new packet
README references were not yet in the Git index. Staging the documented files
restored that gate; the complete staged-tree rerun above passed. Historical
failure and final execution receipts are kept distinct in `validation.json`.
Original pytest transcripts have trailing whitespace removed only, with both
byte hashes in `transcript_normalization.json`.

Delivery identities, coverage and log hashes are in `validation.json`; file
integrity is recorded separately. Wider shapes/layouts,
public staging overhead, repeated exact-MXFP4 attribution and Radiance/model
quality remain open. The original five-slice objective is still open.
