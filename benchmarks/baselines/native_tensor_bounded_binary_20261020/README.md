# Native bounded binary tensor program candidate

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: NATIVE-BOUNDED-BINARY-PRODUCERS-20261020.
Contract: docs/spec/NATIVE_TENSOR_PROGRAM_BOUNDED.md.
Cited by docs/audit/compiler/INTEGRATED_COMPILER_LOG.md and all four backend queues.
Recorder: benchmarks/nvidia/record_native_tensor_bounded_binary.py.

The source-owned compact FP16/BF16 stored-add -> plain FP32-output matmul
retains its dynamic M policy through native Graph/Schedule/Tile/NVIDIA Target
and PTX, capacity/active-view verification and one retained C++ arena.
Traces at M=1 and M=17 preserve the declared larger capacity. Public source
certificates and compiler-derived cache keys retain fixed K/N and storage.

## Initial candidate proof

- Four initial compiler failures locate the static-only ownership boundary.
- 95 focused compiler/transport cases and 43 host-only transport cases pass.
- 462 focused public/ABI/diagnostic/pass/lifecycle checks pass.
- 16 recorder argument cases reject invalid packets before hardware.
- Eight exact RTX5070/SM120 tests pass: shrink/grow M=1/2/5/17/64/65,
  K32/N11 and ragged K35/N19, independent FP64 stored-add then FP32 gamma-K
  comparison, original-input immutability, independent outputs, changed-M
  serialization replay, native invalid-span/equality/capacity refusals that
  preserve the previous generation, and pending resident producer streams.
- 120 exact RTX5070 static owner/MLP/FP32 regressions pass.
- Exact gfx1201 composed MXFP4 control passes. Its images and neighboring
  FP8 K128/K32, MXFP8 and approximate folded MXFP4 images are byte-identical.
- Focused Ruff and zero-error mypy pass. The missing <algorithm> include in
  the first provider build is corrected before native tests; that failed log
  is retained separately. The old frozen control's source-staleness warning
  is intentional baseline attribution, not a candidate tool selection.

Compiler: f1ce99e7b0cc6cd94159f566ebecc40ad0cef6ce5226cb0119e84ef072269160.
Provider: 06fcdb02b05e52f681d35427afc76a8b3a6c772d4fa88a281b69fb70006c9465.
Provider exports: 103, adding only tessera_nvidia_native_frame_prepare_v2.

## Initial capacity-model measurements

The 12-row timing run completes at M=17/1/256/65/5/200 with K4096/N256
for FP16/BF16. Each storage reuses one program and one allocation. All measured
Graph/Schedule/Tile/Target texts, frames and member PTX images are independently
reproduced byte-for-byte in initial-candidate/owning-device/programs/. Four M200/M256 profiles
clear all 24 of their captured duration gates; the other 48 captured windows
are diagnostic because their producer region is too short. At M200 the
admitted program medians are 100.11/100.14 us (FP16/BF16); at M256 they are
104.54/104.53 us. Completed public medians are separate: 1.323/1.405 ms at
M200 and 1.669/1.631 ms at M256. No baseline comparison is claimed.

The first full compiler gate found one stale unknown-schema fixture, which
used newly supported schema2. Version99 preserves unknown-version refusal;
the 107-case focused correction passes. The initial failure and correction
are retained. Initial broad gates completed with 30,905 CPU passes / 3,721
declared skips and 2,770 compiler passes. Current proof follows below. No performance improvement or aggregate closure is
claimed. The independent kernel regions are not additive; completed-public
wall time includes binding, transfers, allocation and readback. Any event
window below 10ms remains diagnostic. First-public timing follows explicit
compile/prepare probes and is not process-cold.

General producer DAG/layout/AD, dynamic K/N and mixed storage remain open.
Sibling architecture execution cannot be inferred from SM120 evidence.

## Current active-memory SSA candidate

Current compiler: 73739736aa3449078c9c9d0c5a508ca365bf71b0bfd22605556b12f83ef8c4cc.
Provider remains 06fcdb02b05e52f681d35427afc76a8b3a6c772d4fa88a281b69fb70006c9465.
Dynamic borrowed-root memrefs and explicit memref.dim/subview SSA now separate
actual views from owned capacity allocations. Two stronger typing assertions
fail on the old model; 176 focused controller cases and 487 selected shared/
public/transport checks pass after refinement. The refreshed RTX5070 cohort
passes 32 combined cases: 24 compiler cases and eight exact-device cases.
Frames, original source, member IR and PTX are independently byte-identical to
the initial candidate. Schedule/Tile/Target host memory SSA intentionally differs.
Current full compiler: **2,779 passed, zero skipped**. First active-SSA CPU run: 30,902
passed / 3,721 declared skips; three audit-routing failures share a missing
historical log field. The corrected entry passes all 19 routing/lifecycle
checks. The raw failure and correction receipts are retained.
All five surface audits pass (two examples skip optional notebook extras);
Ruff, claim lint, current gfx1201 composed numerics and five byte-identical
format image controls pass. Eight static programs retain 18 images.

The current measured artifacts and device receipts are under
`active-ssa/owning-device/`; the full timing JSON and summary bind compiler,
provider, IR/frame and image identities. All 12 variable-M rows reuse one
allocation per storage. At M200 the admitted native-program medians are
100.14/100.10 us (FP16/BF16), with completed-public medians 1.408/1.369 ms.
At M256 they are 104.56/104.63 us and 1.603/1.669 ms respectively.
Only those four profiles clear all 24 captured duration gates; the other 48
captured windows remain diagnostic. Independent producer/consumer/program
regions are not additive. No baseline speedup or route promotion is claimed.
The integrity manifest binds every packet file. Final local gates are below;
publication CI/review and delivery remain.

## Final local delivery

Final local delivery: the CI-equivalent CPU lane passes **30,894** with
**3,721 declared skips**, zero failures/errors. The 11 cases omitted versus
the first CPU run all have explicit performance markers, matching CI's selector;
no other test IDs differ. The initial log-field failure and correction are
retained. All **2,779** compiler cases execute with zero skips. The 323 final
documentation/registry cases, 790 ODS cases, all five surfaces, 33 generated
documents, Ruff, zero-error mypy and Graphify pass. Exact-head public CI/review
and delivery remain pending; all five aggregate programs remain open.

## Reproduction

On the owning SM120 host, configure the Python environment, matching native
provider and the LLVM/MLIR 23.1.1 compiler identified by the packet. From the
repository root:

```sh
python -m benchmarks.nvidia.record_native_tensor_bounded_binary --output /tmp/native-bounded-binary.json
```

The public source used by the tests and recorder is:

```python
import tessera as ts

@ts.jit(target="nvidia_sm120", shape_bounds={"M": 256})
def lhs_add(a, residual, b):
    return ts.ops.matmul(ts.ops.add(a, residual), b, output_dtype="fp32")
```

A/residual use matching compact FP16 or BF16 storage and active M; B has
fixed K/N. Retain the JIT object across calls to reuse its native owner.

## Raw artifact storage

Compiler-emitted MLIR/PTX and raw text/XML receipts are stored with lossless
gzip compression. Decompress before reproducing the original IR/image hashes;
`lossless-artifact-storage.json` records each original MLIR/PTX digest. This
preserves compiler-printer whitespace byte-for-byte. The packet integrity
manifest separately hashes the committed compressed files.

## Provider ABI review correction

Provider ABI review: bounded admission now requires prepare_v2 before
tracing or compilation; static calls continue to require prepare_v1. An older
v1-only provider leaves the existing route eligible and creates no cache entry.
One red prerequisite case becomes eight passing missing/empty/v1/v2-provider
cases; the affected JIT/compiler/transport cohort passes 97. Fresh RTX5070
validation passes eight device cases plus 32 host transport cases (40 total).
Ruff and zero-error mypy pass. Native compiler/runtime bytes are unchanged.
The original timing packet retains its recorded pre-review Python identity;
no new performance claim follows. See the packet's provider-review/ receipts.

The full CPU/compiler receipts and `active-ssa/local-gates/final-source-identity.json`
record the initial reviewed source before this guard correction. The current
Python overlay and fresh device probe are bound in `provider-review/`.
`source-delta.json` verifies that the guard is the only JIT source change;
`pr999-provider-device-identity.json` binds both changed Python files and all
unchanged native compiler/runtime sources to the fresh owning-device run.
