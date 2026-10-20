# Native FP32 original producer roots

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: NATIVE-TENSOR-FP32-ROOTS-20261020.

## Contract and implementation

The original rank-two compact FP32 input passes through storage-preserving
ReLU into the existing ordered scalar-FMA matmul consumer. FP16, BF16 and FP32
weights retain their declared storage. Native Graph -> Schedule -> Tile ->
NVIDIA Target -> LLVM/PTX members preserve both source operations, buffer
roles, derived launch metadata, initialization and final-reader retirement.
No half/TF32 conversion, Python numerical lowering or new native frame ABI
is introduced. Native frame dtype1 already owns checked four-byte FP32 roots.

The old compiler admits the direct FP32 matmul leaf but rejects the same
leaf after FP32 ReLU at the whole-program half-root gate. Eight initial
positive compiler/signature cases fail there. The fix removes that obsolete
restriction and widens host/resident root metadata transport; each native
leaf still validates its storage pair and numerical policy independently.

## Evidence

- 170 focused compiler/JIT/frame cases pass, including 21 new positive and
  adversarial root cases. Two older half-only refusal fixtures are corrected
  while preserving unsupported FP64, pitch and malformed metadata controls.
- 481 shared dtype/op/diagnostic/pass/audit gates pass.
- On RTX5070 SM120, 18 final host/resident cases pass: all three RHS storages,
  aligned/ragged shapes, reordered arguments, changed inputs, retained owner
  allocations, immutable snapshots, serialized replay and independent pending
  producer streams. The earlier 12-case run overlaps and is not added.
- Numerical output is checked against an independent FP64 product with the
  standard FP32 gamma_k accumulation enclosure. All nine recorded envelopes
  reject premature FP16 root conversion and a +0.01 arithmetic perturbation.
- The matching compiler is db3a4e86bb42c937d4da70fb98a45266e8e758241c7c773115906d9a6ab871ec.
  Runtime remains the prior immutable 03bd30ed513d3362d915eb5cca7f61cb7e850b2cd120b2e89f362a210a88ffde.
  Source/compiler/provider/device hashes are preserved in the JSON receipts.
- Full CPU passes 30,788 cases with 3,710 declared skips. All 33 generated
  documents and the 11 audit lifecycle cases pass. The first production
  compiler sweep has 2,662 passes and one obsolete producer-refusal failure.
  That fixture now requires both retained native stages; its targeted lane
  passes 129 cases. The final compiler rerun passes all 2,663 cases with
  zero skips (JUnit execution verified). Publication gates remain active.

The first attempted focused command named a missing test path and ran no
tests (fp32-root-green.txt.gz). The next lane found two stale refusal fixtures
(fp32-root-focused.txt.gz); the corrected lane passes 170 cases. Raw failed
and successful runs remain losslessly compressed with SHA256 identities.

## Timing

Reproduce from the owning WSL checkout with the matching compiler/runtime:

    python -m benchmarks.nvidia.record_native_tensor_mlp --fp32-roots --output /tmp/fp32-roots.json

fp32-root-timing.json has nine correctness-gated shape/storage rows.
Cold calls, completed public host samples, producer/consumer event regions
and whole-program captured/enqueued regions are separate domains. Event
regions are independently measured and must not be added. Completed public
medians range from 0.270 to 0.412 ms in this packet; no comparison against a
precision-changing half-root baseline or default-policy promotion is claimed.

## Siblings and remaining requirements

Apple: not applicable to Metal physical admission; independent program proof required.
ROCm: no RDNA schedule or HIP ABI changes; gfx1151/gfx1201 parity remains separate.
x86: no AVX-512 consumer change; independent storage/lifetime proof required.

General pitched/bounded/dynamic roots, arbitrary producer operators, encoded
storage and broader AD remain open. This named static storage increment
does not close W1.1 or any aggregate five-slice program.


## Final local delivery checks

The zero-error mypy ratchet and all five surface audits pass. Examples have
two declared optional-extra skips; benchmark/research/tool/test surfaces have
no failures. All 33 generated documents are in sync; 41 final operator,
attribute and audit lifecycle cases pass. Canonical metadata now separately
names the original FP32-root producer proof, standalone activation proof and
fused accumulator epilogue proof. PR994 is merged as
4f53a65c884aada5eaf8da8cef3c2a1a9bae0d23 after exact-head CI and resolved
reviews. This successor is synchronized with that main. Its only rebase
conflict was additive specification text; both contracts are preserved.


## Post-main synchronization proof

All nine measured source/Schedule/Tile/Target IR boundaries, checked frames
and native image hashes are unchanged after rebase (post-main-equivalence.json).
Compiled compiler/runtime C++ source is unchanged and matches the immutable
db3a4e86 compiler receipt. The new main contributes the source-fresh census
and bounded standalone activation dispatch gate, without changing these kernels.

Post-sync host admission/scalar/ownership/dispatch checks pass 170 cases.
RTX5070 post-sync regressions pass 27 cases: the 18 original FP32-root cases
and nine standalone activation controls. These overlap earlier runs and are
not added as unique test counts. Final publication CI remains required.


### Final-store integration follow-through

Explicit matmul output storage admits the existing canonical fp16, bf16 and
fp32 spellings in eager execution, abstract tracing and textual Graph shape
inference. Accumulation remains FP32 through bias, activation and residual;
only the completed result is converted. The native SM120 scalar route also
admits FP32-by-FP32 inputs consistently with its existing native contract,
and allocates BF16 descriptor results using the existing typed owner.
This does not widen the half tensor-core output contract or the ROCm
same-FP32 descriptor envelope.

The initial expanded device test passes six FP16 stores and rejects six
BF16 stores at frontend validation. After frontend integration, all twelve
producer final stores pass; an additional warm direct FP32-by-FP32 BF16
control exposes stale Python Schedule admission. The corrected full device
rerun passes all 33 cases. Host frontend controls pass twelve cases; the preceding
shared registry lane passes 493 and compiler regression lane passes 230.
Full CPU validation predates the last admission edit and must be refreshed.
No new native ABI, Graph dtype or broader AD certificate is introduced.

The corrected device rerun passes all 33 cases: 18 original host/resident
producer cases, twelve FP16/BF16 final stores and three direct scalar BF16
calls with warm tracing disabled. The final focused compiler/frontend gate
passes 233 cases. Original nine FP32 timing rows remain the measured
performance envelope; no BF16-store latency claim is added.
