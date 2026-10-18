# Native SM120 paired tensor result generations

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Synchronization key: SM120-NATIVE-JVP-RESIDENT-RESULTS-20261017.

Current candidate extends the existing source/RHS native paired JVP programs
with independent FP32 device results. The original Graph/AD/Schedule/Tile/
Target/LLVM-PTX member images are retained. A matching core/compiler build and
the production three-source native runtime build pass on Super-Bear.
build-receipt.json records successful exits and compiler/provider/source
identities. compiler-build.txt.raw.gz retains the raw core build log.
The profile text mirror normalizes trailing whitespace; its adjacent .raw.gz
retains original console bytes.
The live recorder queries the active CUDA context: RTX 5070, sm_120,
UUID cba12639821a7a104cd3f918f9c0a545. Compiler/provider/source hashes are in
each timing JSON. Exact-device evidence does not transfer to sibling targets.

## Validation

Current-source numerical/lifetime tests cover both active operands, FP16/BF16,
host/resident roots, reordered arguments, retained generations, owner closure,
cache eviction and fresh-process compiler-disabled replay. A textual FP32
RMSNorm consumer follows Graph/Schedule/Tile/Target/PTX and compares both
serial/cooperative results with independent FP64 arithmetic. Context restoration,
foreign refusal, fork-before-lock refusal and asynchronous-consumer cleanup
are tested separately.

stream-controls.txt records five passing current-source cases: two pre-reserved
pending-producer cases, two injected event-retention failures, and one negative
control removing only cuStreamWaitEvent. That negative control must violate the
independent numerical bound. Event-retention failures must observe consumer
completion before a free/download can synchronize it implicitly.

The combined focused lane passes 642 tests with 7 skips and one expected
fork deprecation warning (focused-tests.txt/xml). All 33 generated documents
regenerate; the final drift check passes all 33, and the mypy ratchet remains
at zero errors. The full CPU repeat passes 26,345 tests (5,905 skips; 11 existing warnings).
The compiler-route gate passes 1,629 with zero skips and an executed-proof
certificate; changed code hashes are verified after each gate. The adjacent
saved-LSE forward/JVP/reverse control selection passes 48 RTX5070 cases. Earlier focused
failures found malformed-policy diagnostic expectations and a bare-object
artifact stub; both were corrected. The combined consumer test also exposed a
test readback race: downloading on the allocation session's stream before the
actual writer completed. The corrected test completes the actual writer.

## Performance interpretation

public-uncached.json is the initial public candidate; public-profile-before-cache.txt
attributes repeated serialization/hash work. public-cached-first.json repeats
the named profiles after immutable placement-artifact reuse.
public-cached-isolated.json is the pre-guard repeat after the generated-doc
and mypy processes completed. public-cached-guarded.json is the final native
stream-guard provider repeat, run after the other validation processes ended. public-cached-docs-overlap-diagnostic.json retains
a run overlapping documentation generation; it is not the primary timing proof.

All six profiles use three resident roots and no root uploads. Each is measured
in both first-arm orders, four counterbalanced windows, common repeat counts,
minimum-window admission and independent validation outside timers. CUDA output
placement includes release of the prior unpooled generation. The final generation
is closed outside each profile. Host output placement returns two host arrays.
These are different explicit output endpoints, not equivalent host-output costs.

Initial CUDA/host public ratios were 1.15-1.49. The final guarded-provider repeat is
0.97-1.10: small cases remain slightly slower, larger output cases improve.
All twelve ordered member-image lists match the initial packet. The initial
uncached/cached-first trials share one provider; build-receipt.json records
the distinct final provider rebuilt with the native stream-token guard. Native event controls are recorded separately and do not establish
a kernel-speedup claim. There is no default-route or cross-backend promotion.

The contract is docs/spec/ir/sm120_native_tensor_result_contract.md. Public CI/review,
publication, general layouts/dynamic shapes, FP32 standalone activation and the
original five aggregate slices remain open.

## Native consumer token review

invalid-stream-baseline.json retains two isolated-child crashes for malformed
consumer stream tokens (3 and 4095). The rebuilt guarded provider applies the
existing producer-token boundary before driver calls. guarded-provider-faults.txt
records four passing cases: event-create/record draining, producer-wait negative
control, and native/Python low-token refusal with default-sentinel/recovery
checks. The full guarded-provider focused lane passes 642 cases (7 skips); its own
timing is in public-cached-guarded.json. Pre-guard packets are retained under
their original names. cpu-broad-initial.txt.gz records the two initial broad
failures (uncited packet and source-location-based runtime test). After proper
citation and a public-behavior test replace the brittle inspection, the
20-case focused recovery passes. The final broad and compiler-route repeats above pass; the initial failed
run remains archived and is not reported as completion.

## Reproduce on Super-Bear

The task environment is
`~/scratch/sm120-resident-jvp-result-20261017/env.sh`; it selects the matching
LLVM/MLIR core/NVIDIA compilers, CUDA 13.3 and the guarded native provider.
The checked build command and source/provider hashes are in build-receipt.json.
Use the recorder with `--output` pointing into a scratch directory.
The baseline JSON records all admitted windows, calibration rejections,
device UUID, exact member-image identities and timing endpoint definitions.
Current public CI/review state belongs to the PR; these files certify local
host/device gates and do not claim the five-slice objective is complete.
