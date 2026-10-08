# Ordinary SM120 attention frontend integration

Owner E2E-REAL-6 / NVIDIA-LSE-1 / FRONTEND-IR-MEDIUM-1.
Sync NVIDIA-ORDINARY-ATTENTION-2026-10-06.

Ordinary primal f32 host-array @jit attention now selects canonical native
compilation and checked runtime execution. Python binds declared buffers and
seven dimensions; Graph/Schedule/Tile/Target/LLVM owns arithmetic. Native
Schedule validates distinct function argument roles, allowing Q/K/V and full
bias argument permutations without reordering or replacing the caller Graph.
Descriptor shape projection follows the retained Graph SSA operands.

The standard forward image returns O. Existing saved-O/LSE checkpoint and
native JVP/VJP packages retain their separate pairing and lifetime contracts;
that earlier increment did not expose a public saved-LSE API. The later
public integration below supplies that API; general composed AD remains open.

- device-tests.txt: 28 RTX5070 cases pass, including all six Q/K/V orders,
  full/causal, grouped heads, batch two, both Sq/Sk orderings, unequal D/Dv,
  full bias first/middle arguments, no-eager/no-recompile warm calls,
  changed values and serialized descriptor replay.
- shared-regressions.txt: 53 passed, 18 skipped. Apple/ROCm/x86 attention
  projection and backward packages retain their ordered contracts; these
  host tests establish no sibling physical execution.
- rtx5070.json: 36 independent FP64-oracle rows pass before and after timing.
  Cold ordinary calls trace automatically without test preparation. Three
  public wall samples and three resident native C++ 100-launch CUDA-event
  windows per row are retained, with 20 warmup launches.
- Source, matching compiler/runtime binary hashes, queried GPU UUID/driver/SM,
  image/entry/ABI and Schedule/Tile provenance are recorded. Event windows
  include native/driver dispatch gaps; public walls include validation,
  host allocation and transfers. No speedup or default strategy promotion.

General producer composition, dynamic attention, wider primal dtype semantics,
broadcast ordinary bias, higher AD and asynchronous owners remain open.
FP8/MXFP8/MXFP4 gates remain independent.

Current packet warm public medians: 1.209-1.966 ms. Resident dispatch-event medians: 8.235-26.036 us per launch. These scopes must not be compared as a speedup.

## Direct saved-LSE Graph import (2026-10-06)

Sync NVIDIA-SAVED-GRAPH-2026-10-06; owner NVIDIA-LSE-1 /
FRONTEND-IR-MEDIUM-1 / E2E-REAL-6. The forward/backward packagers now send
the caller's original Graph through native MLIR import. Python deep-copies
frontend binding metadata and decodes the native physical contract; it does
not synthesize checkpoint Graph operations. Native policy validation rejects
unhandled attributes, and SSA roles survive reversed function arguments.

The Graph dialect declares optional forward row LSE and the existing backward
Graph spelling. Shape verification precedes Schedule selection. Graph-to-Schedule
declares the attention dialect dependency; the Python manifest and pass metadata
use its actual ODS namespace, tessera_attn.

- Matching assertions-enabled compiler build succeeds: saved-graph-build.txt.
- Original Graph, policy, immutability and checkpoint contracts: 56 pass,
  saved-graph-unit.txt.
- RTX5070/SM12.0: 45 device checks pass, saved-graph-device.txt. Includes twelve
  saved/recompute forward/backward oracle cases with plain/bias, three shapes,
  reversed argument order, checkpoint lifetime checks, and ordinary/JIT AD
  regressions.
- Registry/audit gates: 333 pass before metadata changes; final metadata,
  diagnostics and dialect-manifest gates: 330 pass, saved-graph-metadata.txt.
- Seven timing samples with 100 resident launches per CUDA event window and
  three host-array launches per wall sample: saved-graph-rtx5070.json.
  Shape [B,Hq,Hkv,Sq,Sk,D,Dv]=[1,4,2,16,16,32,32].
  Forward median 26.725 us device / 1.023 ms end-to-end;
  backward median 80.041 us device / 1.362 ms end-to-end.
  The two timing domains are separate; no speedup or selector promotion is claimed.
  O, LSE and gradients pass independent numerical checks before and after timing.
  The packet records live GPU UUID, SM, driver and compiler/runtime binary hashes.

General tensor composition, dynamic shapes, wider storage, broader AD and
sibling physical execution are not closed by this bounded Graph admission.

Final shared Graph/ODS result, attention forward/backward, native program and
audit regressions: 112 passed, 19 environment skips (saved-graph-shared.txt).
Graphify update was attempted from the authoritative WSL repository but the
CLI is unavailable there; this is a tooling limitation, not device evidence.

## Native policy compatibility and verifier coverage

Neutral integer scale/modifier spellings are consumed natively. Frontend
serialization gives the ODS F64 dropout attribute its typed floating spelling;
the recompute Tile control likewise prints valid floating literals for integral
scales. Backward Graph verification preserves both recompute and saved forms,
including output-width checks. Native source admission never drops an active
unhandled modifier.

- saved-policy-build.txt: matching production compiler rebuild succeeds.
- saved-policy-unit.txt: 164 host Graph/checkpoint/registry contracts pass; one environment skip.
- saved-policy-device.txt: 27 exact RTX5070 tests pass; twenty-four oracle
  saved/recompute combinations cover three shapes, plain/full bias, reversed
  function argument order and default/integer-neutral policy. Three additional
  device cases check checkpoint buffer/lifetime contracts.
- saved-policy-regressions.txt: 22 existing backward route/benchmark and
  regenerated verifier-coverage render/in-sync checks pass.
- saved-verifier-gates.txt: 57 coverage and dialect-registration checks pass.

The earlier generated gate reported two absent verifiers. Both actual native
implementations were namespace-qualified; the scanner now recognizes qualified
method definitions and has a parser regression plus three positive native
contract sentinels. The owning verifier-coverage generator refreshed both CSV
and Markdown. This repairs evidence classification; the compiler verifiers
were retained.

Current seven-sample packet: forward 26.782 us device / 1.042 ms host-array complete launch; backward 80.076 us device / 1.350 ms host-array complete launch. The domains remain separate; no promotion.
Packet: saved-policy-rtx5070.json, live GPU UUID/SM/driver and current
compiler/runtime hashes. Independent O/LSE/gradient oracles pass before and
after timing. At this earlier checkpoint recompute used a legacy Python Tile constructor;
see the subsequent native migration below.


Nullable optional attention modifiers now encode Python None as absence in typed
Graph IR. Unknown attributes are retained for native diagnostics. Final shared
Graph/registry/checkpoint suite: 164 passed, one environment skip. Final Graph,
audit, diagnostic and pass metadata gates: 349 passed, one skip
(saved-policy-final-gates.txt). The complete generated-doc refresh succeeds
(saved-policy-generated.txt); verifier render/in-sync assertions are included
in the focused 22-check regression run.

## Native recompute backward Graph migration

The canonical recompute packager now lowers the original Graph through native
Graph-to-Schedule and Schedule-to-Tile. Its immutable Python record decodes
the native policy certificate and requires exact replay of both passes.
The compiler owns role projection, shape/storage admission and policy checks.
The Python Tile emitter remains an explicit control, with tests forbidding
its use by canonical packaging.

- recompute-build.txt: matching production compiler build succeeds.
- recompute-unit.txt: 407 native contract/checkpoint/diagnostic/pass metadata checks pass.
- recompute-device.txt: 54 exact RTX5070 numerical/lifetime checks pass,
  including FP16/BF16/FP32, full bias, permuted arguments, default/active
  window-softcap-dropout policies, changed input values and existing saved-LSE
  regressions.
- The first device run exposed a storage-less generated symbol. Existing
  transfer ABI code derived a four-byte width for two-byte gradient storage,
  corrupting host memory. Native symbols now carry f16/bf16/f32 explicitly;
  the unit ancestry cases verify that field and the rebuilt device suite passes.

General composed producers, dynamic shapes, broadcast recompute bias, higher
derivatives and sibling architecture physical execution remain open.

Final recompute policy proof includes mixed legal kept/dropped keys, negative
seeds and both signed-int64 extrema. The LCG seed-offset addition now uses the
low 32 bits explicitly in NVIDIA forward/backward lowering, avoiding C++
signed overflow. recompute-policy-device.txt records the 24 mixed-mask checks;
the final 54-case device suite reruns those and three extreme-seed cases with
the final matching compiler and launcher.

recompute-rtx5070.json records seven samples, three warmups, 100 resident
launches per event window and three host-array launches per wall sample:
- Forward: 26.663 us device / 1.093 ms host-array launch.
- Saved backward: 80.056 us device / 1.334 ms host-array launch.
- Recompute backward: 1330.768 us device / 2.514 ms host-array launch.

These domains are distinct. Recompute remains materially slower than saved
backward on this shape; this migration proves compiler ancestry and execution,
with no kernel speedup or default-route promotion claim. The packet includes
live GPU UUID/SM/driver, three compiler/runtime binary hashes and native
recompute Graph/Schedule/Tile digests.

Shared audit/op/dtype gates: 41 passed (recompute-shared-gates.txt).
The complete owning generated-doc refresh succeeds (recompute-generated.txt).

## Public saved-LSE JIT integration (2026-10-06)

Sync NVIDIA-PUBLIC-SAVED-LSE-2026-10-06; owner NVIDIA-LSE-1 /
FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.

`flash_attn(..., lse_checkpoint="saved")` returns `(O, LSE)`. The frontend
preserves the primary output dtype and declares row LSE as f32. Production
attention tracing uses canonical shape rules without executing host attention;
explicit differential certification enables concrete reference evaluation.
The compiler driver imports the original caller Graph into native checkpoint
lowering, preserving SSA argument roles and physical broadcast-bias extents.
Graph -> Schedule -> Tile -> NVIDIA Target -> native PTX owns arithmetic.

- public-saved-lse-regression.txt: 685 passed, 7 skipped on Super-Bear WSL,
  including saved/recompute checkpoint device consumers, public attention,
  reference AD laws and registry/dtype gates.
- public-saved-lse-mypy.txt and public-saved-lse-lint.txt: zero mypy errors
  and clean focused Ruff checks.
- public-saved-lse-rtx5070.json: 18 RTX 5070 / SM120 rows, full/causal,
  grouped heads, three shapes and three operand/bias profiles. Independent
  FP64 O/LSE checks precede timing and cover public JIT, serialized replay
  and resident execution. Five public wall and five resident CUDA-event
  windows retain distinct timing scopes and current source/tool hashes.
- Recorded by benchmarks/nvidia/record_public_saved_lse_attention.py;
  public-saved-lse-benchmark.txt records completion of every row.

Reference JVP/VJP includes the LSE cotangent and grouped-head reduction.
Native differentiation of an explicit LSE cotangent remains open. General
composed/dynamic attention, tuple aliases in textual frontend recovery,
full-suite batching/transpose closure and publication remain open. Sibling
Apple/ROCm/x86 saved-LSE physical consumers require architecture-specific
follow-up; this packet proves no execution on those devices.

## Saved-LSE tuple SSA bindings (2026-10-06)

Sync NVIDIA-SAVED-TUPLE-SSA-2026-10-06; owner NVIDIA-LSE-1 /
FRONTEND-IR-MEDIUM-1 / E2E-REAL-6. AST recovery retains every result through
local tuple assignment, copying, destructuring and constant-index selection.
Scalar rebinding clears that local tuple binding without changing earlier
aliases. Tensor-to-tensor name aliases now reference their actual SSA value
instead of declaring a local name with no defining operation.

- saved-tuple-alias-tests.txt: 122 passed, 1 skipped in host WSL. Includes
  six new RTX 5070 cases for full/causal attention, reversed arguments,
  copied tuples surviving local rebinding and broadcast bias. Cold/warm
  output and LSE match the independent FP64 oracle.
- saved-tuple-alias-mypy.txt / saved-tuple-alias-lint.txt: mypy zero errors,
  focused Ruff clean.
- public-saved-lse-tuple-aliases-rtx5070.json: 18 current-source exact-device
  rows, independent O/LSE checks, portable replay, five public wall windows
  and five resident event windows. Source hashes verified after recording.
  Recorded with record_public_saved_lse_attention.py --tuple-aliases;
  saved-tuple-alias-benchmark.txt retains per-row completion.

The direct-call public-saved-lse-rtx5070.json packet retains the earlier
source snapshot; its hashes are historical after tuple recovery changed.
The new alias packet records the current source and unchanged native tools.
No selector promotion. General nested/dynamic tuple structures, composed
attention and native explicit-LSE cotangent differentiation remain open.
All four backend plans are assessed; no sibling device proof is inferred.
