# Checked native LSE cotangent package and replay

Owner NVIDIA-LSE-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync NVIDIA-LSE-COTANGENT-PACKAGE-2026-10-07. Publication pending.

Typed Graph dO,Q,K,V,O,[bias],LSE,dLSE consumers now package through native
Schedule/Tile/NVIDIA Target/PTX with a distinct checked f32 seed ABI. The
canonical compiler driver validates semantic input shapes; the package
projection carries the row seed, exact shape guards, complete Q/K/V outputs,
Schedule identity and all compiler ancestry. No Python kernel constructor
or production launch loop is added. The image and descriptor survive JSON
round-trip; replay runs with compiler paths deliberately disabled.

The common runtime validates exact scalar bounds, buffer roles, seed dtype,
shape and contiguous strides before loading CUDA. Output overlap with any
other physical role is rejected. Resident launches order every buffer's
producer stream using the existing event/wait contract. Host, resident,
host-buffer event and resident event paths share the seed envelope checks.
A foreign CUDA array interface remains a caller allocation declaration;
independent discovery of arbitrary foreign allocation capacity is not claimed.

RTX 5070 / SM120: 36 package cases pass across three rectangular/batched GQA
shapes, causal/full masks, plain/score-bias and output-only/LSE-only/mixed
signed seeds. Four native replay/timing paths compare Q/K/V to independent
FP64 derivatives. Eight invalid invocation/descriptor cases are rejected
before CUDA loading. contract-registry-tests.txt records 366 passed (44
package cases plus 322 registry tests); earlier combined checkpoint gate has
142 passed. Counts overlap. Stream-contract gate: 11 passed. Ruff and the
new physical contract's Mypy check are clean.

Recorder: benchmarks/nvidia/record_lse_cotangent_package.py.
rtx5070.json records 36 rows, 180 CUDA-event and 180 host end-to-end windows,
with poisoned outputs checked after each window. C++ event loops exclude
upload/readback; host timing includes checked package validation and bridge
copies. These domains are recorded separately. The task's Graphify update
was paused during timing and resumed afterward. Sources and matching core,
NVIDIA compiler and launch runtime hashes were rechecked. Small correctness
windows are characterization evidence, not route or performance promotion.

Remaining: compact activity and bias-gradient package envelopes, automatic
multi-result AD row-seed plumbing, private residual ownership, dynamic and
general composed attention. This static explicit consumer does not close the
full five-slice objective. Shared runtime ABI totality is updated; Apple/x86
and ROCm receive no new physical admission or device proof.
