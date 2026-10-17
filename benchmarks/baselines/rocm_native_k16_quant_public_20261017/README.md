# Public source-derived K16 quantizer package

Owner: ROCM-NVFP4-INGEST-1 / ROCM-FP8-BLOCKSCALE-1 / FRONTEND-IR-MEDIUM-1.
Sync: ROCM-K16-QUANT-PUBLIC-20261017.

## Integrated route and envelope

Ordinary authored Python emits thirteen explicit typed Graph operations:
reshape, FP32 widen, abs, max, two upstream FP32 splats, scale divide/floor,
broadcast, normalize, real E4M3 cast and two output reshapes. The canonical
`block_quant` family enters only at native Tile after verified Graph/Schedule;
it is gfx1201-owned and excluded from generic gfx1151 family claims.
Schedule/Tile/ROCm Target/ROCDL/LLVM produce a native HSACO and checked
BF16-input/E4M3-code/FP32-scale two-result frame. Source input/output names,
byte spans, lifetimes, geometry and source seals remain checked. Python
frontend capture and ABI marshalling perform no input numerical evaluation
or backend lowering. Existing fake-quant reference semantics are unchanged.

The literal frontend contract is finite, explicitly read-only, bit-identical
FP32 array splats used as positional tensor operands. AST and trace share
an immutable shape/bit snapshot, including cache identity. Mutable, mixed-bit
and nonfinite arrays are not silently frozen. Canonical reduction inference
now preserves `axis` and `keepdims`; no eager numerical evaluation is needed.
Upstream `arith.constant` remains visible in whole-program gates with explicit
coverage/manifest rows; it is not a new standalone public Tessera operation.

Only the named compact static BF16 [M,K] profile, K divisible by 16, with
exact amax/448, FP32 minimum-normal scale floor and RNE E4M3 storage is admitted.
Manifest component rows do not promote standalone view/cast/broadcast execution.
Generic batching, differentiation, sibling schedules and full MLP stay open.

## Exact-device proof

[public-device-final-tests.txt](public-device-final-tests.txt): four owning RX9070XT/gfx1201
cases pass: public JIT at three shapes, changed-input compiler-disabled replay,
retained outputs and fresh-process replay with compiler lookup disabled.
Actual public calls additionally trap eager NumPy max/maximum evaluation.
[public-quant-3-32.json](public-quant-3-32.json),
[public-quant-19-12288.json](public-quant-19-12288.json) and
[public-quant-256-12288.json](public-quant-256-12288.json) are the actual portable
native artifacts. They include images and checked source/ABI lineage.

Canonical HSACO SHA256 is
`18ee2db7e8c0c192080ac7931dc20b2942fad175727b091a618564bdb2449dc5`
for all three source envelopes. It differs from the earlier diagnostic image,
so its numerical evidence was rerun rather than inherited:
[canonical-image-exhaustive.json](canonical-image-exhaustive.json) records zero
code and scale mismatches for every shape, all 65,536 BF16 patterns in uniform
K16 groups, heterogeneous groups, 252 explicit FP8 midpoint/BF16-neighbor
groups, zero/subnormal/finite/NaN/Inf cases, changed roots and held outputs.
Finite codes/scales compare bitwise; NaNs compare by classification.
[public-exhaustive-check.txt](public-exhaustive-check.txt) records the run.

[project_public_evidence.py](project_public_evidence.py) projects the actual
portable artifacts for the existing frozen diagnostic checker at
../rocm_native_k16_quant_20261017/check_quantizer.py. It uses no new arithmetic
recipe. [benchmark_public_package.py](benchmark_public_package.py) is the
frozen public/prepared/device timing recorder. Run both from the source root;
their explicit scratch paths and provider environment are recorded.

## Timings and validation

[public-package-device.json](public-package-device.json) pins live GPU UUID,
compiler/provider/recorder hashes, full program gate admission and timing scope.
Correctness precedes every retained timing sample; captured windows exceed
10 ms. No speedup or default-policy performance promotion is claimed.

| Input shape | Captured member (µs) | Retained owner invoke + reads (ms) | Warm ordinary JIT (ms) |
|---|---:|---:|---:|
| 3 × 32 | 4.517 | 0.342 | 1.775 |
| 19 × 12288 | 4.928 | 0.396 | 1.897 |
| 256 × 12288 | 30.338 | 0.803 | 2.758 |

Device time includes graph dispatch but excludes capture/instantiation, CPU
enqueue and copies. Retained-owner host time excludes compilation/preparation
and includes two readbacks. Ordinary JIT includes frontend replay, binding,
native prepare, execution and readback. These columns are separate scopes.

[package-program-gates.txt](package-program-gates.txt): 73 canonical package,
frontend and operation/shape registry cases pass. [public-final-gates.txt](public-final-gates.txt):
514 adjacent native/dtype/diagnostic/pass cases pass. [package-shared-gates.txt](package-shared-gates.txt):
517 shared cases pass with one skip before the final upstream-coverage entry.
[public-final-registry-gates.txt](public-final-registry-gates.txt) records
**725 passing** final primitive/operator/shape/dtype/pass/diagnostic/audit/naming
cases after upstream constant coverage and exact frame-field checks.
[public-mypy.txt](public-mypy.txt) records the zero-error ratchet.
Counts overlap and do not establish the full CI suite.
[package-nvidia-tests.txt](package-nvidia-tests.txt): 12 fresh exact RTX5070
producer/ragged/ReLU controls pass after the shared frontend change.

Required next: combine the quantizer with native NVFP4 gate/up and down
projection under retained private C++ ownership, eliminate intermediate host
readback, and run full checkpoint/MLP/model-quality plus end-to-end timing.
The five-slice goal is not complete.
