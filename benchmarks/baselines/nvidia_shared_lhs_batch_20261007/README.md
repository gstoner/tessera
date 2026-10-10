# Native NVFP4 shared-LHS batch integration — 2026-10-07

Owners: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Synchronization key: NVIDIA-NVFP4-SHARED-LHS-2026-10-07.
Parent packet key: NVIDIA-NVFP4-NATIVE-ORIENTATION-2026-10-06.

The named SM120 profile now admits `batching="shared_lhs"`: A[M,K] and
A scales[M,ceil(K/16)] are shared; B[B,K,N] and its scales[B,ceil(K/16),N]
vary by batch; output is [B,M,N]. Both matrix/scale storage orientations
are verified. Packed odd K and ragged rows/columns keep the existing exact
per-K16 policy.

Public `vmap` axes `(None,0,None,0)` project semantic batch intent into
a separate JIT owner. They preserve the scalar owner, symbolic constraints
and checked physical storage. Native Graph -> Schedule -> Tile -> NVIDIA
Target/LLVM/PTX handles one GPU launch. A/scales keep their base pointers;
native batch indexing offsets B/scales/output. There is no production Python
replication, numerical unpacking or per-member launch loop.

## Exact-device proof

Super-Bear: RTX 5070, compute capability 12.0,
GPU-cba12639-821a-7a10-4cd3-f918f9c0a545; CUDA 13.3.73;
matching core/NVIDIA LLVM/MLIR 23.1.1 builds.

- `native-tests.txt`: 89 passed, including 41 GPU numerical cases across
  four modes/orientations and public vmap; Graph/Schedule policy sealing,
  warm compiler-free reuse, scalar-owner and constraint checks, and pre-launch
  batch-scalar reinterpretation refusal.
- `registry-tests.txt`: 406 passed for operation/dtype attributes,
  diagnostics, pass metadata, backend manifests and numerical fixture wiring.
- `orientation.json`: 32 rows, eight shared-LHS rows, five samples per
  timing domain. Chosen operands have zero measured maximum absolute error
  against independent decoded FP64 products before timing, after every public
  sample and at final portable launch.
- `packet-verification.txt`: current source and compiler/runtime hashes
  match the packet.

Shared-LHS resident native CUDA-event medians span 8.841–13.575 us. Warm
public-call wall medians span 1.461–2.016 ms. Event windows measure resident
execution plus dispatch (100 repetitions, 20 warmups); public wall includes
validation/allocation/copies/synchronized launch. These characterize named
static shapes and strategies, not isolated kernel-only time or a speedup
against a serial/reference arm.

The first native replay exposed a missing public JIT ABI-policy gate and
incorrect new test fixtures; `native-tests-initial.txt` preserves that
failed attempt. The final test retains native refusal: a semantically valid
policy/shape substitution cannot reuse the original sealed Schedule.

## Reproduction

On the owning WSL checkout, source `.build-sm120-w1-1/validation-env.sh`
and use the existing `.venv`:

```sh
python -m pytest -q tests/unit/test_native_nvfp4_vmap.py \
  tests/unit/test_nvidia_nvfp4_runtime_contract.py \
  tests/unit/test_nvfp4_native_transpose_contract.py \
  tests/device/nvidia/test_nvfp4_transpose_jit.py
python benchmarks/nvidia/record_nvfp4_transpose.py --samples 5 \
  --output benchmarks/baselines/nvidia_shared_lhs_batch_20261007/orientation.json
```

## Scope still open

Dynamic and nested batches, arbitrary map axes/composed producers, general
scaled semantics and linear-transpose AD remain open. The generic
`scaled_matmul` coverage axes stay partial/planned; the failing global
batching/transpose closure gate is not weakened or relabeled. No sibling
Apple/x86/ROCm physical proof is inferred. The five-slice aggregate remains
unpublished and its full unit lane is not green.

## Delivery gates

Final-source native tests pass 89 cases. Packet source/tool hashes have been
verified; additional ODS/manifest/coverage/unit-test hashes are recorded in
additional-source-snapshot.json. Audit lifecycle and baseline-citation checks
plus compiler-plan regressions pass 22 cases in docs-tests-final.txt. The generic closure gate remains two
failures and four passes in generic-closure.txt.

The integrated-plan checker passes in plan-check.txt after canonical owner
links, required fields and Latest navigation were repaired while preserving
historical evidence and anchors. The final generated-doc check passes all 32
documents. Current-source gfx1151 math/movement revalidation is recorded in
../gfx1151_current_source_revalidation_20261007/README.md. Publication,
broader owning-family revalidation and the full-unit gate remain open.
