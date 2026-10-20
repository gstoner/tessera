# Native SM120 tensor frame owner

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: NATIVE-TENSOR-PROGRAM-PIPELINE-20261019.

This is a retained C++ runtime milestone. Public JIT/portable admission and
whole-program Target IR integration were open at this checkpoint; the later
native_tensor_target_controller_20261019 packet records the Target increment.
Public JIT/portable integration remains open; no performance promotion.

## Executed evidence

- native-owner-current.xml / native-owner-current.txt: **45 passed**, zero
  skips/failures on RTX5070 / SM12.0. Covers FP16/BF16 ReLU, explicit-policy
  RMSNorm and two-sided ReLU/softmax source programs; changed host inputs,
  completed independent results and fixed device allocation; delayed resident
  writes on different/shared/legacy/per-thread streams; argument failures,
  output aliases, invalid geometry/entries, unknown fields, retired allocation
  IDs, insufficient actual spans, foreign contexts, and process/thread guards.
  The fork test deliberately exercises the pre-CUDA PID gate and records the
  platform's multithreaded-fork deprecation warning.
- native-owner-current-legacy.xml / native-owner-current-legacy.txt:
  **38 passed**, zero skips/failures against the same final provider.
  Existing native/public/resident/portable two-operand routes retain parity.
- native-owner-current-ubsan.txt: the raw extreme-shape rejection passes with
  undefined-behavior sanitizer and halt-on-error enabled. Bounds checks precede
  int128 byte multiplication. This is one of the 45 cases, not another unique
  numerical envelope.
- native-owner-host-gates.txt: **336 passed** for runtime-ABI, audit,
  diagnostic and pass-metadata gates. native-owner-mypy.txt records zero errors
  in the new binding; Ruff also passes.
- native-owner-source-identity.json: exact GPU/UUID/driver, compiler versions,
  production/sanitizer provider hashes and executed repository source hashes.
  native-build-current.txt and native-build-current-ubsan.txt retain compiler
  output (empty on successful warning-free builds).

The emitted kernels and native frames are the eight content-addressed images
shared from ../native_tensor_program_pipeline_20261019/images/. No shader
copies are added here. Every measured case retains its original source,
program and member-image hashes.

## Measurements

Reproducer: benchmarks/nvidia/record_native_tensor_frame_owner.py.
native-owner-current-timings.json and native-owner-current-timings.txt retain
nine shuffled paired rounds of 32 repetitions over six M17/N11/K32 cases.
Correctness is asserted before and after timing; each owner retains one device
arena and fixed capacity. Pinned staging is separately retained.

Final medians span:
- completed adapter calls: **0.184116–0.211543 ms**;
- identical checked C++ calls with prebound views/results: **0.146096–0.167875 ms**;
- captured full-program device events: **0.003106–0.004895 ms**.

The adapter difference includes metadata validation and host result allocation;
raw calls still include snapshots, H2D, kernels, D2H and completion. Event
windows exclude capture/prime/retirement and transfers/readback. Grouped stage
windows are independent and must not be summed. Cold owner binding/JIT is
recorded separately and is not frontend/compiler cost.

These small ragged cases characterize latency. They establish no general
model/throughput advantage or default-route promotion. FP8/MXFP8/MXFP4 physical
evaluation and the broader ROCm performance program remain independent.

## Remaining source-to-execution gate

whole-program-target-probe.json / whole-program-target.mlir.gz, with
whole-program-target.exit / whole-program-target.stderr, preserve an owning
compiler probe: at that checkpoint, lowering the complete program succeeded but left
two tile.program_invoke operations in the host controller while numerical
members lower to NVVM. This is not full controller Target IR closure.

The later Target-controller packet consumes the typed host program. Remaining
work must validate original source/program/member ancestry and integrate
compiler-free warm public replay and portable reconstruction. Do not replace
that work with a Python family dispatcher or regenerated numerical kernels.
