# NVIDIA public softmax alias native integration

Owner E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Sync NVIDIA-PUBLIC-SOFTMAX-ALIAS-2026-10-08.

Standalone ordinary JIT softmax/softmax_safe now uses typed frontend Graph, native Schedule/Tile, NVIDIA Target/PTX and the dtype-specific checked runtime ABI. Shape and output storage come from compiler descriptors. Warm calls and serialized replay forbid compiler subprocesses and eager execution. The alias remains last-axis only; no dynamic/composed/AD or non-last-axis support is claimed.

Owning RTX 5070: fourteen native package parity cases and eighteen ordinary public/portable/changed-input cases pass. FP32, FP16 and BF16 cover singleton, ragged K17 and rank-three K257 rows, including a finite constant row at 1000. Separate registry/diagnostic/pass gates pass 366 tests.

The resident native seam now admits the existing FP32 softmax ABI alongside FP16/BF16. No new runtime ABI or Python arithmetic backend was introduced.

The benchmark completes eighteen correctness-gated profiles. Each safe/ordinary pair has identical native image bytes; descriptor ancestry remains specific to each frontend. Resident CUDA-event kernel-loop samples and warm public host-wall samples are separate. No speedup or selector promotion is claimed.

Recorder: benchmarks/nvidia/benchmark_public_softmax_alias.py.
Raw exact-device, source/compiler/runtime identities: packet.json.
Receipts: scheduled-device.log, public-portable-device.log, registry-gates.log.
The final packet records the timed recorder, selected runtime, compiler and source identities directly. initial_packet.json preserves the earlier annotated packet; it is not substituted for the final run.

Final matching-runtime lane: 32 device cases pass. Resident CUDA-event medians span 0.008306–0.028701 ms; warm public host-wall medians span 1.132061–1.434576 ms. These are separate scopes, not a speedup ratio; public dispatch overhead remains an optimization obligation.

Shared native-chain/movement/launch/manifest/conformance gate: 224 passed, ten skipped. Owning generated-document regeneration completes. Aggregate unit closure is still unproven.
