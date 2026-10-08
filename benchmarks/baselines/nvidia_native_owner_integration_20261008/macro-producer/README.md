# Static macro-CTA producer integration candidate
Owner W1.1 / E2E-REAL-6 / FRONTEND-IR-MEDIUM-1.
Sync NVIDIA-MACRO-PRODUCER-2026-10-08.

The current compiler selects macro_cta_cp_async_2stage_shared_ab_bf16 for long normalization-to-matmul consumers. The unchanged prepared Python admission and native owner only support 16x8 typed global consumers, so four K4096/K8192 serial/cooperative BF16 cases reject before execution.

This isolated candidate admits the actual static f16/bf16 column-major macro Schedule contract, preserves completion ordering, buffer/workspace guards and ownership, and launches the compiler macro entry with 32x32 tiles and 128 threads. Dynamic macro admission remains rejected. Dynamic multi-producer compiler guards are untouched.

All ten existing norm-accuracy cases pass on RTX 5070, including the four originally failing long-row cases with their original independent numerical tolerances. Unchanged control fails those four at artifact validation. Primary checkout/runtime remain frozen for the active full-suite sweep. Typed/dynamic-single-producer/portable owner regression is separately running; performance timings and authoritative integration remain pending.

matmul_prepared.cpp SHA256 d48fa51415b6a2867a0769b1397d0c1bb57b46b906c4460c8497d5ccbcbebc0b

prepared_nvidia_lhs.py SHA256 9197c74ebdc0a9b424627b48b974988e21e55a699975495223461d4166d2c8f5

libtessera_nvidia_ptx_launch.so SHA256 2ec57d992e45e3876cfbc1911e69fe4beb958cbc4d1bb2107ce7c202c0d337ef

## Owning regression and repair

Initial owner regression: 171 pass, one failure. The unchanged control reproduces the failure: duplicate attachment is refused but its historical already-attached message drifted. The candidate restores that specific message while preserving the distinct append-before-attach guard. Final targeted repair plus original norm-accuracy lane: 11 pass. Initial broader regression and final repair logs are retained. No gates were weakened.

Final candidate runtime SHA256 0a166da3238a64ab991b8bc9979873be3b555aa0759f49070708ff1f799e6163
Final native source SHA256 804872425d2f7a262c253c03e758a534f140376303954d4f02cea13a8acfe306

The previous hashes identify the initial tested revision. Integration into authoritative source, compiler-plan updates, Graphify refresh and idle-device timing remain pending. Primary runtime still has SHA e2fd477fbdfc6ae2e91d6e651daa3e55678e21128a8e18ca615dc509753fd290.

## Expanded producer checks

Twelve K4096 cases cover RMSNorm, LayerNorm and softmax in fp16/bf16. Six plain fp32-output cases select macro-CTA; six bias/ReLU/residual fp16-output cases retain the compiler-selected typed route. All pass original independent numerical tolerances, changed-input replay and bit-identical retained-output checks. Maximum absolute error is 0.03125 (including fp16 fused output rounding); no kernel timing claim. The initial fixture incorrectly assumed fused fp16 selects macro; actual compiler provenance corrected that assumption, without changing compiler selection or numerical gates. Evidence: macro_edges.json, check_macro_edges.py.
