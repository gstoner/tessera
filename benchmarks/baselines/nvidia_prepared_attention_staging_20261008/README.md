# Native prepared attention stream and pinned staging

Owner E2E-REAL-6 / AD-RESIDUAL-EVAL-1.
Sync PREPARED-ATTENTION-STAGING-2026-10-08.

Prepared SM120 JVP and saved-LSE backward retain a pinned host staging arena.
Uploads, forward, derivatives and downloads share the owner's nonblocking stream.
Normal return and failure paths drain owned work before staging reuse; caller
outputs are copied only after successful completion. The public ABI remains
synchronous and the native images, arithmetic and package schemas are unchanged.
Owner retirement joins its own stream. CUDA allocation/free may still impose
implicit synchronization; this does not establish asynchronous execution.

64 owning RTX 5070 checks pass, including biased numerics, fork/context/lifetime
guards and next-generation recovery after invalid input extents.
Both independent-stream isolation cases pass: own numerical outputs are complete
while unrelated nonblocking GPU work remains pending. Preserved control JVP
fails that assertion as expected; control backward already passes the isolation
check. This is evidence for removing JVP's context-wide wait, not a claim that the
old backward joined all GPU streams.

identity.json records runtime/source hashes; control.cpp preserves the old source.
No performance gain is claimed. Matched A/B timings remain pending until the
current full-unit lane completes, to avoid concurrent validation workload.


## Matched A/B recorder ready

benchmarks/nvidia/benchmark_prepared_attention_staging_ab.py selects seven
saved JVP/backward artifacts, alternates control/candidate library order over
at least three fresh-process windows, validates loaded library and artifact
identity, and retains per-trial prepared host-wall and separate device events.
The recorder's concurrent-validation refusal was verified while the full WSL
unit process remained live. No A/B timings have been started in this stage.

## Matched prepared-call staging A/B

Seven correctness-gated cases completed in alternating fresh processes, with three paired windows per case and eleven repetitions per window. Native artifacts match within each comparison. Ratios below are control/candidate prepared host-wall time; device-event samples are retained separately in ab_packet.json. These results measure the checked staging adapter, not an isolated kernel gain or general attention closure.

| Derivative | Case | Median control/candidate wall ratio |
| --- | --- | --- |
| jvp | qkv_q_5_0 | 1.040043 |
| jvp | vqk_v_129_1 | 1.093934 |
| jvp | kvq_k_q_5_0 | 1.107587 |
| jvp | vkq_q_k_v_129_1 | 1.107945 |
| vjp | qkv_q_5_0 | 1.299721 |
| vjp | vkq_q_k_v_129_1 | 1.377027 |
| vjp | biasvqk_bias_v_k_q_5_1_1x4x1x1 | 1.608286 |
