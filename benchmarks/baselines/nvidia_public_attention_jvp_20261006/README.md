# Public native attention JVP — 2026-10-06

Owner **AD-RESIDUAL-EVAL-1**; siblings FRONTEND-IR-MEDIUM-1 / W1.1.
Synchronization key **NVIDIA-PUBLIC-ATTENTION-JVP-2026-10-06**. Full five-slice closure remains open.

## Route

The ordinary explicit differentiation entry point, JitFn.native_jvp, now dispatches flash_attn through the canonical native-JVP family registry. Its planner consumes the actual specialized tracer Graph and native paired forward AD. The existing native Graph/AD/Schedule/Tile/arena/NVVM/LLVM images are pinned in the portable saved-LSE program; no new GPU body or Python differentiation rule is introduced.

The parent content-addressed product launches a registered SM120 runtime consumer through the execution matrix. CUDA session ownership covers uploads; private capture owns the saved O/LSE generation; both outputs are downloaded before owner release. The runtime validates pinned image/ABI/activity, frontend role mapping, host fp32 shapes and tangent arity before CUDA allocation.

The family declaration names only its implemented NVIDIA consumer. Registration no longer requires fictional x86/ROCm consumers for every family. Existing families preserve their declared targets. Deterministic flash attention (dropout_p=0) is admitted to the existing frontend differential certificate; stochastic cases remain outside this physical envelope.

The public sweep found an independent frontend bug: the AST Graph cache omitted captured constants. Identical source closures with causal=False/True collided. Captured literal environments now participate in the cache identity, and a regression retains differential parity instead of skipping that gate.

## Exact-device evidence

[packet.json](packet.json): **72 public API cases** on the owning RTX 5070 / SM120, across six frontend permutations, short noncausal/long causal profiles and six requested tangent orders. Independent FP64 primal and centered finite-difference tangent oracles pass; worst tangent error is **1.10436e-8**, under atol=rtol=3e-5. Repeated scaled directions preserve retained prior host outputs.

Warm native_jvp calls execute with every subprocess forbidden. The median across case warm wall medians is **14.6047 ms**. These are synchronous checked host end-to-end samples, including frontend trace/binding, validation, uploads, native module execution and downloads. They are characterization, not isolated kernel timing or a speedup claim. Prior [native product evidence](../nvidia_jvp_portable_20261006/README.md) retains separate device-event characterization; it is not relabeled as the public-route measurement.

[artifacts/](artifacts/) retains each parent product and its child images. Three new processes restore externally pinned parents and launch through the common runtime:

- [Reordered V-only causal](vqk_v_129_1-replay.json).
- [Reordered K/Q noncausal](kvq_k_q_5_0-replay.json).
- [Reordered full Q/K/V causal](vkq_q_k_v_129_1-replay.json).

Hardware/runtime discovery completes before forbidding all subprocesses for package restoration and execution. No source Graph reconstruction, native compilation or frontend reexecution occurs during those runtime replays. Prebuilt CUDA runtime libraries remain required.

## Validation and limits

- [contracts.txt](contracts.txt): **444 tests passed**, covering frontend cache/LRU, family declarations, image/ABI/host guards, execution matrix, operator/dtype contracts, diagnostics and pass metadata.
- [device-tests.txt](device-tests.txt): **4 owning-device tests passed**.
- [fingerprints.txt](fingerprints.txt): all nine recorded current source fingerprints match.
- Applicable audit, plan, generated-document and whitespace gates accompany this increment.
- Apple/x86/ROCm share the cache and explicit-target declaration fixes; SM120 images are not executable sibling-backend evidence. Their native attention tangent consumers remain follow-up required.

This closes ordinary native_jvp dispatch for the named static, distinct, direct fp32 Q/K/V attention envelope. General composition, dynamic shapes, aliases, bias/dropout/higher AD, generic VJP dispatch, sibling physical consumers, retirement of unused reverse compilation/loading, and native prepared validation/launch ownership remain open. Independent FP8/MXFP8/MXFP4 gates and other five-slice obligations remain open.

Follow-on: [native prepared runtime ownership](../nvidia_prepared_attention_jvp_20261006/README.md). This packet retains its original source fingerprints and timings.
