# Portable native attention JVP program — 2026-10-06

Owner **AD-RESIDUAL-EVAL-1**; siblings FRONTEND-IR-MEDIUM-1 / W1.1.
Synchronization key **NVIDIA-JVP-PORTABLE-2026-10-06**. Full five-slice closure remains open.

## Compiler-owned program binding

The existing native forward/backward checkpoint pair and native saved-LSE tangent package now form a canonical JSON container, `tessera.native_attention_jvp_program.v1`, with an externally pinned SHA256 identity. It carries original Tile/Target/backend IR, images/descriptors, tangent image/native sizing library, physical activity, frontend role mapping and original parameter names. Native Graph/AD/Schedule/Tile/LLVM image construction is unchanged; serialization introduces no GPU body constructor or physical ABI.

Before capture allocates or loads a CUDA image, validation checks native checkpoint image/descriptor integrity, exact frontend permutation, physical activity, target, shared checkpoint generation, nine-pointer/index tangent ABI and complete compiler-produced tensor manifest, including grid/block and scratch bounds. Reissued container hashes do not excuse incompatible native mappings or tangent roles.

Capture supports positional, keyword and mixed calls through the pinned frontend signature; parameter labels are kept distinct from physical Q/K/V roles. Duplicate, missing and unknown arguments are rejected before device capture. Historical pre-keyword artifacts are preserved under `history-before-keyword-binding/`.

API: `program.to_json()`, `program.program_digest`, and `NativeAttentionJVPProgram.from_json(text, expected_digest=pin)`. Deserialization and execution do not rebuild Graph IR or select a backend recipe. This portable checked product is a prerequisite for routing attention through the existing public native-JVP family boundary; that dispatch integration is not yet implemented.

## Exact-device proof

[packet.json](packet.json) records 72 owning RTX 5070 / SM120 cases across six argument permutations, short noncausal/long causal profiles and six semantic wrt orders. Every program is serialized and restored before numerical execution. Maximum tangent absolute error against independent FP64 finite differences is **1.42741e-8**, under the original `atol=rtol=3e-5`. Actual caller mutation, repeated scaled directions, retained prior outputs and closed-frame refusal pass.

[artifacts](artifacts/) retains canonical program JSON, native arena and image for every case. The packet queries the NVIDIA GeForce RTX 5070, UUID GPU-cba12639-821a-7a10-4cd3-f918f9c0a545, driver 610.88, and records source/compiler fingerprints. Event dispatch and allocating checked-JVP wall samples are characterization only; no speedup, throughput or isolated kernel claim is made.

Three independent fresh processes restore pinned programs and execute with all subprocess creation forbidden and compiler paths set unavailable:

- [V-only reordered long causal](vqk_v_129_1-replay.json).
- [K/Q reordered short noncausal](kvq_k_q_5_0-replay.json).
- [Full Q/K/V reordered long causal](vkq_q_k_v_129_1-replay.json).

The fresh processes prove primal/tangent oracles, scaled directions and retained results. Prebuilt CUDA runtime libraries remain available; no claim of runtime-library-independent execution is made.

## Validation and remaining scope

- **339 current keyword/portable/native-JVP/diagnostic/pass tests passed**, [keyword-contracts.txt](keyword-contracts.txt).
- **370 prior focused tests passed**, [contracts.txt](contracts.txt), covering portable digest/role checks, refusal before CUDA allocation, frontend/native AD, diagnostics and pass metadata.
- **77 shared tests passed** in the matching gfx1201 compiler checkout, [.validation-jvp-portable/shared.txt](.validation-jvp-portable/shared.txt). This is host contract evidence, not HIP tangent execution.
- Ruff, audit/plan and generated-document gates are retained here.
- No new op/dtype/target/pass/stable diagnostic or physical ABI is introduced. Apple/x86 shared contract outcomes are host-tested; exact-device native tangent integration remains follow-up required.

Remaining: connect this program to the canonical public native-JVP plugin/runtime route; remove the unused reverse image from forward-only compilation/loading; move repeated package validation and launch preparation into native owners where measured. General composed/dynamic/bias/dropout/higher AD, asynchronous ownership, sibling physical consumers and independent FP8/MXFP8/MXFP4 gates remain open.

Follow-on: the named public native_jvp family/runtime route is now proved in [NVIDIA-PUBLIC-ATTENTION-JVP-2026-10-06](../nvidia_public_attention_jvp_20261006/README.md). Remaining statements above describe this earlier portable-program increment; generic composition and forward-only retirement remain open.
