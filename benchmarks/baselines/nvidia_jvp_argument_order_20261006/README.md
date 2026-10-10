# Native attention JVP frontend argument order — 2026-10-06

Owner **AD-RESIDUAL-EVAL-1**; siblings FRONTEND-IR-MEDIUM-1 / W1.1.
Synchronization key **NVIDIA-JVP-ARGUMENT-ORDER-2026-10-06**. The full five-slice objective remains active.

## Implemented integration

The native forward-AD export now verifies that Q/K/V are a distinct permutation of the three direct frontend arguments. It retains the isolated single-operation/single-return envelope and rejects repeated/aliased primal roles. Native AD and Graph/Schedule/Tile still own the paired O/LSE product, physical tangent activity and argument-role contract.

The automatic program obtains the independently verified reverse checkpoint mapping, projects requested frontend indices into physical Q/K/V roles, checks the native JVP activity contract against that projection, and maps capture inputs and tangent submissions accordingly. No new Python GPU constructor, operation, dtype, target, pass, stable diagnostic, or physical ABI is added. The native CUDA kernel is unchanged.

## Exact RTX 5070 proof

[packet.json](packet.json) contains 72 finite-difference cases: all six frontend permutations, two named profiles (Sk5/noncausal and Sk129/causal), and six semantic wrt orders (Q, K, V, K/Q, V/Q, Q/K/V). Q/K/V shapes differ, so incorrect swaps cannot pass by coincidental shape equality. Maximum absolute tangent error is **1.42741e-8**, under the original `atol=rtol=3e-5`.

Actual caller CUDA inputs are overwritten after private forward capture. Scaled repeated directions, retained prior results and closed-frame refusal pass. The recorder gates the actual NVIDIA GeForce RTX 5070 / SM120, UUID GPU-cba12639-821a-7a10-4cd3-f918f9c0a545, driver 610.88, and retains source/compiler fingerprints.

[image-parity.json](image-parity.json) proves identical native tangent image bytes across all six argument permutations within each of twelve same-shape/causal/wrt groups. Physical scheduling is unchanged. [artifacts](artifacts/) retains every native arena and image.

Preloaded CUDA-event dispatch windows and checked allocating JVP wall samples are recorded separately. These are characterization samples with no throughput, isolated-kernel or speedup claim. Across the 72 rows their medians are 0.011690 ms and 0.342958 ms respectively.

## Validation

- Matching full compiler build: [build.txt](build.txt).
- Native AD, mapping, compact backward, broadcast and registry gates: **430 passed**, [contracts.txt](contracts.txt).
- Updated diagnostic/pass metadata gates: **292 passed**, [metadata.txt](metadata.txt).
- Matching gfx1201 shared compiler build: [.validation-jvp-order/build.txt](.validation-jvp-order/build.txt); **91 shared tests passed**, [shared.txt](.validation-jvp-order/shared.txt).
- Owning gfx1201 unchanged norm/epilogue regressions: **18 passed, 80 deselected**, [device.txt](.validation-jvp-order/device.txt).
- Ruff passes; **11 audit tests pass**, compiler-plan ownership/log links pass, and **all 32 generated documents are in sync**, [audit.txt](audit.txt), [plan.txt](plan.txt), [generated-check.txt](generated-check.txt).
- Source/compiler fingerprints match the recorded revision at slice completion, [fingerprints.txt](fingerprints.txt).
- Graphify CLI is unavailable in the authoritative WSL checkout, [graphify.txt](graphify.txt); no fresh graph claim.

The native refusal before the change is retained in [native-permutation-before.txt](native-permutation-before.txt). The first recorder mistakenly traced causal as a fourth tensor operand; [initial-recorder-failure.txt](initial-recorder-failure.txt) preserves that failure. The final recorder uses an actual three-tensor function with a static causal closure. The initial missing program mapping fixture is retained separately in [before.txt](before.txt).

## Backend and remaining scope

ROCm has shared verification/regression proof only: the native physical JVP Schedule remains explicitly SM120. No gfx1151/gfx1201 tangent execution or CUDA timing transfer is claimed. Apple/x86 shared contracts are host-tested; architecture-owned physical integration is follow-up required. General composed/dynamic/biased/dropout/higher-order attention AD remains open, as do larger W1.1/E2E and independent FP8/MXFP8/MXFP4 programs.

Follow-on: [portable native attention JVP program](../nvidia_jvp_portable_20261006/README.md) changes frontend program validation/serialization. This preceding packet retains its original source/measurement hashes; it is not relabeled as the newer build.
