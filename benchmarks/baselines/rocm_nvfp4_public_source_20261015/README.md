# Public native NVFP4 checkpoint composition

Owner ROCM-NVFP4-INGEST-1 / FRONTEND-IR-MEDIUM-1.
Synchronization key: NVFP4-NATIVE-MULTI-OUTPUT-20261015.

## Executed envelope

An ordinary five-host-input Python JIT function expands packed NVFP4
checkpoint bytes into exact E4M3 values and FP32 K16 scales, then runs the
native block-scaled FP8 product. Native MLIR outlines actual SSA and owns
Schedule/Tile/ROCm/LLVM lowering. Both producer outputs remain private GPU
storage. Python only captures Graph and marshals compiler-exported metadata.

The public envelope is static rank-2 FP32 output on RX9070XT/gfx1201.
Cold/warm keyword calls, changed bytes in the same host weight allocation,
JSON replay and fresh-process replay execute correctly at K16/K48/K80.
The child disables both compiler paths, packaging and eager numerical oracles.
This establishes compiler-free replay with a matching prebuilt native provider,
not a Python semantic fallback. Compile reports carry the original Graph hash,
ordered native Schedule/Tile/Target hashes and the artifact identity.

## Original checkpoint and numerical proof

The packet uses both original pinned Qwen3 gate/up NVFP4 projections and
the same captured BF16 layer-0 MLP input source as earlier format controls.
All checkpoint/capture identities and projection boundaries/globals are retained.
M,N,K is 256,24576,4096. Activation quantization is benchmark input preparation;
the compiled program consumes explicit E4M3 activations and FP32 K16 scales.

Independent nibble decoding and FP64 projection define the arithmetic oracle.
Every output satisfies the existing FP32 accumulation bound; there are zero
violations. The three modes produce identical output bytes. Native error vs
decoded FP8 projection is 2.568e-07 relative RMS.
Native projection RMS vs BF16 original is 6.2286%;
the shipped NVFP4-weight projection with unquantized inputs is
6.0802%. This is one combined
layer-0 gate/up projection, not a full MLP/model or perplexity acceptance.

## Timing and host-cost attribution

| Mode | Completed host median (ms) |
| --- | ---: |
| Public JIT | 79.386 |
| Serialized replay | 80.040 |
| Prepared activation update + consumer + read | 10.706 |

Three matched alternating-order windows check outputs after every timing.
Startup packaging, prepare/upload and initial import are recorded separately.
Per-member captured HIP graph-dispatch medians are 0.756 ms expansion and
1.327 ms product, using 64 repeats with all measured windows above 20 ms.
These are graph-dispatch intervals, not instruction-counter or host time.

Each public/replay sample records one native allocation-cache miss.
The device buffers alone total 208928784 bytes, exceeding the
existing 128 MiB idle-retention budget; native staging adds further cost.
Public/replay upload 57933840 input bytes per call.
The explicit prepared strategy retains expanded weights and uploads only
1310720 activation bytes per call, with no cache acquire.
The timing difference includes allocation, validation, transfer and expansion;
the packet does not assign a percentage to each contribution or claim a
model-serving speedup. The idle-cache budget and physical schedule are unchanged.

## Gates and reproduction

A frozen host-WSL gate passes 829 cases covering public/replay/prepared execution,
legacy native contracts, primal/JVP routes, audit lifecycle and diagnostic/pass
metadata. Strict imported-contract mypy and Ruff pass for the changed modules.
Compiler version is LLVM/MLIR23.1.1 with assertions. Full source/compiler/provider,
image/binding identities, raw samples and compile report are retained.

Source the matching host environment and set the native image/movement provider
to the recorded multi-result runtime. Run
benchmarks/rocm/benchmark_nvfp4_original_composed.py with --activations pointing
at nvfp4_model_activations_20261015/inputs.npz, --m 256 and --output.
Inputs/checkpoint shards are reused from checked scratch caches, not committed
again. Child replay needs no compiler, but requires the matching native provider.

Remaining: bounded/wider composition, representative model-quality acceptance,
and wider five-slice performance/NVIDIA obligations. Native target relabeling
rejects before provider access. No gfx1151, CUDA or Apple evidence is inferred,
and no default numerical/physical strategy is promoted.

## Post-PR949 synchronization

Merged main 51749071b without compiler/runtime source changes. The focused
owning-device plus audit/diagnostic/pass gate passes 353 cases. All 33 generated
documents and the compiler-plan gate pass. Initial reruns exposed missing and
stale provider environment settings after WSL restart; the final gate explicitly
sets MULTI_OUTPUT, NATIVE_IMAGE and NATIVE_MOVEMENT to the recorded partial
program provider. Red/configured/final logs are retained separately.
