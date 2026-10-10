# Native prepared attention JVP

Owner: AD-RESIDUAL-EVAL-1; siblings FRONTEND-IR-MEDIUM-1 / W1.1.
Synchronization key: NVIDIA-PREPARED-ATTENTION-JVP-2026-10-06.

The Python frontend and verified native Graph/paired AD/Schedule/Tile/NVVM/LLVM route remain authoritative. A new native C++ owner retains the forward and tangent modules, compiler-produced sizing library, aligned nine-buffer arena and CUDA events. Native code binds frontend inputs and requested tangent roles. Python retains bounded registration and checked host argument binding. The GPU bodies are unchanged.

## Exact-device evidence

RTX 5070 / sm120, driver 610.88, Super-Bear WSL. Both current source fingerprint sets match the packets.

- 72 matched common-runtime A/B cases, seven alternating rounds, independent FP64 primal and centered finite-difference tangent oracles. Maximum absolute error 1.5232e-8.
- Median per-case prepared/unprepared wall ratio: 0.116622 (about 8.57x faster). Across-case wall medians: prepared 1.4160 ms; unprepared 12.0721 ms. These include checked synchronous host dispatch, copies and execution.
- Separate forward/tangent CUDA event windows are recorded per case in packet.json; they are not compared with wall time.
- 72 ordinary public native_jvp cases pass. Warm public wall median 2.9325 ms is characterization, not a matched public speedup.
- 443 focused host contract/registry/ABI tests and 12 exact-device tests pass. The intentional multithreaded fork test emits one expected deprecation warning.
- Three externally pinned fresh-process common-runtime replays pass with compiler subprocess creation forbidden after runtime discovery.

The A/B arms consume identical pinned images. The control uses the previous real compiled native adapter, not an eager or metadata approximation. Static short noncausal and long causal profiles cover six frontend permutations and six requested tangent orders. No general throughput or sibling-backend performance claim follows.

## Ownership and failure checks

The owner checks byte extents, distinct output ranges, closed handles, process identity before acquiring potentially inherited locks, and CUDA context pointer plus unique context identity. Python cache keys retain the actual thread object, avoiding recycled thread-ID reuse; atfork drops inherited registrations without CUDA cleanup. Host output arrays remain caller-owned. Execution is synchronous; asynchronous overlap and arbitrary layouts are outside this proof.

Four private C ABI exports have declarations, definitions and generated runtime ABI registry entries. No operation, dtype, target, pass or diagnostic is added.

## Remaining work

Unused reverse image compilation and serialization remain open, though this forward-product execution loads only forward and tangent images. General native VJP dispatch, composed/dynamic/bias/dropout/higher AD, sibling physical tangent consumers and full five-slice closure remain open. CUDA-specific ownership is not applicable to Apple/ROCm/x86 physical execution; shared host contracts are tested, and architecture-owned native consumers require follow-up.

Files: packet.json, public_packet.json, artifacts/, *-replay.json, contracts.txt, device-tests.txt. Historical packets are retained under history-* with their original fingerprints.

Follow-on: [forward-only compiler product](../nvidia_forward_attention_jvp_20261006/README.md) retires unused reverse executable compilation/serialization. This packet retains its original fingerprints and measured scope.
