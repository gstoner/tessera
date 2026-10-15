# Native multi-result expansion/product diagnostic

Owner ROCM-NVFP4-INGEST-1 / FRONTEND-IR-MEDIUM-1.
Synchronization key: NVFP4-NATIVE-MULTI-OUTPUT-20261015.

RX9070XT/gfx1201 executes native MLIR-outlined expansion followed by a native
Schedule/Tile FP8 K16 consumer. Both expansion outputs stay in private native
allocations; only the final product is downloaded. Three small ragged shapes
exercise K16/K48/K80. Independent nibble decoding, distinct projection globals
and activation scaling define the FP64 oracle. Outputs are checked before and
after timing. Power-of-two scales do not replace the separate expansion
rounding-boundary tests.

Packets record exact device/compiler/provider/image/Graph/plan identities,
three 4096-repeat per-member captured HIP graph windows and three completed
native invocation samples. Member timings include graph dispatch and exclude
capture/instantiate. Native invocation includes submission/completion, excluding
prepare, uploads and readback. These different clock domains cannot be combined
into a speedup. Tiny shapes characterize integration, not serving throughput.

Reproduce in host WSL on gfx1201 with matching LLVM/MLIR23.1.1:
set TESSERA_GFX1201_DEVICE_PROOF=1, TESSERA_ROCM_MULTI_OUTPUT_LIB to the new native
owner provider and TESSERA_MULTI_OUTPUT_EVIDENCE_DIR to scratch output; run
tests/device/rocm/test_native_multi_output_owner.py. Without the evidence
directory, correctness/admission tests run without timing.

Public composed JIT/prepared bindings, import-once/activation-only updates,
bounded source-preserving frames, larger checkpoint measurements and full-model
quality acceptance remain open. No sibling hardware or promotion claim.

## Diagnostic measurements

| MxNxK | Expansion graph dispatch (us) | Product graph dispatch (us) | Completed native invoke (ms) | Max error |
| --- | ---: | ---: | ---: | ---: |
| 19x17x48 | 3.961 | 6.373 | 0.460 | 0.0 |
| 1x7x16 | 3.814 | 5.588 | 0.492 | 0.0 |
| 33x23x80 | 3.862 | 10.813 | 0.809 | 0.0 |

Final focused native/ABI/device/audit/registry gate: 359 passed. The source
manifest binds the measured files and compiler/provider identities in packets.
