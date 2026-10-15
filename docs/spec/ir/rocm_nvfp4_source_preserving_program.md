# Native source-preserving NVFP4 program

Owner: ROCM-NVFP4-INGEST-1 / FRONTEND-IR-MEDIUM-1.
Synchronization key: NVFP4-NATIVE-MULTI-OUTPUT-20261015.

The native scaled-primal export admits an explicit gfx1201 single-block Graph:
five distinct roots (packed codes, E4M3 scales, FP64 projection globals, E4M3
activation, FP32 activation scales), one tessera.nvfp4_expand and one returned
tessera.scaled_matmul. Weight storage is N-by-K with transposeB=true. Scale
granularity is block [1,16], format FP32, accumulation FP32 and execution
exact_per_block. Product storage is FP32 or BF16. Graph verification checks
the expansion's precise numerical policy. Conflicting consumer policies reject.

Schema tessera.native.nvfp4_program.v3 means this static source-preserving
program specifically. It does not reinterpret v1 (lossy static), v2 (lossy
bounded), or any attention schema. Bounded source-preserving frames currently
reject at native export and require a separate future contract.

Native MLIR assigns roots 0..4 in original argument order, expansion outputs
5 and 6, and product output 7. Role indices preserve permuted frontend arguments.
Producer results have first_write=0, last_read=1 and private_scratch ownership.
Product output has first_write=1, last_read=2 and returned_output ownership.
Outlined members clone actual operations with their operands, results, numerical
attributes and module target. Python consumes serialized metadata and checks
projected members against the original export; it does not reconstruct Graph.

The additive TesseraRocmProgramMultiOutputStep and
tessera_rocm_program_prepare_multi_output preserve the single-result ABI.
Inputs precede outputs in flattened kernel memref arguments, then scalars.
Total memrefs per stage are at most seven, inputs at most six, scalars at most
eight. Fresh output slots must follow compiler SSA result order. Native C++
independently derives first writes and last reads and checks all declarations
before allocation. The image-plan cache includes output identities. Private
intermediates cannot be read as returned outputs. Existing ordered submission,
generation-checked transactional readback and teardown rules remain.

The owning-device fixture executes compiler-derived plans through the native
ABI directly. Public composed JIT/prepared binding, import-once weights and
partial activation updates remain separate work.
