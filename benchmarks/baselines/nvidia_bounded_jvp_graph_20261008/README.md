# Bounded saved-LSE attention JVP on SM120

Owner E2E-REAL-6 / AD-RESIDUAL-EVAL-1.
Sync NVIDIA-BOUNDED-JVP-2026-10-08.

The typed textual Graph retains symbolic Sq/Sk, explicit seven-dimension
capacities, verified native forward AD and source-shaped inactive zero regions.
Native Schedule/Tile exports actual-size sequence scalars, a schema-2 tensor
manifest and a checked product row grid. GPU pitches, key loops, full/broadcast
bias addressing and the end-aligned causal gap use actual dimensions. Saved
generation identity includes the symbolic policy and capacities. No Python
kernel or capacity-sized substitute computes the product.

## Proof

- Matching LLVM/MLIR 23.1.1 core and NVIDIA target compiler builds succeed;
  CUDA 13.4.59 and owning RTX 5070 / SM120 driver 610.88 were probed.
- 574 host WSL native Graph/AD/Schedule/Tile/ABI/registry/audit checks pass,
  zero skipped. Invalid capacities, grid overflow, altered Schedule contracts,
  source-incompatible inactive regions and static compatibility are covered.
- Ten native programs execute 40 numerical cases with Sq/Sk = (1,1), (3,7),
  (7,3), (9,11), reusing one forward/JVP image per program across shapes.
  Q-only, K-only, V-only, reordered Q/K/V, broadcast-bias and full-bias
  directions are included. Independent FP64 analytic and finite-difference
  oracles, saved natural-log LSE, tangent linearity and retained results pass.
- Maximum absolute JVP error: 3.0811439e-08.
- Three fresh processes replay unbiased, broadcast-bias and full-bias
  serialized programs over nine shapes with compiler/process access forbidden.
- Preloaded JVP CUDA-event medians range
  0.009212–0.013440 ms.
  Complete capture/JVP/close host-wall medians range
  5.169666–6.425011 ms.
  These scopes are recorded separately; no speedup or route promotion is claimed.

## Reproduction

Set TESSERA_OPT and TESSERA_NVIDIA_OPT to the matching tools, PYTHONPATH to
python and the project root, and LLC/LLVM_LINK to LLVM 23 tools. Run the
benchmark module benchmarks.nvidia.benchmark_bounded_attention_jvp with
--output device_full.json on the owning RTX 5070 host. Fresh-process replay uses
benchmarks.nvidia.replay_bounded_attention_jvp with --program, --digest and
--output. All commands run in host WSL.

## Boundaries and repair history

This proves synchronous textual-Graph/native-resident execution. Public Python
@jit dynamic selection, the prepared C++ dynamic ABI, explicit asynchronous
consumer retirement and wider capacities remain follow-up required. Static
prepared ABI and sibling architecture capabilities are unchanged.

The first runtime test pass exposed a null static-capacity access and a
temporary C++ manifest-name lifetime error; both were repaired and rebuilt.
The initial hardware attempt lacked the target compiler. A subsequent recorder
attempt used the wrong image-digest field after its numerical assertions.
Those failures and terminal successful runs are retained as compressed logs.
The earlier receipt.json and symbolic_exports.json are historical Graph-stage
proof, with their original source/tool hashes; runtime_receipt.json records
current implementation hashes and complete proof. No generic scaled-matmul
batching/transpose closure or model-level NVFP4 quality claim is made.
