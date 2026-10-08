# Bounded public JIT native attention AD — 2026-10-08

Owner E2E-REAL-6 / AD-RESIDUAL-EVAL-1.
Sync NVIDIA-ATTENTION-BOUNDED-JIT-2026-10-08.

The public frontend records a positive Sq/Sk capacity request without changing
traced operand types. The native paired AD pass projects the named isolated
attention Graph, verifies fixed batch/head/width and trace capacity, creates
symbolic saved O/LSE and adjoints, and exports sealed Schedule/Tile products.
Native image packaging carries the capacity policy into checked ABI guards.

Matching LLVM/MLIR 23.1.1 compiler build succeeds. The initial focused lane had
69 passes and a fixture using integer wrt identifiers instead of argument
names. Its correction passes all 70 tests; both logs are preserved.
Eight owning RTX 5070 cases pass across six actual sequence shapes, full bias
or no bias, compact/noncompact gradients and synchronous/asynchronous capture.
Serialized programs replay with subprocess compilation forbidden. Changed
cotangents and different-shape generations preserve earlier gradients.

The recorder compiles the public tuple-result JIT with native paired AD and
compact requested gradients, restores its serialized program, and validates
O/LSE and selected gradients before and after timing. Eight timing arms retain
one forward/backward image pair per bias policy across runtime shapes.
Maximum recorded gradient error is 7.22480105186385e-8.
Completed capture medians: 3.879593–5.496566 ms.
Completed backward host medians: 0.158535–0.203479 ms.
Complete-owner CUDA event medians: 0.155904–0.201559 ms.

Capture includes host allocations, module load, snapshots and completion.
Backward events include allocations, seed copies, kernels, frees and driver
enqueue gaps. These are characterization measurements, not isolated kernel
timings, counterbalanced speedup claims or automatic policy promotion.

packet.json binds the actual GPU UUID, compiler/runtime binaries and affected
sources, including the native projection/AD pass. Compiler SHA256:
0f22de3dbdbea8d0d3843904897421bf5bacddf3f5d72d8d654df3d89b3ac962.

Still open: dynamic physical broadcast-bias extent carriers, dynamic JVP,
arbitrary composed/nested attention, automatic external-consumer lifetime
tracking, fresh aggregate validation and focused publication. Full bias
execution does not establish broadcast-bias execution. Sibling backend
assessments do not constitute owning-device parity.

## Final affected regression gate

652 host WSL tests pass after updating six static checkpoint test doubles
with the new empty shape_bounds field. The initial six failures and 646 passes
are retained in final_gates_initial.log; final_gates.log records the rerun.
Saved-output/LSE generation mismatch assertions remain unchanged.
This focused gate does not replace the two still-open generic full-suite
batching/transpose failures or establish an aggregate green result.
