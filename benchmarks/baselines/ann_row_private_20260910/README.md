# Row-private ANN and checked dynamic readers

The native pass reuses its row-independence proof to reserve one row for each
entry-owned mutable ANN buffer. Loads/stores into those private buffers use row
zero; external input/output indices retain the owning GPU row. Copies expand
before compaction, static dimension queries preserve original logical extents,
and constants/nested generation slots remain complete. An internal SSA set
carries authorization through cloning; user IR attributes do not authorize it.

The 64x8 terminal-square workload now fits the unchanged 4096-byte bound. CUDA
SM120 and ROCm gfx1151 independently pass native numerical oracles. Single-run
original/rewrite package ratios are about 0.991x/0.994x: incumbents remain. These
are not old/new compiler comparisons and do not authorize performance promotion.

CUDA Nsight records median kernel 879.5 ns, H2D 288 ns and D2H 320 ns; host launch
API 19,615 ns and D2H API 49,645.5 ns. Samples include warmups and oracle calls.
Host and device costs are separate timing domains. ROCm counters remain open.
Reproduce with record_native_ann_execution.py using --rows 64 --width 8
--activation square --fuse-elementwise --parallel-rows on each owning backend.

Checked dynamic public frames and paired source-VJP products also expose
multi-stream read scopes. Host tests cover successful shape checks, refusal
before completion, heterogeneous product views and external-failure unwinding.
These are host ownership tests, not new exact-device AD-frame evidence.

251 focused compiler/ownership/registry tests passed with four environment skips.
General recovery, native exception allocation/collection, CPython deoptimization,
general saved products, attention raising and tiled SSD remain open.

Assertions-enabled compiler SHA-256 used on both owning GPUs:
`c193fe91f9a1f1aeb9c3a4ab53e4555de1d8b602d5aab7dcb82412b4211698f3`.
After extracting shared reader composition, 38 ownership tests pass; audit
checks (11), plan consistency, Ruff and the zero-error mypy ratchet pass.
All 30 generated-document checks pass in WSL; full unit suite not run.
