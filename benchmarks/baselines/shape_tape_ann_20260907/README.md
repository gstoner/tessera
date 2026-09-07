# Shape-varying host tape and GPU ANN arbitration

Owner: F3 / FA-1 / AD-RESIDUAL-EVAL-1 / IR-NATIVE-FOUNDATION-1.

- `shrinking-host.json`: native LLVM x86 execution on Princess-Luna, widths
  4/8/16, two saved logical shape rows, and two backward calls per capture.
  Payload allocation is bounded by 16; logical extents shrink each iteration.
  The recorded JIT includes upstream ownership-based temporary deallocation
  after DPS copies. Trace validation checks emitted deallocations/native frees;
  repeated execution checks saved payloads and derivatives.
- `arbiter-{nvidia,rocm}.json`: actual native GPU ReLU and absolute-value
  programs at zero and adequate rewrite budgets. Original remains the default;
  forced rewrite passes only with its numerical budget. Scoped close removes
  candidates. Targets: RTX 5070 sm_120 and Radeon 8060S gfx1151 respectively.
- `ann-{target}-{1..9}.json`: independent processes, 31 randomized paired samples
  after warmup, original versus rewritten package, both using the serialized
  upstream elementwise-fusion pipeline. Wall time includes H2D/dispatch/D2H.
- Summary files derive medians from raw samples and apply the fixed nine-run
  median order-statistic interval. The 2% lower-bound gate refuses both targets:
  CUDA 0.97773x; ROCm 0.98430x. Neither is promoted.

Recorders: `record_shape_varying_tape.py`, `record_gpu_ann_arbiter.py`, and
`record_native_ann_execution.py --fuse-elementwise`. Source and artifact hashes
bind the packets to the measured worktree. Older packets remain evidence only
for their own fingerprints. No Metal proof or kernel-clock measurement is
claimed. These serial schedules do not establish tuned parallel ANN performance.

Open: arbitrary CFG recovery, dynamic GPU residual slots, reader-complete async
reclamation, reduction/spectral error consumers and measured tuned promotion.
