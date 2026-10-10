# gfx1201 immutable checked owner storage metadata

Owner E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Sync SCALED-OWNER-METADATA-2026-10-08.

The cached checked ABI now retains immutable argument count, storage shapes, dtype tokens, byte capacities and output slots. Native preparation, input binding and output reads consume this metadata. The complete per-owner diagnostic JSON snapshot is lazy and remains independent; mutating its shapes, counts, bytes or output list cannot change launch inputs or suppress checked output reads. No native HIP ABI, numerical policy, image, Graph, Schedule or Tile arithmetic changes.

Validation: 25 matching-compiler host unit checks pass, including reencoded package/witness mutations and diagnostic input/output forgery. 80 RX 9070 XT/gfx1201 device checks pass, including independent owner/cache isolation, primal/JVP/VJP numerical execution and compiler-free warm calls. One existing timeout-plugin configuration warning remains.

Recorder: benchmarks/rocm/benchmark_scaled_owner_metadata_ab.py.
Historical control: benchmarks/baselines/gfx1201_scaled_owner_metadata_20261008/owner_control.py.

Matched counterbalanced A/B: 48 profile executions, 24 matched comparisons, four batch-owning operands and primal/JVP/VJP. Native program, member-image and frontend-Graph hashes match in each pair. Maximum numerical error 1.15684470559e-07. Median control/candidate public warm-wall ratio 0.979850675: candidate measured about 2.1% slower. This packet does not establish a speedup or performance promotion. Native event windows are recorded separately and include program dispatch; no isolated kernel claim.

GPU name/UUID, compiler/runtime and control/candidate/recorder hashes are in matched-ab.json. The candidate hash matches the coordinated source. Kernel and ABI checks remain native; this is an adapter metadata/lifetime enhancement, not a Python numerical backend.

Remaining: attribute total public launch overhead, tune or reassess lazy parsing from evidence, broader image-key families and general scaled batching/AD closure. CUDA/Metal/x86 and gfx1151 require separate owning proof.

Historical candidate bytes are preserved in owner_candidate.py; the newer dtype-binding change has its own separate packet and timing comparison.
