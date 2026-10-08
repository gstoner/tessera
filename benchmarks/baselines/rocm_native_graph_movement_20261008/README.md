# Native-owned ROCm movement graph replay — 2026-10-08

Owner E2E-REAL-6. Synchronization ROCM-NATIVE-GRAPH-MOVEMENT-20261008.
Base PR895; this follow-up does not resolve generic scaled-product closure.

The existing ordinary JIT frontend and native Graph/Schedule/Tile/ROCm/LLVM
packages supply the images and sealed ABI. The native resident owner now
captures the actual one-kernel movement or two-kernel paged-KV/softmax sequence.
Python binds the request; it constructs no HIP nodes, kernel bodies or schedules.
One native replay call owns submission and completion. Image leases, private
allocations, stream, output generations and graph resources share one owner lock.
Graph executables retire before their referenced buffers/functions. Failed
cleanup retains the owner for explicit retry. Uploads change contents without
changing captured addresses. The prepared handle may close independently.

Use the existing owning JIT preparation to obtain a resident owner, then call
`owner.capture()` after upload and `owner.execute(captured=True)`. This is
explicit selection. Direct submission remains the default for every profile.
Old native libraries continue to support direct execution; requesting capture
from them raises an explicit unsupported-library error.

## Validation

- Super-Bear WSL: 325 focused host checks pass, seven owning-ROCm cases skip;
  Ruff and the zero-error mypy ratchet pass. A further 101 ABI/dashboard/
  generated-document regression checks pass with no skips.
- Princess-Luna live gfx1151: 25 tests pass, no skips; seven executed graph
  profiles cover small/large paged reads, MoE token gathering and three
  paged-KV/softmax pairs. Fork/type/library guards are included in the total.
- Tajasaurus live gfx1201: eleven graph/host checks plus twelve resident
  regressions pass, no skips. Five executed graph profiles cover paged reads
  and paged-KV/softmax. gfx1201 prepared MoE dispatch remains unadmitted.
- Device checks prove capture-before-upload/invoke-before-capture refusal,
  idempotent capture, prepared-handle retirement, changed source/index contents,
  compiler-forbidden warm execution, retained downloaded output, stale generation
  refusal and close. Native captures verify the actual one/two kernel node count.
- The modified C++ runtime was rebuilt on both owning hosts. Compiler tools are
  the existing recorded LLVM/MLIR 23.1.1 builds; this is not a new whole-compiler
  matching-source build claim. Each packet records tool/runtime and source hashes.
  All four modified execution/recorder source hashes match both device packets.

Initial capture-node and host-environment failures are retained in compressed
logs. They are superseded by terminal successful owning runs.

## Counterbalanced A/B characterization

Recorded by `benchmarks/rocm/benchmark_native_graph_movement.py`. The recorder
uses the same owner, allocations, images, entries and inputs in both arms.
It validates independent numerical results before and after every arm, and after
changing inputs/index contents. Warm compilation is forbidden and image load/
unload counters cannot change. Seven alternating A/B pairs per profile retain
all samples. Whole short pairs are retained and retried until each completed
host window lasts at least 20 ms. Host costs exclude upload and download;
they include Python binding, native submission and synchronous completion.

| Owning architecture | Single movement host cost, graph/direct | Paged-KV → softmax host cost, graph/direct |
| --- | --- | --- |
| gfx1151 | 1.083–1.135× | 0.864–0.954× |
| gfx1201 | 1.375–1.699× | 0.772–0.957× |

Single-kernel graph replay is slower in these samples. Paired graphs reduce
completed resident-call cost by about 4.6–13.6% on gfx1151 and 4.3–22.8% on
gfx1201. This is a within-image submission comparison, not a kernel speedup or
a comparison against a vendor kernel. Direct samples retain separate producer
and consumer event intervals. Graph samples time the complete captured sequence
externally: those event scopes differ and their ratios are not kernel gains.
No automatic selector or default is promoted.

Reproduce from the repository root with the owning HIP/compiler environment:
`python benchmarks/rocm/benchmark_native_graph_movement.py --architecture gfx1151 --output <scratch-directory>`
(or gfx1201). Device packets contain exact samples, architecture/name/PCI,
source/tool/runtime hashes and rejected short windows. General KV layouts,
asynchronous external consumers, generic scaled batching/transpose and original
NVFP4 checkpoint quality remain separate open obligations.

## Final ancestry refresh and timing variability

The final recorder verifies adjacent Graph/Schedule/Tile/Target/native lineage
for **both** producer and softmax consumer, and stores these witnesses per row.
The table above uses this final series. The previous series is retained with its
original recorder hash as `previous-series.json.gz`; it is not current-source
evidence. The final complete raw samples are `device.json.gz`, with a readable
`summary.json` that preserves source/tool identities, medians and sample counts.
Differences across these series show timing variability. These are scoped
characterization results; they do not establish a stable speedup or promotion.
