# Bounded SM120 row-major RHS integration

Owner W1.1 / FRONTEND-IR-MEDIUM-1.
Synchronization key NVIDIA-DYNAMIC-ROW-RHS-2026-10-06.

## Native IR and ABI extension

The named bounded FP16/BF16 normalization/softmax -> matmul program now
preserves compact row-major RHS storage through Graph/Schedule/Tile/Target/
NVVM/LLVM and its checked runtime. Native Schedule admits row-major dynamic
B, derives the row-major composed index map, carries active LDB into tile.view
and gathers typed column-major B registers using explicit transpose intent.

The bounded default retains column-major packing. An explicit
rhs_storage_order="row_major" option on bounded @jit or
compile_native_lhs_matmul requests the row route; "col_major" is also accepted.
An override cannot replace a conflicting authored Graph fact, and projection
does not mutate caller-owned Graph IR. Invalid options are checked before tools.

Two explicit row-major strided ABI identifiers distinguish physical B storage
from the existing column-major variants. The six scalar arguments remain
M/N/K/LDA/LDB/LDD. Dynamic row symbols carry a compiler-owned row_rhs marker.
Native preparation checks that marker against its retained storage contract
and reflects the loaded pointer/scalar parameter ABI before activation.

Native host and resident launchers compute row-major pitch minima and physical
spans; row B requires LDB >= N and span (K-1)*LDB+N. C++ prepared owners derive
compact LDB=N for row B and LDB=K for column B. Active frame validation preserves
immutable capacities, private intermediate lifetime and grow-only shared scratch.
Host C/F and sliced views are packed to the sealed program's physical order.

Resident submission now checks CUDA-interface shapes and physical strides
against M/N/K/LDA/LDB/LDD for both strided layouts before launch. A padded
allocation can execute with matching pitch; a caller cannot declare a compact
or different pitch for it. Generic strided upload still requires compact host
inputs. The test padded view borrows a larger owning allocation explicitly.

No Python GPU body or intermediate numerical computation is introduced.
The ordinary bounded source/live-code certificate and program/context caches
remain in force. No new operation, dtype, target, pass or C ABI export is added.

## Exact-device validation

The matching LLVM/MLIR 23.1.1 compiler tools and CUDA runtime are rebuilt.
Super-Bear reports NVIDIA GeForce RTX 5070, UUID
GPU-cba12639-821a-7a10-4cd3-f918f9c0a545, compute capability 12.0.

The expanded regression lane has 781 passes and four skips, including 406
owning NVIDIA device cases. The new row suite contains 95 cases:
all seven axis subsets x three producers x two storage types x plain/fused
epilogues; public/prepared/portable numerical agreement and no compiler work
on warm shape changes; genuine padded resident pitch and incorrect-pitch
refusal; raw native shape/dtype/byte/pitch/output guard preservation; and two
fresh-process replays with compiler calls forbidden.

An existing partial-static-axis guard test assumed an empty global scratch
pool. It now compares full before/after scratch state, preserving the intended
no-mutation invariant when earlier cases have warmed the shared pool.
See regression-tests.txt and build.txt. The four skips are three generic
static-runtime archive gates and one Darwin/Metal gate; no row-route case skips.
A separate 28-case frontend-contract lane also passes, adding an authored-layout
conflict/copy proof beyond the expanded regression run.

## Matched benchmark and selection decision

Eight independent processes record row/column physical storage with
native-owner/control arms and reversed process order. Each process has 48
correctness-gated cases: three producers, two storage types, two epilogue
modes and four active M/K/N frames under capacity 128/1024/64.
Cold ordinary calls start below capacity at 17/35/19; warm frames reuse one
program and native owner. Public and portable outputs are bitwise equal.

All 384 rows pass. analyze.py verifies exact case coverage, current source
hashes, GPU/compiler/runtime identity and matched images/Schedule/ABI/contract
within each control/prepared pair. Row and column images intentionally differ.
Prepared scratch witnesses remain stable across warmed active shapes.

Prepared/control median ordinary-call wall ratios are 0.0781749/0.0765815
for row storage and 0.0747253/0.0750605 for column storage, approximately
92% lower warm host-call wall time. This is native orchestration coverage.

Row/column median ordinary wall ratios are 1.06658 forward and 1.03469 reverse.
Consumer event ratios are 1.09243 and 1.14834. Row support is numerically valid,
but its gather path still needs tuning. The first pre-option run had 11/14
per-case host regressions above 10%; its historical packets remain in pre_option/.
The bounded default remains column-major, with explicit row admission available.
No automatic performance strategy or quantized format is promoted.

Separate producer/consumer CUDA event dispatch windows include driver gaps.
Ordinary wall includes binding/packing/transfers/two launches/completion/readback;
cold trace/compile wall remains separate. These scopes cannot be interchanged.

## Reproduction

On Super-Bear WSL, source .build-sm120-w1-1/validation-env.sh.
Run benchmarks/nvidia/benchmark_dynamic_row_lhs.py with
TESSERA_DYNAMIC_RHS_ORDER=C or F and both TESSERA_NVIDIA_PREPARED_LHS and
TESSERA_NVIDIA_PREPARED_LHS_REPLAY set to 0 for control or 1 for prepared.
Set TESSERA_DYNAMIC_ROW_LHS_PACKET to the JSON output; run analyze.py from
the repository root after the eight arms. Physical RHS order is explicitly
sealed into each frontend package rather than inferred from later frames.

## Documentation and drift gates

Compiler progress/freshness documents are regenerated. Eleven audit tests,
focused Ruff and diff whitespace checks pass. All four backend queues assess
the shared frontend/storage/runtime changes without transferring device proof.
Graphify query/update were attempted; the CLI is unavailable on this WSL host.

## Remaining scope

This extends the named bounded straight-line host-array route and its resident
consumer pitch contract. General producer composition/bufferization, control
flow, AD, arbitrary layouts, asynchronous/resident program ownership, wider
formats and sibling physical consumers remain open. The full five-slice
objective remains active; no universal compiler closure is claimed.
