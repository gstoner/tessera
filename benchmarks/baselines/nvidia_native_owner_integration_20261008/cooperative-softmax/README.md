# Isolated cooperative SM120 softmax lowering
Owner W1.1 / E2E-REAL-6.
Sync NVIDIA-COOPERATIVE-SOFTMAX-2026-10-08.

Matching integrated long-chain events identify serial softmax at approximately 0.992 ms versus 0.057 ms matmul for K4096. This isolated native lowering candidate adds opt-in schedule=cooperative_128: one CTA owns a row, lanes scan strided columns, fp32 max/sum reduce through shared scratch, and a broadcast barrier protects scratch reuse. Default serial lowering is unchanged. Invalid schedules are refused.

Compiler build/IR proof and owning numerical/timing evidence are pending. Graph-to-Schedule/Tile hashing, Tile verifier policy, package geometry, portable/prepared/resident launch contracts and all dtype/nonfinite envelopes must be connected before public admission or promotion. This is not a Python backend constructor or an executable frontend proof.

Compiler proof complete: three storage fixtures, byte-identical serial output, 54 cooperative barriers and invalid-schedule refusal. Candidate compiler SHA 6429d819562c32af6c4432f97d70f2aa792e30e3f299c6f3ef23ba9f5fe69f8c. Compiler-only evidence, not execution or timing.

## Native Graph/Schedule/Tile contract proof

The isolated contract compiler SHA is b8fe61ff3e13322eb43b096a035e75ac2bdbd0bd3810eb810b8714317918029e. The earlier compiler_proof.json belongs to the initial lowering-only compiler; schedule_contract_proof.json and frontend_contract_proof.json belong to this complete upstream contract candidate. Default serial Graph/Schedule/Tile output remains byte-identical to primary. Three storage policies replay exactly and refuse altered policy/geometry. Six actual public frontend traces preserve Graph ownership and alias Tile equality. Forty isolated scheduled-kernel regression tests pass under project configuration. Source hashes: contract_source_identities.json; reconstruction deltas: schedule-contract.patch, tile-contract.patch, python-contract.patch and test-contract.patch.

Frozen proof scripts: check_schedule_contract.py and check_frontend_contract.py. They capture the authorized scratch environment and require reconstructing the candidate source/tool from the patches; no binary is checked in. Runtime packaging, checked geometry dispatch, GPU numerics and timings remain pending. No execution/promotion claim.

### Isolated full-package execution and matched softmax A/B

The matching native compiler and isolated CUDA runtime now execute the explicit
cooperative policy through actual frontend tracing, native Schedule/Tile,
NVIDIA Target IR/PTX and checked host/resident launch. There are 51 profiles
(102 host/resident numerical checks), covering FP16/BF16/FP32, rank two/three,
K=1/17/257/4096/4097, constant large rows, NaN and positive/negative infinity.
The independent oracle uses float64 stable softmax. Source/binary identities
are recorded in package_device_proof.json.

Five alternating windows compare serial and cooperative packages using the
same compiler/runtime, with correctness before and after each window.
For 128x4096/4097 across the three storage types, resident CUDA-event speedup
is 81.86–98.74x; separate host-call speedup is 2.25–2.96x. These compare the
existing serial implementation with the explicit candidate, not a vendor
baseline or full producer chain. The 3x17 rows show no consistent benefit.
All raw samples, including outliers, are retained in softmax_ab.json.

The first package attempt exposed a stale Python serial entry projection,
which is corrected in the candidate. A second attempt selected the original
NVIDIA target tool; the successful proof explicitly binds both Schedule and
Target lowering to the candidate compiler. Failed logs are preserved.
Primary compiler/runtime admission remains unchanged. Prepared producer-chain
attachment, public automatic schedule selection, authoritative integration,
broader workload checks and publication remain open.

Four forged geometry/storage/policy descriptors are refused before CUDA registration.
The repaired candidate contract suite passes 40 tests; its earlier stale
diagnostic expectation failure is preserved separately.

### Cooperative softmax prepared producer-chain execution

The isolated package API now projects an explicit softmax schedule onto its
copy of the traced Graph before native outlining. Native member Graph,
Schedule hash, Tile symbol and buffer lifetime certificates bind the policy.
The CUDA prepared owner recognizes the exact serial/cooperative softmax
symbols and verifies the cooperative launch flag. Existing native ownership
and dynamic multi-producer admission guards remain unchanged.

Twelve RTX 5070 profiles cover FP16/BF16, one/two/three producers, ragged
17x35x19 typed-fragment consumers and 128x4096x64 macro consumers.
Independent float64 oracles round each producer to its declared storage.
Changed-input reuse passes; previously returned outputs remain bit-identical.
Maximum error in changed-input checks is 2.2273e-6. The chain runtime has its
own identity (8d685c33da88f5384ffae5d99a1ea02a988068e88a88e899ed972bc26f53897a).
These are execution/lifetime checks; matched complete-chain timing, public
schedule selection and authoritative integration remain open.

The matching candidate passes 111 existing native partition and owning macro
producer regression checks. This does not replace a fresh aggregate unit run.

### Matched complete-chain host timing

Four FP16/BF16 two/three-producer profiles at M128/K4096/N64 compare serial
and cooperative softmax with the same consumer image and matching candidate
compiler/runtime. Five alternating windows of eleven calls include native
copies and completion. Correctness passes before and after each window.
Median complete-chain host speedups are 3.317–3.362x. Raw samples and
outliers are preserved in chain_ab.json. This host measurement is separate
from the standalone resident CUDA-event packet; it is not a chain device
kernel timing claim. Automatic selection and authoritative integration
remain open, along with the broader five-slice program.

### Authoritative explicit-policy integration

The proved explicit cooperative policy is now integrated in native Graph
Schedule selection, checked Tile projection/verifier, NVIDIA lowering,
Python package/runtime ABI checks and the native prepared owner.
The matching authoritative compiler and CUDA runtime rebuild successfully.
The default remains serial. This integration does not enable dynamic
multi-producer admission or automatically select a physical policy.
Five native positive/negative softmax FileCheck fixtures pass, including
the unchanged serial path and rejection of workgroup size 32.

Durable tests in tests/device/nvidia/test_cooperative_softmax.py cover
host/resident nonfinite/ragged numerics, one/two/three-producer lifetimes and
pre-registration descriptor corruption. Matching-source regression validation
is running; prior isolated packets do not substitute for its final result.
The owning recorder is benchmarks/nvidia/benchmark_cooperative_softmax_chain.py.

The first combined authoritative lane finishes with 674 passed and 14 failed.
All fourteen failures are retained pre-F2 softmax baseline descriptors whose
serial schedule spelling is thread_per_row_128. The runtime now explicitly
normalizes that historical spelling to serial while checking the exact symbol,
dtype ABI and serial geometry. Frozen numerical baselines are unchanged.
The affected owning lane is rerunning. The initial failure log is preserved.
All 15 documentation drift checks pass. These results do not establish a
green aggregate full-unit gate or generic scaled-product closure.

The affected owning rerun passes all 240 tests, including the unchanged
retained serial baselines and the new cooperative numerical/lifetime/refusal
checks. The matching authoritative serial Target is byte-identical to the
preintegration control (authoritative_serial_compatibility.json). The initial
674-pass/14-failure lane is retained as history, not relabelled green.
An authoritative whole-chain A/B run is in progress. Automatic schedule
selection and generic scaled-product batching/transpose closure remain open.

The authoritative matched whole-chain benchmark completes four profiles.
Prepared host-call median speedups range 3.280–3.405x
for FP16/BF16 two/three-producer chains at M128/K4096/N64.
Each serial/cooperative pair uses the same consumer image, with correctness
before/after five alternating windows. Copies and completion are included;
this is not a chain device-kernel timing claim. Matching current binary and
source identities are in authoritative_chain_ab.json. The durable recorder
executes authoritative imports directly, without candidate AST/module injection.
