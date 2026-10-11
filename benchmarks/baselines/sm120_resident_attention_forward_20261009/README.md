# Ordinary resident attention forward: native execution evidence

Owner: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Synchronization: SM120-RESIDENT-ATTENTION-FORWARD-20261009.

RTX 5070 (SM120), Super-Bear WSL, CUDA 13.3 and matching LLVM/MLIR 23.1.1.
Each packet records live GPU UUID/driver/architecture, source/compiler/provider
hashes, descriptor/image identity and adjacent native compiler-stage digests.

Route: Python frontend → typed Graph MLIR → native Schedule/Tile MLIR →
NVIDIA Target IR → LLVM/PTX → checked native owner → execution.

Twelve profiles per packet: FP32 plain and saved-LSE causal GQA with broadcast
bias at K=5/129, plus FP16/BF16 ordinary causal GQA with/without dense FP32
bias at K=5/129. Rank-four inputs are compact. Ordinary output preserves
the declared input/result storage; saved-LSE profiles use FP32 output and a
separate rank-three FP32 LSE. Native arithmetic accumulates and normalizes
in FP32 before final half-result truncation. Existing half-input/FP32-result
package identities remain distinct from new half-result identities.

Resident and host frames share the same compiled image and retained owner.
Independent FP64 attention output/LSE checks, rounded to the declared output
storage, precede nine alternating timing rounds and follow every timed call.
Device events measure the forward kernel only. Completed public calls include
private snapshots, producer ordering, binding and host output/LSE downloads.

Maximum error against the rounded oracle: 4.31066e-07.
Resident/host completed-call ratios span 1.050–1.253; no speedup is claimed.

Validation: 56 forward/component/adjacent resident AD device cases plus four
serialized-package half-output tests pass. The latter retain sentinel bytes
before/after the output span. Public short/long storage cases exercise pending
producer streams, reordered bias roots, warm calls with compiler/tracing/eager
execution forbidden and retained prior outputs. Native rejection/recovery
checks cover capacity, alignment, producer counts, overlap and product ownership.
519 focused shared drift gates pass (six skips); 61 semantic/packaging tests
pass (six skips). These focused lanes do not establish full-suite/CI success.

Dynamic/composed/nested tuples, half saved-LSE/AD and broader backend parity
remain open. This packet closes the named static execution profile only.

Reproduce from the repository root in matching host WSL:

    python benchmarks/nvidia/record_resident_attention_forward.py --output run.json --repetitions 9

Use matching production core/NVIDIA tools and a fresh native provider.
Changed source or binary fingerprints require a fresh packet.

Final delivery gates: 110 claim/generated-registry/audit/runtime ABI cases pass;
all 32 generated-document drift checks and the integrated plan lifecycle gate pass.
