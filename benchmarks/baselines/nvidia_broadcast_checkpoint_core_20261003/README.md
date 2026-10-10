# SM120 broadcast checkpoint native arithmetic core

Owner E2E-REAL-6 / AD-HIGHER-1. Sync NVIDIA-BROADCAST-CHECKPOINT-CORE-2026-10-03.
Implementation is landing; the checked broadcast package/tape ABI is unfinished.

## Native implementation

Public paired AD preserves the rank-four physical score-bias type. Every bias
axis must equal one or the corresponding logical B/Hq/Sq/Sk extent. The bias
cotangent has the physical input shape. The Graph verifier and native checkpoint
Schedule contract agree; physical shape and reduction policy enter the Schedule
hash and replay. Tile retains the shape and the NVIDIA consumer indexes broadcast
inputs without a dense expansion.

One GPU thread owns one physical bias-gradient element. Native structured loops
sum all matching logical scores in B/Hq/Q/K order, including grouped-query and
end-aligned causal masking. There are no atomics or dense dBias intermediates.
The existing full-shape path retains its direct per-score writer. The shared
Tile verifier now counts and validates the saved-output/LSE bias-gradient pointer,
including its f32 storage and shape attributes.

Generated-checkpoint projection now matches the logical shape field by a word
boundary, so the earlier alphabetic bias_shape field cannot be mistaken for it.
Public frontend tracing/export tests cover the resulting physical cotangent shape.

## Exact-device evidence

rtx5070.json queries RTX 5070 UUID, driver 610.88 and compute capability 12.0.
Twelve rows at B/Hq/Hkv/Sq/Sk/D/Dv = 2/4/2/5/7/4/3 cover each individual broadcast
axis, combined B/H/Q, and all-axis broadcasting, full and causal.
The independent f64 oracle repeats KV groups, computes softmax and Q/K/V/dense
score derivatives, then reduces the dense bias derivative to the physical shape.
Forward O/LSE and all four gradients pass; maximum gradient absolute error is
below 1.1e-7. Thirty-two poison words after each physical gradient stay untouched.

The compiler-authored checkpoint Graph goes through verified native Schedule,
Tile and NVIDIA Target/NVVM/PTX. The recorder launches raw resident native entries
on an explicit session stream, with session-owned buffers. This is an arithmetic
and capacity gate; it is NOT checked broadcast package/tape execution.

Five CUDA-event windows per backward row follow numerical proof and warmup.
Each has 100 resident launches; windows include bridge/driver dispatch gaps.
Post-timing gradients and guards pass. No isolated instruction timing or speedup
claim is made; the highly collapsed bias uses serial per-owner reductions.

contracts.txt retains native frontend/export, hash-mutation, malformed-shape,
temporary package gate, existing checkpoint and diagnostic/pass registry proof.
existing-device-routes.txt separately rechecks existing exact-shape packages.
Compiler and source hashes retain the precise recording revision.

## Required next work

Add a distinct checked broadcast checkpoint ABI that preserves physical input
and gradient capacity. Wire host copies and launch geometry from four physical
bias extents; use the same shape contract for private saved-state capture,
backward output allocation and producer/consumer pairing identity. Complete
public JIT capture/repeated backward, stream/lifetime and malformed-descriptor
proof, including saved and recompute oracle comparison and separate device/E2E
timings. Until then, package_scheduled_checkpoint explicitly rejects broadcast
metadata before compiling an old dense-copy descriptor. This is a temporary
integration guard, not the completed feature.

Apple, ROCm and x86 share the mathematical Graph/AD verification change but have
no native physical consumer proved here. Those queues require their own native
consumer/ABI assessment and exact-device follow-ups; CUDA execution does not
establish sibling parity. Dynamic bias extents, lower-rank normalization,
broadcast JVP/higher derivatives and general composed AD remain open.

Pairing follow-up: the checkpoint identity binds physical bias extents and the fixed reduction policy. Full-shaped bias preserves its existing identity. 51 focused contract/resident/JVP tests pass, including mismatched-pair refusal before compilation. Registry/audit gates passed 303 tests; final authored-document audit passed 11. All 32 generated documents were regenerated and the compiler plan gate passed. The temporary production package gate remains; Graphify is unavailable in WSL.
