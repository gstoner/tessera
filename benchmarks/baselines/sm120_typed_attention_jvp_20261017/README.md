# SM120 typed attention JVP integration

Owner: E2E-REAL-6 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Sync: SM120-TYPED-JVP-20261017.
Contract: docs/spec/SM120_TYPED_ATTENTION_JVP_CONTRACT.md.
Recorder: benchmarks/nvidia/benchmark_typed_attention_jvp.py.

## Compiler and execution boundary

Authored Python or textual Graph attention differentiates through the native
TangentInterface, verified Graph AD, Schedule, Tile and NVIDIA/LLVM PTX.
Typed public compilation, portable resident generations and the native static
prepared owner bind FP16/BF16 Q/K/V and tangent roles. Bias and LSE roles stay
FP32. Textual half-input/FP32-output graphs use the checked public owner.
No derivative kernel arithmetic is constructed in Python.

Half-output score-active JVP accumulates an unrounded FP32 weighted-V moment
from the existing V load. Reusing rounded half O produces a different
derivative: rounding-moment.json records the independent FP64 finite-difference
distinguishing controls; rounding-reference.py.xz retains their frozen CPU
recorder. This source does not provide GPU proof. Existing FP32 artifacts retain identical native
arena, image, forward image, ABI, entry and schema across sixteen controls.
Schema 5 still means saved-LSE plus bounded; schema 6 is static saved-LSE.

The shared tensor binder accepts explicit semantic BF16 over two-byte opaque
CUDA/HIP metadata, with physical dtype, width, endian, stride, alignment,
read-only and alias guards. The typed native prepare ABI checks full canonical
forward symbol identity, including the digest suffix: an FP32 output flag
cannot bind a half-output image through a shared prefix. Legacy f32 prepare
ABIs retain their meaning.

## Exact-device proof

Owning host: Super-Bear, RTX5070 / sm_120, WSL, CUDA 13.3 and assertions-enabled
LLVM/MLIR 23.1.1. Final source/tool/runtime/image identities and receipts are
recorded alongside this packet. No sibling architecture evidence transfers.

The public device suite covers static half outputs, saved LSE, causal/open
modes, broadcast bias, active subsets, compiler-free warm replay, retained
generations, pending producer streams, allocation/flag/stream refusals,
trace-at-one bounded shrink/grow/capacity guards, fresh-process compiler-absent
static/bounded replay and authored FP32 output. Prepared and diagnostic host
paths compare bit-for-bit on sixteen cases; independent FP64 checks are
separate. Owning native-member cases include both half and FP32 output roles.

Final combined source/device/registry gate: **650 passed**, four existing
fork deprecation warnings, 164.85 seconds. Focused shared dtype/diagnostic/
pass/half-AD gates separately pass 484 cases; cohorts overlap and are not summed.
The complete production compiler-route gate passes **1,587** cases with zero
skips and a non-empty execution certificate. All **33** generated documents
pass drift checking; audit ownership/routing passes 19 cases.
All five public surface audits and both claim linters pass. The broad CI CPU
lane initially passed 26,310 with four documentation/recorder failures: the
new contract lacked its date, and the recorder check ran before the packet
README was tracked. The date is fixed and the reference is committed; all
37 focused spec/naming/freshness/routing gates now pass. A stable-head broad
rerun remains required. The initial red receipt is retained.
Public CI and complete review-thread audit remain required before merge.

## Timing scope

Eight independent-oracle profiles characterize Sq3/Sk5 and Sq17/Sk129, GQA2,
D=Dv=16, causal saved-LSE attention. Each FP16/BF16 arm has an FP32 storage
control using the same already-quantized values. Compilation and first
preparation happen before timing. Three warmups and nine samples retain
separate native forward/JVP CUDA event windows and completed public calls.
Public calls include metadata checks, upload, native execution and readback.
Oracle checks run before and after timing, outside the measured intervals.

These are characterization rows, not a paired performance promotion. Event
windows are not hardware-counter attribution or universal isolated-kernel
claims. Native program archives and per-member image hashes bind each row.
No automatic schedule, precision rewrite or default selection is promoted.

Reproduce from the owning WSL checkout and matching tool/runtime environment:

    python -m benchmarks.nvidia.benchmark_typed_attention_jvp       --output benchmarks/baselines/sm120_typed_attention_jvp_20261017/timings.json

## Remaining work

Ordinary prepared host JVP is static; bounded packages execute through portable
resident generations. Wider masks/layouts, mixed Q/K/V formats, nested/composed
AD, higher derivatives and ordinary bounded host preparation remain follow-up
work. ROCm semantic BF16 adoption needs its own HIP proof; Apple/x86 receive
no new physical attention admission. All five aggregate engineering rows stay
open. This increment does not establish model-quality or performance closure.
