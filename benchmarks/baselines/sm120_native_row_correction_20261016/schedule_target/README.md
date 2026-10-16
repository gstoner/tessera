# Native row-correction Schedule, target and device proof

Owner: E2E-REAL-6 / AD-RESIDUAL-EVAL-1 / FRONTEND-IR-MEDIUM-1.
Synchronization: SM120-NATIVE-ROW-CORRECTION-20261016.

The original textual Graph passes through native paired AD, a content-sealed
Schedule, typed launch-level Tile and NVIDIA/LLVM/PTX lowering. The residual
producer computes ordered FP32 sum(P * dot(dO,V)) minus the optional dLSE seed,
without rounded saved O. The existing inline backward and new producer share
the reduction emitter. Source-pair lineage, argument roles, dtype/shape facts,
row bytes and geometry are hashed and replay-checked. Only compact row-major
physical inputs are admitted; preserved unsupported layouts are refused.

The portable production-driver lane passes 434 target/AD/registry cases with
zero skips/failures. RTX 5070 / SM120 passes 192 device cases with zero skips:
96 new producer cases and 96 existing forward/backward cases after the math
refactor. New cases execute the same image with two changed input/seed sets,
compare against an independent FP64 probability oracle using actual saved FP32
LSE, and check guarded output stores. They cover FP16/BF16, half/F32 dO, GQA,
causal/noncausal, physical broadcast bias, optional row seeds, Sq=1, Sq>Sk and
ragged Sk=129. The device lane compiles and assembles PTX before execution.

record.json binds source snapshots, production compiler/tool/provider hashes,
exact GPU identity and JUnit receipts. Failed discovery runs are preserved and
identified: a missing native Tile worklist registration was repaired and rebuilt
before the successful runs; an incorrect NVVM test spelling was corrected after
inspecting generated target/PTX.

This is executable producer proof through a diagnostic resident launcher. The
public prepared reverse owner still uses its existing zero-workspace route.
Workspace-consuming backward, additive checked ownership/serialization, public
frontend replay/stream/lifetime proof and separate stage/completed-call timings
remain open. No speedup, automatic selector or full five-slice closure is claimed.
