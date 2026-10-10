# SM120 producer edge RHS layout integration

Owner W1.1 / FRONTEND-IR-MEDIUM-1.
Synchronization key NVIDIA-LHS-RHS-LAYOUT-2026-10-06.

## Compiler contract

Ordinary static FP16/BF16 RMSNorm, LayerNorm and last-axis softmax on the
matmul LHS now retain C/F RHS storage through the checked producer edge.
The frontend records the RHS storage fact on a copied semantic Graph;
native Graph -> Schedule -> Tile -> NVIDIA Target -> NVVM/LLVM/PTX remains
the executable recipe authority. Python does not construct Tile arithmetic
or shaders. Existing row-RHS ABIs cover plain and bias/ReLU/residual cases
with final FP16 or FP32 output.

The edge validates image/entry/ABI identity before allocation, binds RHS
provenance to its ABI and buffer layout, and normalizes padded host views
to that compiler-owned physical allocation. A layout switch retains a
distinct sealed consumer image/cache key. Producer output remains private
row-major storage until consumer completion. Host edge alias checks remain
required, and portable replay preserves the whole semantic certificate.

## Exact-device validation and timing

RTX 5070, compute capability 12.0, UUID and driver are queried in every packet.
The device suite covers both storage orders, both dtypes, all three producers,
plain/fused output, complete and ragged shapes, portable replay and warmed
layout switching without recompilation. Additional direct caller-owned and
padded-resident tests prove allocation pitch normalization and pre-launch
alias refusal. 70 owning-device tests pass, including four fresh-process portable replays
with compiler lookup and packaging disabled. 676 focused host regressions
pass (66 skipped); regression-tests.txt includes the first 66 device cases,
and storage-tests.txt includes the additional four plus six repeated guards.

Four independent processes run C/F then F/C with identical operand values,
compiler and recorder sources. Each packet contains 24 correctness-gated
rows at M,K,N = 17,35,19 and 128,1024,64. Producer images are identical
across layouts; consumer images intentionally differ. Every row checks
an independent FP64 oracle before timing and checks resident output after
timing. analyze.py rejects mismatched source/compiler/device/envelopes.

Cold/public/portable wall time includes the respective host work.
Separate producer and consumer CUDA-event windows use 20 warmups and 100
native repetitions, with three samples each. These event windows include
driver dispatch gaps; they are not isolated kernel-only measurements.
analysis.json retains matched per-case ratios and current native runtime hash.
The median per-case consumer row/column ratios are 0.968235 and 0.977215
in forward/reverse process order (about 3.2% and 2.3% lower event time).
Public warm-wall ratios are 1.020660 and 0.997934; no consistent public-call
gain is established. This is layout parity evidence, not universal strategy
promotion.

## Reproduction

On Super-Bear WSL, source .build-sm120-w1-1/validation-env.sh.
Run the benchmark with TESSERA_LHS_RHS_ORDER=C or F and TESSERA_LHS_PACKET
set to the desired JSON path. Run analyze.py from the repository root.
Run tests/device/nvidia/test_lhs_tensor_jit.py and test_lhs_rhs_storage.py
through host pytest. See the captured regression log for the focused gates.

## Remaining scope

General producer/composed AD graphs, dynamic row-RHS, general A layouts,
native resident/asynchronous ownership and sibling physical consumers remain
open. FP8, MXFP8 and MXFP4 are still required before a general physical
strategy decision. The full five-slice objective remains active.
