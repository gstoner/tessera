# Native LSE-cotangent attention consumer

Owner NVIDIA-LSE-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync NVIDIA-LSE-COTANGENT-LEAF-2026-10-07. Native physical consumer proved;
public Graph AD and package integration are not yet complete.

The existing Tile attention backward operation has an optional boolean
lse_cotangent attribute. Enabled use requires saved f32 O/LSE, deterministic
direct execution and one [B,Hq,Sq] f32 pointer immediately after row_lse.
The native SM120 materializer adjusts delta to dot(dO,O)-dLSE, yielding
P*(dO.V-dot(dO,O)+dLSE) for Q/K/bias. V gradients are unchanged.
False/absent use preserves the old operand layout and arithmetic.

The shared Tile verifier checks the boolean, saved storage/policy and extra
pointer count. ROCm explicitly rejects the new physical consumer until an
owning implementation exists. Apple/x86 have no admitted consumer for this
launch operation. No new public operation/dtype or packaged runtime ABI is
claimed. Graph checkpoint operands, Schedule projection, checked descriptor
and public multi-result AD seeds still require integration.

Numerical fixture: tests/device/nvidia/test_lse_cotangent_native.py.
Three shapes include GQA, batching and both rectangular directions; full and
causal attention; plain and exact bias; zero LSE seed, zero output seed and
mixed signed nonuniform seeds. Independent FP64 score derivatives check all
Q/K/V and bias outputs. This handwritten Tile fixture is diagnostic proof of
the physical leaf, not a Python production constructor or end-to-end proof.

Matching LLVM/MLIR 23.1.1 tessera-opt and tessera-nvidia-opt build passes.
RTX 5070 native numerical/contract lane: 41 passed (36 numerical cases, five
refusals). record_lse_cotangent_leaf.py produces rtx5070.json: 36 rows and
180 poisoned CUDA-event windows pass independent FP64 checks. Timings include
Python diagnostic driver gaps; they are feature characterization, not a default
selector promotion or end-to-end speedup. The concurrent generated-document process was suspended for the final
measurements and resumed afterward. Zero-seed enabled/legacy event ratios
range from 0.977 to 1.070; driver gaps and microsecond timing variation prevent
a hardware-only overhead claim. Full Graph-to-package tests and package-level
performance comparisons remain required.

Regression gates: 384 pass for saved checkpoint replay, native attention schedules, diagnostic/pass metadata, operator registry and tensor dtype audit. Documentation gates: 15 pass. Ruff is clean. These counts overlap earlier lanes.
