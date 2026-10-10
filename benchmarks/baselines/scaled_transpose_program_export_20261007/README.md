# Native scale-adjoint program export

Owner: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Synchronization key: SCALED-TRANSPOSE-PROGRAM-2026-10-07.

Paired AD exports the actual generated tensor/SCF regions. It derives captured
tensor inputs in original frame argument order, preserves frontend argument
permutations, records requested gradient output order, and computes byte
storage and first-write/last-read lifetimes from actual SSA uses. Single-scale,
both-scale and reordered requests are supported. Unreturned matrix-storage
zero gradients are excluded from program members; the complete backward
Graph stays in the immutable witness.

Every member is cloned with IRMapping onto its captured tensor arguments.
Member projection retains the full program witness and does not reconstruct
numerical operations. Native export requires a fresh pairing and rejects
unrequested/duplicate derivative roles and captured values outside the frame.

Evidence:
- Final-source registry, export, dtype/op and audit drift gates: 368 passed;
  compiler-plan ownership/link and Ruff gates pass.
- Final native export plus existing immutable package checks: 55 passed,
  including 31 new capture/order/lifetime/negative cases.
- Focused export/diagnostic/pass/op/dtype checks before the final permutation
  increment: 356 passed. Final-source drift gates are recorded separately.
- Native AD fixture lane: all 75 PASS.
- Existing owning RTX 5070 NVFP4 leading-map regressions: 11 passed,
  46 deselected; CUDA 13.3.73. This establishes existing-route regression,
  not NVIDIA scale-adjoint execution.
- Matching final compiler build passes. Initial header/include and
  witness-string fixture failures are preserved.

The scale_vjp manifest is a native program contract. Reduction member
Schedule/Tile lowering, image/ABI packaging, HIP ownership integration,
public reverse AD, owning gfx1201 gradient numerics and separate kernel,
program and public timings remain open. No reverse execution or generic
closure status is promoted, and no reverse performance claim is made.
