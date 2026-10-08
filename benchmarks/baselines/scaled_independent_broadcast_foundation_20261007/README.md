# Independent scale/matrix batch semantics foundation

Owning items: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1.
Synchronization key: INDEPENDENT-SCALE-BATCH-2026-10-07.

The explicit reference-only batching="broadcast" oracle now broadcasts
matrix A, matrix B, scale A and scale B independently. Logical matrix/scale
suffixes retain their exact ragged K/N group contract. Right-aligned leading
prefixes may be shared, mapped or singleton; incompatible equal-product
prefixes are rejected. Both matrix transpose orientations are preserved.

Sixty cases cover all fifteen nonempty mapped/shared combinations and four
transpose orientations over two leading dimensions. Each compares against
scalar coordinate accumulation and checks every scale coordinate through
independent finite differences with output cotangents. Four additional cases
cover singleton and unequal-rank prefixes; one rejects incompatible prefixes.
Together with existing native-map projection tests, the initial host WSL
lane passes 98 tests. The final lane passes 417 tests without warnings, including existing oracle,
native-map, audit lifecycle, diagnostic and pass metadata regressions.
Raw receipt: final-gates.log. This is semantic conformance, not physical execution or timing.

## Native engineering dependencies

The current frontend native_vmap.batch_specs / _native_scaled_vmap accept
three matrix-coupled policies. rocm_typed_scaled_native.contract ties
scale ranks and prefixes to their matrices. Native ScaledMatmulOp::verify
does the same. PMPasses.cpp flattens paired matrix/scale storage and carries
only shared-LHS/shared-RHS/independent-RHS physical policies.
LinearTransposeInterface.cpp also derives a scale gradient's batch
reduction from the corresponding matrix's mapped state.

Production integration must carry all four independent batch index maps
through Graph verification, Schedule/Tile physical addressing and image/ABI
identity. Scale transpose must reduce each shared/singleton axis according
to the scale's own index map. Forward scale seeds must retain that same
mapping. Merely changing the frontend or verifier would not establish the
execution contract. Native admission and coverage states remain unchanged.

## Four-backend assessment

gfx1201: native typed FP8/MXFP8 primal/scale-AD integration required.
gfx1151: no inherited RDNA4 FP8 WMMA execution; separate physical route needed.
SM120: packed NVFP4 needs its own four-operand batch address and ABI proof.
Apple: Metal lowering/scale-AD execution requires independent device evidence.
x86: CPU lowering/scale-AD execution requires independent runtime evidence.

No selector promotion, kernel benchmark, exact-device claim, generic closure,
or full-suite green is inferred from this CPU numerical foundation.
