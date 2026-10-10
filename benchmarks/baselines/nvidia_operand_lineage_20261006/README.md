# SM120 canonical tensor input lineage

Owner W1.1; sync NVIDIA-W11-OPERAND-LINEAGE-2026-10-06.
Sibling frontend integration owner FRONTEND-IR-MEDIUM-1.

## Compiler integration

The canonical tensor-reduction recovery pass now traces LHS/RHS through
tensor.extract_slice, tensor.pad, tensor.insert_slice and tensor.cast to their
actual distinct entry arguments. It reconstructs the Graph contraction using
those roles and orders bias/residual operands by the op contract independently
of frontend argument order. Full-function equivalence against fresh MLIR
tiling still proves padding, slices, accumulator, loops, return and epilogue
before replacing the live body. Input lineage alone never authorizes recovery.

Native Schedule/Tile retains compiler-owned tile.view and typed
pack/zero/MMA/unpack/store. The generated NVIDIA kernel owns the canonical
A/B/bias/residual pointer ABI. Package validation projects types from the
verified retained Graph operand bindings, rather than assuming frontend
arguments 0/1 mean A/B. Python performs contract checks and package binding;
it constructs no shader, Tile arithmetic or recovery Graph.

## Exact-device proof

rtx5070.json queries RTX 5070 UUID, driver 610.88 and compute capability 12.0.
Twelve FP16/BF16 plain and bias/ReLU/residual rows cover (M,K,N)
16x32x16, ragged 17x35x19 and 64x256x64.
Plain inputs use frontend order [B,A]. Fused inputs use [residual,B,A,bias].
Six additional device tests also cover [bias,residual,B,A].
All host/resident launches require native_gpu and numerical agreement before
and after timing. Maximum benchmark absolute error is 1.55e-6; the additional
device tests use independent FP64 references.

Every row proves full registered pipeline executable-function/ABI parity and
byte-identical PTX against the executed checked native Schedule package.
Source, compiler, Tensor/Tile/Target/image hashes and frontend argument orders
are retained. The historical source hashes refer to the recorder run, before
the final shifted-padding negative test was added.

Five resident CUDA-event windows use 200 native C++ launches after 20 warmups.
Event medians range 8.91–10.29 µs and include driver dispatch gaps.
Host package launch medians range 0.306–0.447 ms and include transfers/lifecycle.
Compiler stages are recorded separately. These are scope-specific timings;
this packet makes no kernel speedup or Python-overhead reduction claim.

## Validation and remaining scope

80 initial producer tests passed; the final 373 producer/diagnostic/pass-metadata
regression tests include the added shifted-padding refusal. Existing tampered
accumulator, bounds, return arithmetic, target and epilogue checks remain.
The full five-slice goal remains open: general producer/composed attention/AD
graphs, dynamic reconstruction, noncanonical accumulators and wider layout/
dtype envelopes still need native consumers and exact-device proof.
SM<120 producer construction is unchanged. ROCm/Apple/x86 do not execute this
SM120 recovery or use its NVIDIA-only role projection; no sibling physical
schedule or device evidence transfers.


Final frontend-order totality: all 24 four-argument permutations and both
two-argument permutations pass native recovery/direct-Tile parity for FP16
and BF16. The expanded producer/diagnostic/pass-metadata run passes 419 checks
(totality-tests.txt). This proves this input-role envelope, not arbitrary
producer graphs or composed attention AD.

## Ordinary public frontend execution

The static SM120 host-array matmul call now compiles and launches the canonical
Graph -> Schedule -> Tile -> LLVM/NVVM image through the checked descriptor.
Python declares actual compact RHS storage using rhs_storage_order; the native
Schedule selects and verifies tile.view storage and typed-fragment transpose.
The native static envelope admits fused epilogues and final f16 output with
separate row-major RHS ABI variants. F32 accumulation is retained through the
epilogue and rounded only at output.

Twelve owning-device tests cover permuted FP16/BF16 inputs, C/F RHS, plain and
bias/ReLU/residual calls, f32/f16 output, serialized portable replay, changed
input values, and cached execution with eager fallback/recompilation forbidden.
The affected native/JIT/registry regression lane passes 591 tests, with 53
explicitly skipped tests. These skips are not device evidence.

Python still performs trace/binding/allocation work per public call. This slice
establishes canonical execution; it does not claim to eliminate that overhead.
public_jit.json measures cold public wall, warm public wall, and resident
CUDA-event windows separately. General composed Graph/AD, dynamic storage and
other backends require their own consumers and exact-device validation.

The twelve 17x35x19 rows record cold public calls at 372–698 ms,
warm public calls at 0.663–0.866 ms, and resident CUDA-event window medians
at 8.71–10.19 microseconds per dispatch. These scopes must not be subtracted
as pure Python overhead: the warm host route includes transfer/allocation
and synchronization that the resident window excludes. No baseline speedup
claim is made. Maximum fp64-oracle absolute error is 1.24e-7 for f32 rows;
f16 rows match the rounded oracle exactly for these inputs.

## Subsequent prepared owner

The repeat-binding overhead above is addressed for this static host envelope
by [native prepared matmul ownership](../nvidia_prepared_matmul_20261006/README.md).
Its later source/binary fingerprints and matched A/B evidence remain separate
from the historical public_jit.json packet here. General composed/dynamic/AD
and physical A layout integration remain open.
