---
last_updated: 2026-10-16
---

# Normalization AD accumulator contract

Owner: AD-FWD-CORE-1 / AD-HIGHER-1 / NUMPOL-CARRIER-1.
Synchronization key: NORM-AD-ACCUM-20261015.

## Semantics

RMSNorm and LayerNorm JVP/VJP preserve an explicitly requested, realizable
wider floating-point accumulator. Native TangentInterface/AdjointInterface
rules use the same accumulator decoder as generic Graph-to-linalg lowering.
The differentiated Graph casts source, affine operands and derivative seeds
to the selected working element type before statistics, projections and
reductions. Returned tangents and cotangents are cast to each original
argument's storage element type only after its derivative arithmetic.
Primal computation retains the original normalization operation and policy.

For centered normalization, z = (x - mean(x)) * inverse_std. For RMSNorm,
z = x * inverse_rms. JVP computes inverse * (dx - z * mean(dx*z)),
also subtracting mean(dx) for centered normalization. The result is
multiplied by gamma when present; affine tangents add z*dgamma and dbeta.
VJP first projects the incoming cotangent through gamma, then applies
the corresponding centered or RMS projection; affine gradients reduce
the physical incoming cotangent times z, or the cotangent alone, across
all leading axes. This defines the smooth derivative before the final
storage conversion; no derivative of the quantization step is introduced.

For an explicitly wider RMSNorm accumulator, the native derivative now uses
an algebraically equivalent weighted-residual form to retain epsilon in radial
directions. With r = inverse_rms, w = (x*r)*r, a = mean(x*w) and
b = mean(direction*w), it computes
r * ((direction*a - x*b) + direction*((epsilon*r)*r)).
Multiplication order avoids forming r*r, which can overflow even while r
is finite for small positive epsilon. The tiny-epsilon tests require finite
zero-seed derivatives and compare nonzero radial derivatives to an independent
FP64 oracle without changing the encoding bound.
The same helper serves JVP and the gamma-projected VJP direction. For a seed
identical to x, the two residual products cancel and the epsilon term survives,
rather than being lost by subtracting nearly equal large projection terms.
All operations remain in the requested working type; this is no implicit
FP64 promotion. Statistics, primal output, affine derivative terms and final
storage conversion retain their contracts.

The new finite scalar epsilon and addition travel through actual typed Graph
SSA into the checked SM120 row outline, Schedule, scalar Tile body and native
lowering. Body seals still bind the concrete operation graph. The numerical
and architecture validation below must pass before publication; a prototype
or a changed expression is not a precision certificate.

An unspecified accumulator retains existing storage-typed Graph arithmetic.
A BF16 storage / FP16 accumulator request is unrealizable, despite equal
bit width, and must not be silently substituted. Other policy fields are
not newly certified by this change.

## Shapes, lineage and native lowering

Direct rank-two forward normalization admits a dynamic leading extent with
a fixed positive channel extent and matching unencoded floating storage.
Every normalization operand must correspond bijectively to a distinct root
argument. Argument order may differ from normalization operand order;
requested derivative roles remain tied to the original function arguments.
This admission does not extend compound dynamic producer graphs or unknown
channel dimensions.

Both derivative rules assert equality of dynamic source/seed dimensions
before reading seed elements. The assertion message is
"normalization derivative seed shape must match primal".
Static verifier checks remain applicable. Native LLVM assertion tests require
the intended assertion message and SIGABRT; an unrelated crash is a failure.

Unencoded shape-preserving floating tessera.cast operations without an
explicit cast numeric policy lower through linalg to native floating
extension/truncation. Distinct equal-width formats and casts with independent
policy are not admitted by this bridge.

## Provenance and backend assessment

The generic linalg boundary may still conservatively report
not_yet_carried:NUMPOL-CARRIER-1 when reverse-mode dead-code elimination removes
the original normalization before the policy carrier sees it. Working types
and native numerical results prove accumulator arithmetic for this envelope;
the conservative metadata debt remains open. No blanket function marker may
declare unrelated policies consumed.

- NVIDIA: follow-up required for active-source normalization/ReLU derivative
  SSA through a genuine multi-result Schedule/Tile program and checked CUDA
  owner. Existing v7 RHS-only paired packages and their schemas are unchanged.
- ROCm gfx1201/gfx1151: shared native AD semantics apply; physical derivative
  producer packages require their own lowering and exact-device certificates.
- Apple: shared native AD semantics apply; no new Metal or Apple CPU package
  execution envelope is implied by generic linalg/LLVM proof.
- x86: native LLVM object execution provides numerical proof for these
  generated functions. Public JIT package integration is a separate gate.

## Validation requirements

Execute the generated native code against an independent FP64 oracle for
FP16/BF16 storage with FP32 accumulation, and FP32 storage with FP64
accumulation. Cover RMSNorm/LayerNorm, JVP/VJP, affine operands, dynamic
leading extents and permuted arguments. Include large-magnitude overflow,
mixed signs and large-offset cancellation. Nontrivial low-precision cases
permit at most one adjacent output-storage encoding from the independently
rounded oracle; structured precision-loss regressions retain tighter bounds.

Run registry/diagnostic gates, existing AD/numeric-policy regressions, and
positive/negative dynamic seed checks. The Python IR interpreter must execute
scalar dimension comparisons and cf.assert, not discard them. Exact native
output bits are transported through an integer-printing C driver; the CPU
compiler proof does not establish GPU correctness or performance.

## 2026-10-16 tiny-epsilon numerical gate

The shared RMS JVP/VJP repair multiplies x by each inverse factor before
the reductions, and scales epsilon by each inverse factor before applying the
direction. It avoids materializing an overflowing inverse square when inverse
is finite. Thirty-two native tiny-epsilon regressions reproduce NaNs before the
repair; all 153 native LLVM precision cases now pass, including zero seeds in
FP16/BF16->FP32 and FP32->FP64, affine/permuted/static/dynamic cases and nonzero
BF16 radial inputs. Twenty RTX5070 cases cover nonzero tiny-epsilon BF16
through public host/resident, portable and event/captured profile routes.

Frozen 46a5dd8da721055a51b672adde1eb6e534789c3b passes 26,889 non-slow unit passes (7,678 skips),
912 required compiler-route passes and 416 owning RTX5070 passes, with no skips
in either required proof lane. All phase source/tool/helper witnesses match.
Eight timed packages also pass independent FP64 checks and standalone replay
with compiler subprocesses disabled. The latest same-owner resident comparison
shows about 19–28% lower public latency at the larger synthetic M256 cases,
and 40–57% higher latency at smaller cases. No default, quantized-route or
sibling physical promotion follows. The earlier d1ba6e98d and e164fd4d8
records retain their original source scope; they do not validate this changed
arithmetic. The five-slice aggregate and wider integration gates remain open.
