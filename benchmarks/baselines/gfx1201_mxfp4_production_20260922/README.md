# gfx1201 MXFP4 production-schedule evidence

This packet records the versioned fragment-layout decode ABI and the safe
decode/prefill production selector for
`ROCM-MXFP4-W4A8-1` on Tajasarus, an AMD Radeon RX 9070 XT (`gfx1201`). It is
bound to the Tessera source revision recorded in `evidence.json` and to
the generator and benchmark hashes in `evidence.json`.

## Method

- Inputs are the same logical E4M3 activations, packed E2M1 weights, E8M0 K32
  scale plane, and per-token FP32 scale for every engine.
- E8M0 exponent deltas are at most two, so the independently implemented
  row-reference fold is exact for this comparison.
- Every engine must produce the same BF16 bits before it is timed.
- A sampled independent FP32-dequantized reference gates each engine before
  cross-engine agreement; agreement with Radiance is not used as the oracle.
- Checkpoint weights are converted once before launch. Decode consumes the
  versioned gfx12 fragment ABI; prefill retains the proved transposed ABI.
- Three operand copies rotate between launches; production weight footprints
  exceed the device last-level cache.
- Timings are HIP events around 12 launches, with six warmups and nine samples.
- Effective weight bandwidth counts packed weights plus their E8M0 scale plane.

Radiance was independently built from
`ggz14/radiance-vllm-mxfp4@dfdfa3832922c9a4253133f09c1f5c0d39748fc7`.
libr4d was independently built from the canonical
`StillDeadcode/libr4d@5dc6302b87d598d1d3bf2ad3b50aab365461a63c` and is
reported only inside its `M <= 64` contract. No source was copied from either
tree; neither inspected root exposed a license or SPDX declaration.

## Results

| Workload (M x N x K) | Tessera schedule | Tessera ms | Radiance ms | libr4d ms |
|---|---:|---:|---:|---:|
| decode 8 x 5120 x 8704 | split-K 8 + fragment | 0.050867 | 0.029875 | 0.039361 |
| decode 8 x 17408 x 5120 | split-K 8 + fragment | 0.090106 | 0.086794 | 0.105388 |
| prefill 256 x 5120 x 8704 | group-M 8 + transposed | 0.641708 | 0.138573 | not applicable |
| prefill 1024 x 17408 x 5120 | group-M 8 + transposed | 4.843409 | 0.888253 | not applicable |

The expanded owning-device fixture passes 16 rows: scalar/WMMA/generic routes,
M=1/5/64, N=48/80, the M64/65 crossover, long-K and wide-N cases, poisoned
rows, and a captured HIP graph. Fragment decode is about 1.73–1.74x faster than
the prior packet; the second production shape is within 1.04x of Radiance and
faster than libr4d. An attempted fragment+LDS prefill reuse path failed the
wide-N exact oracle and was rejected. Correct direct fragment prefill regressed,
so the selector keeps prefill on the proved transposed/group-M ABI.

Remaining work is padded multistage prefill, a backend-neutral K-step scheduling
barrier, cache-modifier and waves-per-EU tuning, then resource/ISA evidence.
