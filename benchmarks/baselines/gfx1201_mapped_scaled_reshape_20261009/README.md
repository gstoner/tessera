# gfx1201 mapped scaled reshape integration

Synchronization key: MAPPED-SCALED-RESHAPE-INTEGRATION-20261009.
Owners: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6 / LAYOUT-ALG-1.

## Executed contract

An annotated textual FP32 frontend builds a scaled product (3x7 by 7x4),
reshapes its 3x4 result to 6x2, then consumes that value in a second
scaled product (6x2 by 2x5). Native vmap projection preserves each SSA
value's actual leading map prefix, including shared intermediate values.
The shape attribute, dtype and element count are validated before projection.
Production arithmetic and differentiation remain in native MLIR passes,
Schedule/Tile/ROCm Target lowering, LLVM HSACO and checked HIP programs.
Python contains only Graph metadata and an explicitly cold numerical oracle.

## Evidence

- 434 WSL compiler/projection/reshape tests pass, including all 127 mapped-root
  masks in primal/JVP/VJP and four invalid authored-shape regressions.
- 60 RX9070XT/gfx1201 cases pass: five root masks, one/two leading maps,
  first/last result axis, primal/JVP/VJP, all seven roots, changed inputs/seeds,
  compiler/eager-forbidden warm replay and retained output lifetimes.
- 326 focused registry tests pass; Ruff and mypy ratchet pass with zero errors.
- Two fresh-process packets contain 15 programs each, seven windows and 128
  native repetitions per window. Independent float64 per-plane block-scale
  oracles validate results before native timing and after each program,
  member and public window. Maximum absolute error is 2.33e-10.
- Source digests match the recorder/test/frontend files. Transferred compiler
  SHA256: 39d59eee080a6b0d44edaefae009a0947982a7e1c28aeb44f68d897c9011aa56.
  LLVM/MLIR 23.1.1 with assertions; device UUID and runtime/tool hashes are
  recorded in each packet.

## Timing domains

| Mode | Native program median range (ms) | Compilation-warm public median range (ms) |
| --- | ---: | ---: |
| primal | 0.008587–0.010832 | 2.030–2.280 |
| forward | 0.049498–0.051262 | 4.474–4.989 |
| reverse | 0.036388–0.052924 | 2.640–2.888 |

Native program events include the ordered HIP graph members. Grouped member
profiles are dispatch-inclusive diagnostics and cannot be added to estimate
interleaved program time. Public calls include preparation/allocation.
These tiny ragged tests establish correctness and attribution, not a throughput
gain or a promoted quantized schedule.

## Remaining scope

Unannotated generic vmap, arbitrary composition/dynamic shapes/aliases and
generic scaled-matmul batching/transpose closure remain open. FP8, MXFP8,
MXFP4 and NVFP4 schedules are unchanged. No gfx1151, CUDA, Metal or x86
device parity follows from this gfx1201 packet.

Recorder: benchmarks/rocm/record_mapped_scaled_reshape.py.
Packets: run1.json and run2.json.
