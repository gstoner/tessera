# Exact MXFP4 residual gate — gfx1201

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Synchronization key MXFP4-RESIDUAL-GATE-20261014.

[Packet](gfx1201.json) queries the active RX9070XT/gfx1201 device, records HIP
identity and source/binary hashes, and rejects another architecture.

Named route: Python capture → typed Graph scaled product → native MLIR
Schedule/Tile → gfx1201 FP8 WMMA with packed E2M1 decode and mixed row-F32/
column-E8 K32 scales → BF16 product store → exact F32 widening → F32 gate.
Only the gate is differentiated. This preserves the physical BF16 rounding
boundary; row-scale and encoded-storage AD remain outside the contract.

Independent literal E2M1/E4M3/E8 decoding and F64 block-dot oracle precede
timing. Exact MXFP4 oracle error is zero for these representable test values.
FP8 and MXFP8 controls use the existing independent oracle and tolerance.
Every timed domain checks outputs again, and warm compiler subprocesses are
forbidden. These are correctness/latency characterization rows, not a
matched-format speedup comparison: the control shapes and AD roles differ.

## Timing domains

Captured member events repeat each pure SSA member 2048 times. They exclude
host preparation/copy and are diagnostic grouped windows. Program events
repeat 128 complete invocations, including host dispatch gaps, and exclude
preparation/update/readback. Completed public primal/JVP/reverse calls include
binding, preparation, upload, launch, readback and independent output ownership.
Each cell is a median of seven samples; units are explicit.

| M×K×N | Product µs | Cast µs | Gate µs | Primal program event µs | Public primal ms | Public JVP ms | Public reverse ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17×64×19 | 19.416 | 2.024 | 2.006 | 24.295 | 1.808 | 1.165 | 1.558 |
| 256×128×80 | 16.644 | 1.594 | 1.773 | 20.847 | 3.306 | 4.111 | 3.813 |
| 256×128×160 | 16.549 | 1.615 | 1.826 | 20.744 | 3.606 | 1.371 | 1.666 |

For this K=128 exact residual route, doubling N from 80 to 160 at M=256 does
not increase the measured product-member window (about 16.6 µs). This is
evidence about this schedule/envelope, not an explanation of the original
Radiance gap. Public N=80 JVP/reverse latency remains higher than N=160;
those host-inclusive values must not be called per-column kernel cost.
Further host-launch/allocation attribution is required.

General packed layouts, dynamic image identity, scale derivatives through
rounding, model-quality acceptance and default schedule selection remain open.
No Apple, NVIDIA, gfx1151 or x86 execution evidence is transferred.


## Native transport attribution

[Transport ABBA packet](transport_abba.json) forces the existing pageable /
pinned / pinned / pageable overrides on identical M=256/K=128 inputs.
Twenty completed calls per domain retain exact oracle checks and forbid warm
compiler subprocesses. The recorder queries the active GPU identity and pins
the native provider/source. Profiling located the default N=80 cost in native
upload/readback. The current automatic policy requires at least 256 KiB of
inputs; N=80 paired AD falls below it, while N=160 exceeds it.

N=80 public JVP is 4.850–5.124 ms pageable versus 1.322–1.485 ms pinned;
N=160 is 4.880–4.946 ms versus 1.304–1.321 ms. This explains the host-inclusive
N80/N160 discontinuity in this residual route. It does not explain Radiance
kernel cost. The default transport policy is unchanged; measuring a bounded
traffic-based threshold across FP8/MXFP8/MXFP4 remains the next engineering
step. Reproduce with:

```sh
python benchmarks/rocm/record_mxfp4_host_transport.py --output transport_abba.json
```
