# What the cooperative EBM Langevin kernel costs against the Python-emitted lane

Owner: W4-PRODUCT-1 / AD-SOLVER-IFT-1 (sync `EBM-BIVECTOR-OVERHEAD-2026-09-16`).
The measurement the GA/EBM review asks for before the native route may
displace `rocm_ebm_langevin_compiled`. `record_ebm_langevin_overhead.py` run
on each owning device.

## What differs, and why it is the whole measurement

| | native (row-program kernel) | Python-emitted |
|---|---|---|
| gradient | derived by the compiler, evaluated **inside** the kernel | supplied by the caller, computed on the host |
| a K-step loop | **1** launch, 1 host round trip | **K** launches, K round trips |
| noise | inside the kernel | inside the kernel |

The Python-emitted kernel's signature is `(y, grad)`, so the loop cannot live
inside it. That is not an implementation detail to optimize away; it is the
route.

## Measured

Temperature 0, where both routes compute the same iterated descent, so the
comparison is like-for-like — and the recorder **checks the two agree before
it keeps any timing**. Both are dispatched at the same wrapper depth
(`runtime.launch` on an artifact), so this measures routes, not a wrapper.
Median of 9 reps after a warm-up.

| Host | Device | rows×features | K | native | Python-emitted | ratio |
|---|---|---|---|---|---|---|
| Princess-Luna | gfx1151 | 4×8 | 1 | 1.07 ms | 1.68 ms (1 launch) | 1.6× |
| Princess-Luna | gfx1151 | 4×8 | 8 | 1.08 ms | 14.26 ms (8) | 13.2× |
| Princess-Luna | gfx1151 | 16×64 | 8 | 1.05 ms | 13.88 ms (8) | 13.3× |
| Princess-Luna | gfx1151 | 16×64 | 32 | 1.12 ms | 57.97 ms (32) | 51.9× |
| Princess-Luna | gfx1151 | 64×256 | 32 | 1.17 ms | 62.06 ms (32) | 53.0× |
| Tajasarus | gfx1201 | (same five) | | 0.99–1.27 ms | **no lane** | — |
| The-Super-Bear | sm_120 | (same five) | | 1.17–1.24 ms | **no lane** | — |

**The shape of the result, not the ratio, is the finding.** The native route
is flat in K — about 1.1 ms whether the loop runs 1 step or 32, and whether
the state is 32 or 16384 elements — because the whole loop is one launch and
one round trip. The Python-emitted route costs about 1.8 ms per step because
each step is a launch and a host gradient. The ratio is therefore a *dispatch*
result: it grows with K by construction and says nothing about which kernel
computes faster.

**On two of the three devices the cooperative kernel is the only compiled
Langevin lane at all.** gfx1201 has no promoted Python-emitted EBM family
("no promoted family plugins for gfx1201; gfx1200/gfx1250 remain fail-closed
pending exact-device evidence") and sm_120 never had one. Only gfx1151 can
run the comparison.

## Not claimed

* **No promotion.** This packet moves no lane. `promotion_eligible` is false
  in every file.
* **Wall clock only** (`latency_source = "host_wall_clock"`). Neither WSL2
  ROCm box exposes `/dev/kfd`, so `rocprofv3` returns no dispatch or counter
  records: a kernel-time attribution is unavailable here, and absent counters
  classify as `unverified`, never as a measurement.
* **WSL2 timings do not promote**; bare-metal calibration is owed on the
  NVIDIA rows.
* The T > 0 rows are the native route alone. The two routes' Philox counter
  policies differ, so their samples are not comparable and are not compared.

## Reproduce on an owning host

```bash
PYTHONPATH=python:. python benchmarks/record_ebm_langevin_overhead.py --backend rocm --chip gfx1151 \
  --compiler build/tools/tessera-opt/tessera-opt --output <dir>/rocm_gfx1151.json
```
