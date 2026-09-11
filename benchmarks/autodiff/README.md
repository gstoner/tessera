# Autodiff benchmark entry points

See the [compiler alignment review](../COMPILER_ALIGNMENT.md#autodiff-entry-points)
for all eight scripts, their compiler boundaries and missing proof. The solver
and hand-built Graph IR probes remain useful compiler regressions; they do not
stand for general automatic frontend AD.

Recent public resident GPU VJP, asynchronous ownership and checkpoint workloads
are indexed there alongside the existing scripts. Run owning-device probes on
the named target, record exact artifacts, and keep host-wall and device-clock
measurements separate. Historical baseline eligibility is not inherited by new
runs. Performance promotion requires clean bare-metal comparative evidence.

### Public resident SSD comparisons

`benchmark_public_ssd.py --backend nvidia --compiler /path/to/tessera-opt
--output /tmp/public-ssd.json` also supports ROCm. It exercises public
`tessera.control.vjp` over three SSD shapes, including a partial checkpoint
chunk, two cotangents and all five differentiated inputs. An independent
float64 NumPy recurrence and central differences check native forward and
reverse outputs. This synchronous family-protocol comparison complements the
asynchronous ownership recorder; it does not prove arbitrary frontend AD,
checkpoint cotangents or performance promotion.
