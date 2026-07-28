"""Measure the AOT (`.metallib`) lane against the JIT (`newLibraryWithSource:`) lane.

Both lanes end in the same `dispatch_matmul_epilogue_coopmat` in the runtime and
run the same synthesized MSL, so any difference is the **compile strategy**, not
the kernel and not the dispatch.

Two numbers matter and they answer different questions:

* **cold** — first launch in a fresh process. JIT pays `newLibraryWithSource:`
  (front-end compile); AOT pays `newLibraryWithURL:` (load a prebuilt blob).
  This is what a short-lived process or a first inference request sees.
* **warm** — steady state once the pipeline is in `ctx.kernel_cache`. Both are a
  map lookup, so they should converge; a gap here would mean the AOT path is
  doing something the JIT path is not, and is a bug rather than a result.

Cold has to be measured in a *fresh process* — the runtime's pipeline cache is
process-global, so a second call in the same process is warm by construction.
The AOT lane's offline `xcrun metal`/`metallib` cost is deliberately excluded:
it is build-time, paid once per machine, and cached on disk.

Run:  python3 benchmarks/apple_gpu/benchmark_aot_vs_jit.py
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO / "python"

#: One measurement in a fresh interpreter: import, build inputs, time the first
#: launch (cold), then time `warm_reps` further launches.
_CHILD = r'''
import json, sys, time
sys.path.insert(0, {python_root!r})
import numpy as np
from tessera.compiler.fusion_core import FusedRegion
from tessera.compiler.emit import apple_msl, apple_air

lane, M, N, K, warm_reps = {lane!r}, {m}, {n}, {k}, {warm_reps}
region = FusedRegion(epilogue=("gelu",))
rng = np.random.default_rng(0)
A = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
B = (rng.standard_normal((K, N)) * 0.1).astype(np.float16)

if lane == "aot":
    # Pre-build the metallib so the timed call measures load+dispatch, not the
    # offline toolchain run (which is build-time and cached on disk).
    src = apple_air.get_emitter(apple_air.AIR_TARGET).emit(
        region, dtype="f16", dims=(M, N, K), variant=apple_msl.COOPMAT)
    apple_air.compile_msl_to_metallib(src.source, entry=src.entry)
    run = lambda: apple_air.run_fused_region_aot(region, A, B)
else:
    run = lambda: apple_msl.run_fused_region(region, A, B)

t0 = time.perf_counter()
out, execution = run()
cold_ms = (time.perf_counter() - t0) * 1e3

warm = []
for _ in range(warm_reps):
    t = time.perf_counter()
    run()
    warm.append((time.perf_counter() - t) * 1e3)

print(json.dumps({{"lane": lane, "execution": execution, "cold_ms": cold_ms,
                   "warm_ms": warm}}))
'''


def measure(lane: str, m: int, n: int, k: int, *, warm_reps: int = 20) -> dict:
    """Run one lane in a fresh process so `cold` is genuinely cold."""
    script = _CHILD.format(python_root=str(PYTHON_ROOT), lane=lane, m=m, n=n,
                           k=k, warm_reps=warm_reps)
    result = subprocess.run([sys.executable, "-c", script],
                            capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"{lane} measurement failed:\n{result.stderr.strip()}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> int:
    shapes = [(128, 128, 128), (512, 512, 512), (1024, 1024, 1024)]
    print(f"{'shape':>16}  {'lane':<5} {'execution':<16} "
          f"{'cold ms':>9} {'warm median ms':>15}")
    rows = []
    for m, n, k in shapes:
        for lane in ("jit", "aot"):
            record = measure(lane, m, n, k)
            warm_median = statistics.median(record["warm_ms"])
            rows.append({**record, "m": m, "n": n, "k": k,
                         "warm_median_ms": warm_median})
            print(f"{f'{m}x{n}x{k}':>16}  {lane:<5} {record['execution']:<16} "
                  f"{record['cold_ms']:>9.2f} {warm_median:>15.3f}")

    print()
    for m, n, k in shapes:
        jit = next(r for r in rows if r["lane"] == "jit" and r["m"] == m)
        aot = next(r for r in rows if r["lane"] == "aot" and r["m"] == m)
        if jit["execution"] == "reference" or aot["execution"] == "reference":
            print(f"{m}x{n}x{k}: a lane fell back to reference — no comparison")
            continue
        saved = jit["cold_ms"] - aot["cold_ms"]
        print(f"{m}x{n}x{k}: cold {saved:+.2f} ms for AOT "
              f"({jit['cold_ms']:.2f} → {aot['cold_ms']:.2f}); "
              f"warm {jit['warm_median_ms']:.3f} vs {aot['warm_median_ms']:.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
