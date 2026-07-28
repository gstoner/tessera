"""AOT (`.metallib`) vs JIT (`newLibraryWithSource:`) pipeline-creation cost.

Both lanes run the same synthesized MSL and end in the same
`dispatch_matmul_epilogue_coopmat`, so a difference is the **compile strategy**
and nothing else.

## Why a naive version of this benchmark lies

Metal keeps an on-disk shader cache that survives process exit. Measured here:
the *same* kernel compiled in one process, then again in a brand-new process,
went 140.8 ms → 0.5 ms. So "time the first launch in a fresh process" measures
whether that kernel was ever built on this machine, not what the lane costs.
It is what made an earlier run of this file report JIT first-launch times
swinging between 0.7 ms and 176 ms.

The control is a **nonce**: every sample injects a unique token into the MSL, so
the source — and therefore the cache key, for both the JIT compile and the AOT
metallib — has never been seen. Every measured compile is genuinely cold.

## What is measured

* `pipeline_ms` — obtaining a usable pipeline for a never-before-seen kernel on
  an already-initialised device, plus one dispatch. For JIT that includes
  `newLibraryWithSource:`; for AOT it includes `newLibraryWithURL:` on a
  metallib built *outside* the timer.
* `metallib_build_ms` — the AOT lane's offline `xcrun metal` + `xcrun metallib`
  cost. Excluded from `pipeline_ms` because it is build-time and cached on disk,
  but reported so the amortisation point is visible rather than hidden.

Device init (~55 ms of `MTLCreateSystemDefaultDevice` + queue + first command
buffer) is warmed before any sample; it swamps the compile and is identical for
both lanes.

Lanes are interleaved sample by sample so thermal drift hits both equally.

Run:  python3 benchmarks/apple_gpu/benchmark_aot_vs_jit.py [--samples N]
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from tessera.compiler.emit import apple_air, apple_msl
from tessera.compiler.fusion_core import FusedRegion

#: Shape for the probe kernel. Small on purpose — this measures pipeline
#: creation, not throughput, so dispatch should not dominate.
_M = _N = _K = 256
_TILE = 32
_REGION = FusedRegion(epilogue=("gelu",))


def _nonce_source(nonce: str) -> tuple[str, str]:
    """The synthesized coopmat MSL made unique, plus its entry-point name.

    The nonce renames the kernel entry point rather than adding an unused
    constant. An unused `constant` is dead code: the Metal compiler drops it and
    two "different" sources produce **byte-identical** metallibs, so the AOT
    lane would be loading the same artifact each sample and could be served by a
    content-addressed cache — measuring nothing. Renaming the entry changes the
    emitted symbol, so every sample is a genuinely distinct library while
    computing exactly the same thing.
    """
    entry = f"{apple_msl._ENTRY_COOPMAT}_{nonce}"
    body = apple_msl.synthesize_matmul_epilogue_coopmat_msl(
        _REGION, dtype="f16", tile=_TILE)
    return body.replace(apple_msl._ENTRY_COOPMAT, entry), entry


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((_M, _K)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((_K, _N)) * 0.1).astype(np.float16)
    return a, b, np.empty((_M, _N), np.float16)


def _pointer(array: np.ndarray) -> Any:
    return array.ctypes.data_as(ctypes.c_void_p)


def _time_jit(source: str, entry: str, arrays) -> tuple[float, int]:
    """Compile from MSL source and dispatch; returns (ms, ok)."""
    a, b, out = arrays
    symbol = apple_msl._synth_coopmat_symbol()
    start = time.perf_counter()
    ok = symbol(source.encode("utf-8"), entry.encode("utf-8"),
                _pointer(a), _pointer(b), None, _pointer(out),
                _M, _N, _K, 0, 2, _TILE)
    return (time.perf_counter() - start) * 1e3, int(ok)


def _time_aot(source: str, entry: str, arrays,
              cache_dir: Path) -> tuple[float, float, int]:
    """Build the metallib (untimed), then load + dispatch; (build_ms, ms, ok)."""
    a, b, out = arrays
    build_start = time.perf_counter()
    metallib = apple_air.compile_msl_to_metallib(
        source, entry=entry, cache_dir=cache_dir)
    build_ms = (time.perf_counter() - build_start) * 1e3

    symbol = apple_air._metallib_coopmat_symbol()
    start = time.perf_counter()
    ok = symbol(str(metallib).encode("utf-8"), entry.encode("utf-8"),
                _pointer(a), _pointer(b), None, _pointer(out),
                _M, _N, _K, 0, 2, _TILE)
    return build_ms, (time.perf_counter() - start) * 1e3, int(ok)


def _quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "p25": statistics.quantiles(ordered, n=4)[0] if len(ordered) > 3 else ordered[0],
        "median": statistics.median(ordered),
        "p75": statistics.quantiles(ordered, n=4)[2] if len(ordered) > 3 else ordered[-1],
        "max": ordered[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if not apple_air.metal_toolchain_available():
        print(apple_air.toolchain_hint())
        return 1

    arrays = _inputs()
    # Warm the device, the queue, and the MSL compiler service. Two throwaway
    # cold compiles: the first compile in a process is not representative of
    # the rest, and attributing that to whichever lane ran first would be a
    # pure ordering artefact.
    apple_msl.run_fused_region(FusedRegion(epilogue=("relu",)),
                               arrays[0], arrays[1])
    for _ in range(2):
        _time_jit(*_nonce_source(f"{uuid.uuid4().int % 10**9}"), arrays)

    cache_dir = Path(apple_air._CACHE_DIR) / "bench"
    jit_ms: list[float] = []
    aot_ms: list[float] = []
    build_ms: list[float] = []

    for index in range(args.samples):
        # A fresh nonce per lane per sample: no shared source means no lane can
        # warm the other's cache entry.
        jit_source, jit_entry = _nonce_source(f"{uuid.uuid4().int % 10**9}")
        aot_source, aot_entry = _nonce_source(f"{uuid.uuid4().int % 10**9}")

        # Interleave, alternating which lane goes first so any residual
        # ordering effect cancels rather than accumulating on one lane.
        if index % 2 == 0:
            ms_jit, ok_jit = _time_jit(jit_source, jit_entry, arrays)
            b_ms, ms_aot, ok_aot = _time_aot(aot_source, aot_entry, arrays, cache_dir)
        else:
            b_ms, ms_aot, ok_aot = _time_aot(aot_source, aot_entry, arrays, cache_dir)
            ms_jit, ok_jit = _time_jit(jit_source, jit_entry, arrays)

        if not ok_jit or not ok_aot:
            print(f"sample {index}: a lane declined the device "
                  f"(jit={ok_jit}, aot={ok_aot}) — aborting")
            return 1
        jit_ms.append(ms_jit)
        aot_ms.append(ms_aot)
        build_ms.append(b_ms)

    jit_q, aot_q, build_q = _quantiles(jit_ms), _quantiles(aot_ms), _quantiles(build_ms)
    print(f"cold pipeline creation + dispatch, {_M}x{_N}x{_K} f16 coopmat, "
          f"n={args.samples}, unique kernel per sample\n")
    header = f"{'lane':<26} {'min':>8} {'p25':>8} {'median':>8} {'p75':>8} {'max':>8}"
    print(header)
    for label, q in (("JIT newLibraryWithSource:", jit_q),
                     ("AOT newLibraryWithURL:", aot_q),
                     ("AOT offline build (excl.)", build_q)):
        print(f"{label:<26} {q['min']:>8.1f} {q['p25']:>8.1f} {q['median']:>8.1f} "
              f"{q['p75']:>8.1f} {q['max']:>8.1f}")

    saved = jit_q["median"] - aot_q["median"]
    print(f"\nmedian pipeline creation: AOT is {saved:+.1f} ms vs JIT "
          f"({jit_q['median']:.1f} → {aot_q['median']:.1f})")
    if saved > 0:
        print(f"offline build costs {build_q['median']:.1f} ms and is paid once "
              f"per kernel per machine — it repays after "
              f"{build_q['median'] / saved:.1f} cold launches.")
    else:
        print("AOT does not reduce pipeline-creation cost at this shape.")

    if args.json:
        args.json.write_text(json.dumps(
            {"shape": [_M, _N, _K], "dtype": "f16", "samples": args.samples,
             "jit_pipeline_ms": jit_q, "aot_pipeline_ms": aot_q,
             "aot_offline_build_ms": build_q}, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
