#!/usr/bin/env python3
"""End-to-end timing of the ROCm spectral STFT/ISTFT lanes on an exact chip.

The ROCm sibling of ``benchmark_nvidia_spectral.py`` for the STFT/ISTFT
family: forward calls and the native JVP/VJP packages at one audio-sized
shape, through the public ``@tessera.jit`` entry points a caller uses, so a
latency includes host staging, allocation and dispatch as well as kernels.
Forward outputs are checked against NumPy before timing; the reverse and
tangent packages are covered for correctness by
``tests/unit/test_autodiff_spectral_target_binding.py`` and only timed here.

Set ``TESSERA_ROCM_CHIP`` to the chip under test (gfx1151 or gfx1201) and, on
gfx1201, ``TESSERA_GFX1201_DEVICE_PROOF=1``; every row records the chip and
the composite image's architecture so a result can never be read as the other
chip's (proofs do not transfer between them).

Rows carry the Decision #12 fields plus ``route`` and ``latency_source``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from typing import Any, Callable

import numpy as np

import tessera

NFFT, HOP, BATCH, SAMPLES = 512, 128, 8, 16000
FRAMES = (SAMPLES - NFFT) // HOP + 1
LENGTH = (FRAMES - 1) * HOP + NFFT


# Literals, not closure variables: the tracer binds a free variable as a
# graph value, and n_fft/hop/length must be attributes.
@tessera.jit(target="rocm")
def stft_forward(x, window):
    return tessera.ops.stft(x, window, axis=-1, n_fft=512, hop=128,
                            center=False, onesided=True, norm="backward")


@tessera.jit(target="rocm")
def istft_forward(spectrum, window):
    return tessera.ops.istft(spectrum, window, axis=-1, n_fft=512, hop=128,
                             center=False, onesided=True, length=16000,
                             norm="backward")


@tessera.jit(target="rocm", autodiff="jvp", wrt=("x", "window"))
def stft_jvp(x, window):
    return tessera.ops.stft(x, window, axis=-1, n_fft=512, hop=128,
                            center=False, onesided=True, norm="backward")


@tessera.jit(target="rocm", autodiff="reverse", wrt=("x", "window"))
def stft_vjp(x, window):
    return tessera.ops.stft(x, window, axis=-1, n_fft=512, hop=128,
                            center=False, onesided=True, norm="backward")


@tessera.jit(target="rocm", autodiff="jvp", wrt=("spectrum", "window"))
def istft_jvp(spectrum, window):
    return tessera.ops.istft(spectrum, window, axis=-1, n_fft=512, hop=128,
                             center=False, onesided=True, length=16000,
                             norm="backward")


@tessera.jit(target="rocm", autodiff="reverse", wrt=("spectrum", "window"))
def istft_vjp(spectrum, window):
    return tessera.ops.istft(spectrum, window, axis=-1, n_fft=512, hop=128,
                             center=False, onesided=True, length=16000,
                             norm="backward")


def _stft_reference(x, window):
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(FRAMES)[:, None]
    return np.fft.rfft(x[:, idx] * window, axis=-1)


def _istft_reference(spectrum, window):
    frames = np.fft.irfft(spectrum, n=NFFT, axis=-1) * window
    out = np.zeros((spectrum.shape[0], LENGTH))
    weight = np.zeros(LENGTH)
    for frame in range(FRAMES):
        out[:, frame * HOP:frame * HOP + NFFT] += frames[:, frame]
        weight[frame * HOP:frame * HOP + NFFT] += window.astype(np.float64) ** 2
    return out / np.maximum(weight, 1e-12)


def _first_call(call: Callable[[], Any]) -> tuple[Any, float]:
    """The case's first invocation, timed: ``cold_ms`` is this call. It
    includes compilation, package images and plans that no earlier case in the
    process already created; the result is also what correctness is checked
    against, so no untimed call warms the route first."""
    start = time.perf_counter_ns()
    result = call()
    return result, (time.perf_counter_ns() - start) * 1e-6


def _time(call: Callable[[], Any], warmup: int, repeats: int) -> list[float]:
    """Warm samples only; the cold call is measured by ``_first_call``."""
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        call()
        samples.append((time.perf_counter_ns() - start) * 1e-6)
    return samples


def _image_arch() -> str | None:
    try:
        from tessera.compiler.emit import spectral_candidates
        lib = spectral_candidates._amd_composite_lib()
        return lib.ts_spectral_composite_arch_amd().decode() if lib is not None else None
    except Exception:  # the row still records the requested chip
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--only", default="", help="comma-separated case-name prefixes")
    parser.add_argument("--output")
    args = parser.parse_args()

    chip = os.environ.get("TESSERA_ROCM_CHIP")
    if not chip:
        raise SystemExit("set TESSERA_ROCM_CHIP (gfx1151 or gfx1201)")
    image_arch = _image_arch()

    rng = np.random.default_rng(11)
    x = rng.standard_normal((BATCH, SAMPLES)).astype(np.float32)
    window = (0.25 + np.hanning(NFFT)).astype(np.float32)
    dx = rng.standard_normal(x.shape).astype(np.float32)
    dwindow = rng.standard_normal(window.shape).astype(np.float32)
    spectrum = np.ascontiguousarray(_stft_reference(x, window).astype(np.complex64))
    dspectrum = (rng.standard_normal(spectrum.shape) +
                 1j * rng.standard_normal(spectrum.shape)).astype(np.complex64)
    cotangent = rng.standard_normal((BATCH, LENGTH)).astype(np.float32)

    cases = (
        ("stft_8x16000_n512_h128", "forward", lambda: stft_forward(x, window),
         lambda: _stft_reference(x, window)),
        ("istft_8x122x257", "forward", lambda: istft_forward(spectrum, window),
         lambda: _istft_reference(spectrum, window)),
        ("stft_jvp_8x16000_n512_h128", "native_jvp",
         lambda: stft_jvp.native_jvp(x, window, tangents=(dx, dwindow)), None),
        ("stft_vjp_8x16000_n512_h128", "native_backward",
         lambda: stft_vjp.native_backward(x, window, out_cotangents=dspectrum), None),
        ("istft_jvp_8x122x257", "native_jvp",
         lambda: istft_jvp.native_jvp(spectrum, window, tangents=(dspectrum, dwindow)), None),
        ("istft_vjp_8x122x257", "native_backward",
         lambda: istft_vjp.native_backward(spectrum, window, out_cotangents=cotangent), None),
    )
    prefixes = tuple(p for p in args.only.split(",") if p)
    rows = []
    for name, route, call, reference in cases:
        if prefixes and not name.startswith(prefixes):
            continue
        row: dict[str, Any] = {
            "backend": "rocm", "op": name.split("_")[0], "case": name,
            "shape": [[BATCH, SAMPLES]], "dtype": "float32", "device": chip,
            "image_arch": image_arch, "tessera_version": "0.1.0", "route": route,
            "latency_source": "host_wall_synchronized",
        }
        try:
            first, cold = _first_call(call)
            if reference is not None:
                actual = np.asarray(first[0] if isinstance(first, tuple) else first)
                expected = reference()
                scale = max(1.0, float(np.max(np.abs(expected))))
                row["max_rel_error"] = float(np.max(np.abs(actual - expected))) / scale
                if row["max_rel_error"] > 1e-4:
                    raise RuntimeError(f"disagrees with NumPy: {row['max_rel_error']:.3e}")
            samples_ms = _time(call, args.warmup, args.repeats)
        except Exception as exc:  # report, do not time a wrong or refused case
            row.update(ok=False, error=f"{type(exc).__name__}: {exc}"[:500])
            rows.append(row)
            print(json.dumps(row), flush=True)
            continue
        row.update(ok=True, cold_ms=cold, latency_ms=float(statistics.median(samples_ms)),
                   p10_ms=float(np.percentile(samples_ms, 10)),
                   p90_ms=float(np.percentile(samples_ms, 90)), numpy_ms=None,
                   tflops=None, memory_bw_gb_s=None, repeats=args.repeats)
        rows.append(row)
        print(json.dumps(row), flush=True)
    packet = {
        "schema": "tessera.rocm_spectral_benchmark.v1",
        "host": platform.node(), "platform": platform.platform(),
        "device": chip, "image_arch": image_arch, "rows": rows,
    }
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(packet, handle, indent=2)


if __name__ == "__main__":
    main()
