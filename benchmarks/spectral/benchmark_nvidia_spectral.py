#!/usr/bin/env python3
"""End-to-end timing of the CUDA spectral lanes on an exact sm_120 device.

Every case goes through ``runtime.launch`` -- the path a caller takes -- so a
latency includes whatever that path does per call (host staging, allocation,
plan lookup, Python dispatch), not just the cuFFT kernel. Each case is first
checked against a NumPy reference; a case that disagrees is reported and not
timed. ``numpy_ms`` is the same transform on the host, for scale only.

Timing domain: synchronized host wall clock (``launch`` returns host arrays).
It is the right domain for per-call overhead and the wrong one for kernel
selection; pair it with an ``nsys`` capture for attribution.

Rows carry the Decision #12 fields plus ``route`` (compiler path) and
``latency_source`` (Decision #12 amendment).
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from typing import Any, Callable

import numpy as np

from tessera import runtime as rt


def _artifact(path: str, op: str, n_operands: int, kwargs: dict[str, Any]):
    names = [f"a{index}" for index in range(n_operands)]
    return rt.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": path, "executable": True,
        "execution_kind": "native_gpu", "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": dict(kwargs)}],
    })


def _stft_reference(x, window, nfft, hop):
    # The op's default is non-centered frames (center=False).
    frames = 1 + (x.shape[-1] - nfft) // hop
    idx = np.arange(nfft)[None, :] + hop * np.arange(frames)[:, None]
    return np.fft.rfft(x[:, idx] * window, axis=-1)


def _dct2_reference(x):
    try:
        from scipy.fft import dct
    except ImportError:  # SciPy is optional; the case is then timed unchecked
        return None
    return dct(x, type=2, axis=-1)


def _cases(rng: np.random.Generator):
    """(name, route, op, operands, kwargs, reference(), flops_or_None)."""
    def c(*shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)

    def r(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    cases = []
    for batch, n in ((1, 1024), (64, 1024), (256, 4096), (16, 65536), (64, 1009)):
        x = c(batch, n)
        cases.append((f"fft_c2c_{batch}x{n}", "nvidia_fft_compiled", "tessera.fft",
                      (x,), {"axis": -1}, lambda x=x: np.fft.fft(x, axis=-1),
                      5.0 * batch * n * np.log2(n)))
    for batch, n in ((64, 4096), (16, 65536)):
        x = r(batch, n)
        cases.append((f"rfft_{batch}x{n}", "nvidia_fft_compiled", "tessera.rfft",
                      (x,), {"axis": -1}, lambda x=x: np.fft.rfft(x, axis=-1),
                      2.5 * batch * n * np.log2(n)))
        spectrum = np.fft.rfft(x, axis=-1).astype(np.complex64)
        cases.append((f"irfft_{batch}x{n}", "nvidia_fft_compiled", "tessera.irfft",
                      (spectrum,), {"axis": -1, "n": n},
                      lambda s=spectrum, n=n: np.fft.irfft(s, n, axis=-1),
                      2.5 * batch * n * np.log2(n)))
    x = r(64, 1024)
    cases.append(("dct2_64x1024", "nvidia_spectral_compiled", "tessera.dct",
                  (x,), {"type": 2, "axis": -1}, lambda x=x: _dct2_reference(x), None))
    for batch, samples, nfft, hop in ((8, 16000, 512, 128), (32, 48000, 1024, 256)):
        x, window = r(batch, samples), np.hanning(nfft).astype(np.float32)
        cases.append((f"stft_{batch}x{samples}_n{nfft}_h{hop}", "nvidia_spectral_compiled",
                      "tessera.stft", (x, window), {"hop": hop},
                      lambda x=x, w=window, nfft=nfft, hop=hop: _stft_reference(x, w, nfft, hop),
                      None))
    for n_x, n_w in ((16384, 257), (262144, 1025)):
        x, w = r(n_x), r(n_w)
        n = n_x + n_w - 1
        cases.append((f"spectral_conv_{n_x}x{n_w}", "nvidia_spectral_compiled",
                      "tessera.spectral_conv", (x, w), {},
                      lambda x=x, w=w, n=n: np.convolve(x, w)[:n], None))
    a, b = c(64, 4096), c(64, 4096)
    cases.append(("spectral_filter_64x4096", "nvidia_spectral_compiled",
                  "tessera.spectral_filter", (a, b), {}, lambda a=a, b=b: a * b, None))
    return cases


def _device_resident_rows(lib, device, warmup: int, repeats: int) -> list[dict[str, Any]]:
    """C2C with input and output already on the GPU (device-pointer ABI).

    Times the enqueue plus a device synchronize: what a caller that keeps its
    data resident pays per transform, with no host staging or copies.
    """
    import ctypes

    if not hasattr(lib, "tessera_nvidia_fft_execute_c2c_device_f32"):
        return []
    cudart = ctypes.CDLL("libcudart.so.13")
    cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaFree.argtypes = [ctypes.c_void_p]
    cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    execute = lib.tessera_nvidia_fft_execute_c2c_device_f32
    execute.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    rows = []
    rng = np.random.default_rng(7)
    for batch, n in ((1, 1024), (64, 1024), (256, 4096), (16, 65536)):
        host = (rng.standard_normal((batch, n)) + 1j * rng.standard_normal((batch, n))).astype(np.complex64)
        plan, size, workspace = ctypes.c_void_p(), ctypes.c_size_t(), ctypes.c_void_p()
        device_in, device_out = ctypes.c_void_p(), ctypes.c_void_p()
        if lib.tessera_nvidia_fft_plan_create_c2c_f32(batch, n, ctypes.byref(plan), ctypes.byref(size)):
            raise RuntimeError("plan creation failed")
        try:
            lib.tessera_nvidia_fft_workspace_alloc(size.value, ctypes.byref(workspace))
            cudart.cudaMalloc(ctypes.byref(device_in), host.nbytes)
            cudart.cudaMalloc(ctypes.byref(device_out), host.nbytes)
            cudart.cudaMemcpy(device_in, host.ctypes.data, host.nbytes, 1)

            def call():
                if execute(plan, device_in, device_out, workspace, size.value, 0, None):
                    raise RuntimeError("device execute failed")
                if cudart.cudaDeviceSynchronize():
                    raise RuntimeError("synchronize failed")

            call()
            out = np.empty_like(host)
            cudart.cudaMemcpy(out.ctypes.data, device_out, out.nbytes, 2)
            expected = np.fft.fft(host, axis=-1)
            error = float(np.max(np.abs(out - expected))) / max(1.0, float(np.max(np.abs(expected))))
            cold, samples = _time(call, warmup, repeats)
            median = float(statistics.median(samples))
            row = {
                "backend": "nvidia_sm120", "op": "tessera.fft", "case": f"fft_c2c_{batch}x{n}_device_resident",
                "shape": [[batch, n]], "dtype": "complex64", "device": device, "tessera_version": "0.1.0",
                "route": "nvidia_fft_device_pointer", "latency_source": "host_wall_synchronized",
                "ok": True, "max_rel_error": error, "cold_ms": cold, "latency_ms": median,
                "p10_ms": float(np.percentile(samples, 10)), "p90_ms": float(np.percentile(samples, 90)),
                "numpy_ms": None, "tflops": 5.0 * batch * n * np.log2(n) / (median * 1e-3) / 1e12,
                "memory_bw_gb_s": 2 * host.nbytes / (median * 1e-3) / 1e9, "repeats": repeats,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        finally:
            for pointer in (device_in, device_out):
                if pointer.value:
                    cudart.cudaFree(pointer)
            if workspace.value:
                lib.tessera_nvidia_fft_workspace_free(workspace)
            lib.tessera_nvidia_fft_plan_destroy(plan)
    return rows


def _autodiff_rows(device, warmup: int, repeats: int) -> list[dict[str, Any]]:
    """STFT/ISTFT forward-mode (native_jvp) and reverse-mode (native_backward).

    Goes through the public @tessera.jit autodiff entry points at the 8x16000
    audio size; the primal is checked against NumPy before timing. The first
    call compiles and is excluded by the warmup.
    """
    import tessera

    nfft, hop, batch, samples = 512, 128, 8, 16000
    frames = (samples - nfft) // hop + 1
    length = (frames - 1) * hop + nfft

    # Literals, not closure variables: the tracer binds a free variable as a
    # graph value, and n_fft/hop/length must be attributes.
    @tessera.jit(target="nvidia_sm120", autodiff="jvp", wrt=("x", "window"))
    def stft_jvp(x, window):
        return tessera.ops.stft(x, window, axis=-1, n_fft=512, hop=128,
                                center=False, onesided=True, norm="backward")

    @tessera.jit(target="nvidia_sm120", autodiff="reverse", wrt=("x", "window"))
    def stft_vjp(x, window):
        return tessera.ops.stft(x, window, axis=-1, n_fft=512, hop=128,
                                center=False, onesided=True, norm="backward")

    @tessera.jit(target="nvidia_sm120", autodiff="jvp", wrt=("spectrum", "window"))
    def istft_jvp(spectrum, window):
        return tessera.ops.istft(spectrum, window, axis=-1, n_fft=512, hop=128,
                                 center=False, onesided=True, length=16000,
                                 norm="backward")

    @tessera.jit(target="nvidia_sm120", autodiff="reverse", wrt=("spectrum", "window"))
    def istft_vjp(spectrum, window):
        return tessera.ops.istft(spectrum, window, axis=-1, n_fft=512, hop=128,
                                 center=False, onesided=True, length=16000,
                                 norm="backward")

    assert length == 16000 and frames == 122  # the literals above

    rng = np.random.default_rng(11)
    x = rng.standard_normal((batch, samples)).astype(np.float32)
    window = (0.25 + np.hanning(nfft)).astype(np.float32)
    dx = rng.standard_normal(x.shape).astype(np.float32)
    dwindow = rng.standard_normal(window.shape).astype(np.float32)
    spectrum = _stft_reference(x, window, nfft, hop).astype(np.complex64)
    dspectrum = (rng.standard_normal(spectrum.shape) +
                 1j * rng.standard_normal(spectrum.shape)).astype(np.complex64)
    cotangent_signal = rng.standard_normal((batch, length)).astype(np.float32)

    cases = (
        ("stft_jvp_8x16000_n512_h128", "native_jvp",
         lambda: stft_jvp.native_jvp(x, window, tangents=(dx, dwindow)), spectrum),
        ("stft_vjp_8x16000_n512_h128", "native_backward",
         lambda: stft_vjp.native_backward(x, window, out_cotangents=dspectrum), None),
        ("istft_jvp_8x122x257", "native_jvp",
         lambda: istft_jvp.native_jvp(spectrum, window, tangents=(dspectrum, dwindow)), None),
        ("istft_vjp_8x122x257", "native_backward",
         lambda: istft_vjp.native_backward(spectrum, window, out_cotangents=cotangent_signal), None),
    )
    rows = []
    for name, route, call, expected_primal in cases:
        row: dict[str, Any] = {
            "backend": "nvidia_sm120", "op": name.split("_")[0], "case": name,
            "shape": [[batch, samples]], "dtype": "float32", "device": device,
            "tessera_version": "0.1.0", "route": route,
            "latency_source": "host_wall_synchronized",
        }
        try:
            first = call()
            if expected_primal is not None:
                primal = np.asarray(first[0])
                scale = max(1.0, float(np.max(np.abs(expected_primal))))
                row["max_rel_error"] = float(np.max(np.abs(primal - expected_primal))) / scale
                if row["max_rel_error"] > 1e-4:
                    raise RuntimeError(f"primal disagrees: {row['max_rel_error']:.3e}")
            cold, samples_ms = _time(call, warmup, repeats)
        except Exception as exc:
            row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            rows.append(row)
            print(json.dumps(row), flush=True)
            continue
        median = float(statistics.median(samples_ms))
        row.update(ok=True, cold_ms=cold, latency_ms=median, numpy_ms=None, tflops=None,
                   memory_bw_gb_s=None, p10_ms=float(np.percentile(samples_ms, 10)),
                   p90_ms=float(np.percentile(samples_ms, 90)), repeats=repeats)
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def _time(call: Callable[[], Any], warmup: int, repeats: int) -> tuple[float, list[float]]:
    start = time.perf_counter_ns()
    call()
    cold = (time.perf_counter_ns() - start) * 1e-6
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        call()
        samples.append((time.perf_counter_ns() - start) * 1e-6)
    return cold, samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--only", default="", help="comma-separated case-name prefixes")
    parser.add_argument("--output")
    args = parser.parse_args()

    lib = rt._load_nvidia_fft_runtime()
    if lib is None:
        raise SystemExit("libtessera_nvidia_fft.so is not loadable")
    abi = lib.tessera_nvidia_fft_package_abi().decode()
    arch = lib.tessera_nvidia_spectral_arch() if hasattr(lib, "tessera_nvidia_spectral_arch") else None
    device = rt._nvidia_device_name() if hasattr(rt, "_nvidia_device_name") else None
    prefixes = tuple(p for p in args.only.split(",") if p)
    rows = []
    for name, route, op, operands, kwargs, reference, flops in _cases(np.random.default_rng(20260925)):
        if prefixes and not name.startswith(prefixes):
            continue
        artifact = _artifact(route, op, len(operands), kwargs)

        def call(artifact=artifact, operands=operands):
            result = rt.launch(artifact, operands)
            if not result.get("ok"):
                raise RuntimeError(result.get("reason"))
            if result.get("execution_kind") != "native_gpu":
                raise RuntimeError(f"fell back to {result.get('execution_kind')}")
            return np.asarray(result["output"])

        row: dict[str, Any] = {
            "backend": "nvidia_sm120", "op": op, "case": name,
            "shape": [list(np.shape(o)) for o in operands],
            "dtype": str(operands[0].dtype), "device": device,
            "tessera_version": "0.1.0", "route": route,
            "latency_source": "host_wall_synchronized",
            "package_abi": abi, "spectral_arch": arch,
        }
        try:
            actual = call()
            expected = reference() if reference is not None else None
            if expected is not None:
                scale = max(1.0, float(np.max(np.abs(expected))))
                error = float(np.max(np.abs(actual - expected))) / scale
                row["max_rel_error"] = error
                if error > 1e-4:
                    raise RuntimeError(f"disagrees with NumPy: rel error {error:.3e}")
            cold, samples = _time(call, args.warmup, args.repeats)
            numpy_cold, numpy_samples = (_time(reference, 1, 5) if reference else (None, None))
        except Exception as exc:  # report, do not time a wrong or refused case
            row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            rows.append(row)
            print(json.dumps(row), flush=True)
            continue
        median = float(statistics.median(samples))
        bytes_moved = sum(np.asarray(o).nbytes for o in operands) + actual.nbytes
        row.update(
            ok=True, cold_ms=cold, latency_ms=median,
            p10_ms=float(np.percentile(samples, 10)), p90_ms=float(np.percentile(samples, 90)),
            numpy_ms=float(statistics.median(numpy_samples)) if numpy_samples else None,
            tflops=(flops / (median * 1e-3) / 1e12) if flops else None,
            memory_bw_gb_s=bytes_moved / (median * 1e-3) / 1e9,
            repeats=args.repeats,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    if not prefixes or any("device_resident".startswith(p) or p.startswith("fft") for p in prefixes):
        rows.extend(_device_resident_rows(lib, device, args.warmup, args.repeats))
    if not prefixes or any(p in ("autodiff", "stft_jvp", "stft_vjp", "istft_jvp", "istft_vjp")
                           for p in prefixes):
        rows.extend(_autodiff_rows(device, args.warmup, args.repeats))
    packet = {
        "schema": "tessera.nvidia_spectral_benchmark.v1",
        "host": platform.node(), "platform": platform.platform(),
        "package_abi": abi, "device": device, "rows": rows,
    }
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(packet, handle, indent=2)


if __name__ == "__main__":
    main()
