#!/usr/bin/env python3
"""Matched gfx1201 MXFP4 W4A8 production benchmark.

The Tessera route is always available.  Two independently built comparison
engines may be supplied explicitly:

* ``--radiance-module``: the pybind module built from
  ``radiance_mxfp4_fp8.hip``;
* ``--libr4d-library``: a shared library exporting
  ``r4d_gemm_mxfp4a8_nt_m64``.

Every engine receives the same logical E4M3 activation, E2M1 weight, E8M0
scale plane, and per-token scale.  Only the physical weight layout differs.
The generated values make the row-reference fold exact, so a folded
comparison cannot hide approximation error.  Outputs must agree bit-for-bit
before timing.  Timed launches rotate weight copies whose combined footprint
exceeds the gfx1201 last-level cache for the production shapes.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import sys
from typing import Any, Callable

import ml_dtypes
import numpy as np

from tessera import runtime as rt
from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler.rocm_mxfp4_native import MXFP4Schedule, package_mxfp4_w4a8_wmma


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    return subprocess.check_output(
        ("git", "-C", str(ROOT), "rev-parse", "HEAD"), text=True
    ).strip()


def _selected_device_name(hip: ctypes.CDLL) -> str:
    device = ctypes.c_int()
    name = ctypes.create_string_buffer(256)
    if hip.hipGetDevice(ctypes.byref(device)) != 0:
        raise RuntimeError("hipGetDevice failed while identifying benchmark device")
    if hip.hipDeviceGetName(name, len(name), device.value) != 0:
        raise RuntimeError("hipDeviceGetName failed while identifying benchmark device")
    return name.value.decode("utf-8", errors="strict").strip()


@dataclass(frozen=True)
class Case:
    workload: str
    m: int
    n: int
    k: int

    @property
    def label(self) -> str:
        return f"{self.workload}_{self.m}x{self.n}x{self.k}"


CASES = (
    Case("decode", 8, 5120, 8704),
    Case("decode", 8, 17408, 5120),
    Case("prefill", 256, 5120, 8704),
    Case("prefill", 1024, 17408, 5120),
)


def _logical_inputs(case: Case) -> dict[str, np.ndarray]:
    """Build exact-fold-friendly values without materializing unpacked W."""

    m, n, k = case.m, case.n, case.k
    if k % 32 or n % 16:
        raise ValueError("matched MXFP4 cases require K % 32 == 0 and N % 16 == 0")

    # E4M3 encodings of {1, 2, -1, -2}.  E2M1 packed nibbles cycle through
    # {+1, +2, -1, -2}; every product and partial sum is a half-integer well
    # inside exact FP32 range even at the production K values.
    a_codes = np.array((0x38, 0x40, 0xB8, 0xC0), dtype=np.uint8)
    a = np.empty((m, k), dtype=np.uint8)
    k_axis = np.arange(k, dtype=np.int64)
    for row in range(m):
        a[row] = a_codes[(k_axis + row) & 3]

    e2m1_codes = np.array((2, 4, 10, 12), dtype=np.uint8)
    packed = np.empty((n, k // 2), dtype=np.uint8)
    pair_axis = np.arange(k // 2, dtype=np.int64)
    for row in range(n):
        low = e2m1_codes[(2 * pair_axis + row) & 3]
        high = e2m1_codes[(2 * pair_axis + row + 1) & 3]
        packed[row] = low | (high << 4)

    # Delta <= 2 keeps the independent row-reference route exactly
    # representable in E4M3, including the reserved-zero semantics staying out
    # of this performance comparison.
    groups = k // 32
    b_scale = np.empty((groups, n), dtype=np.uint8)
    n_axis = np.arange(n, dtype=np.int64)
    for group in range(groups):
        b_scale[group] = 126 + ((group + n_axis) % 3).astype(np.uint8)
    row_reference = b_scale.max(axis=0).astype(np.uint8, copy=False)
    return {
        "a": np.ascontiguousarray(a),
        "a_scale": np.ones((m,), dtype=np.float32),
        "packed_row_major": np.ascontiguousarray(packed),
        "b_scale": np.ascontiguousarray(b_scale),
        "row_reference": np.ascontiguousarray(row_reference),
        "output": np.zeros((m, n), dtype=ml_dtypes.bfloat16),
    }


def _sampled_exact_reference(
    case: Case, inputs: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an independent FP32-dequantized oracle for sampled output cells."""
    row_count = min(case.m, 8)
    col_count = min(case.n, 16)
    rows = np.unique(np.linspace(0, case.m - 1, row_count, dtype=np.int64))
    cols = np.unique(np.linspace(0, case.n - 1, col_count, dtype=np.int64))
    activation = inputs["a"][rows].view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    activation *= inputs["a_scale"][rows, None]
    codes = mx.unpack_e2m1_codes(inputs["packed_row_major"][cols])
    weights = mx.exact_weights(codes, inputs["b_scale"][:, cols])
    expected = (activation @ weights.T).astype(ml_dtypes.bfloat16)
    return rows, cols, expected


def _fragment_order(packed: np.ndarray, n: int, k: int) -> np.ndarray:
    return np.ascontiguousarray(
        packed.reshape(n // 16, 16, k // 16, 2, 4)
        .transpose(0, 2, 3, 1, 4)
        .reshape(n, k // 2)
    )


class _DeviceArrays:
    def __init__(self, hip: ctypes.CDLL, arrays: tuple[np.ndarray, ...]) -> None:
        self.hip = hip
        self.host = arrays
        self.device: list[ctypes.c_void_p] = []
        for array in arrays:
            pointer = ctypes.c_void_p()
            if hip.hipMalloc(ctypes.byref(pointer), int(array.nbytes)) != 0:
                self.close()
                raise RuntimeError("matched MXFP4 benchmark hipMalloc failed")
            self.device.append(pointer)
            if hip.hipMemcpy(
                pointer, array.ctypes.data_as(ctypes.c_void_p), int(array.nbytes), 1
            ) != 0:
                self.close()
                raise RuntimeError("matched MXFP4 benchmark host-to-device copy failed")

    def download(self, index: int) -> np.ndarray:
        host = self.host[index]
        if self.hip.hipMemcpy(
            host.ctypes.data_as(ctypes.c_void_p),
            self.device[index],
            int(host.nbytes),
            2,
        ) != 0:
            raise RuntimeError("matched MXFP4 benchmark device-to-host copy failed")
        return host.copy()

    def close(self) -> None:
        for pointer in reversed(getattr(self, "device", [])):
            if pointer.value:
                self.hip.hipFree(pointer)
        self.device = []


class _Engine:
    def __init__(
        self,
        name: str,
        hip: ctypes.CDLL,
        copies: list[_DeviceArrays],
        launch: Callable[[_DeviceArrays], None],
        output_index: int,
        metadata: dict[str, object],
    ) -> None:
        self.name = name
        self.hip = hip
        self.copies = copies
        self._launch = launch
        self.output_index = output_index
        self.metadata = metadata
        self._next = 0

    def launch(self) -> None:
        current = self.copies[self._next]
        self._next = (self._next + 1) % len(self.copies)
        self._launch(current)

    def output(self) -> np.ndarray:
        self._next = 0
        self.launch()
        if self.hip.hipDeviceSynchronize() != 0:
            raise RuntimeError(f"{self.name} correctness synchronization failed")
        used = self.copies[0]
        return used.download(self.output_index)

    def close(self) -> None:
        for copy in self.copies:
            copy.close()


def _copies(
    hip: ctypes.CDLL, arrays: tuple[np.ndarray, ...], count: int
) -> list[_DeviceArrays]:
    return [
        _DeviceArrays(hip, tuple(np.zeros_like(a) if i == len(arrays) - 1 else a for i, a in enumerate(arrays)))
        for _ in range(count)
    ]


def _tessera_engine(
    hip: ctypes.CDLL,
    case: Case,
    inputs: dict[str, np.ndarray],
    copies: int,
    group_m: int | None,
    split_k: int | None,
    stream: ctypes.c_void_p | None = None,
    k_step_schedule: str | None = None,
) -> _Engine:
    schedule = None
    if group_m is not None or split_k is not None or k_step_schedule is not None:
        schedule = MXFP4Schedule(
            case.workload,
            group_m=group_m or 1,
            split_k=split_k or 1,
            k_step_schedule=k_step_schedule or "isolated_scale_group",
        )
    package = package_mxfp4_w4a8_wmma(case.m, case.n, case.k, schedule=schedule)
    module = ctypes.c_void_p()
    function = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(module), package.image.payload) != 0:
        raise RuntimeError("Tessera MXFP4 module load failed")
    if hip.hipModuleGetFunction(
        ctypes.byref(function), module, package.descriptor.entry_symbol.encode()
    ) != 0:
        hip.hipModuleUnload(module)
        raise RuntimeError("Tessera MXFP4 entry is missing")
    arrays = (
        inputs["a"],
        mx.convert_weight_layout(
            inputs["packed_row_major"],
            source=mx.MXFP4_CHECKPOINT_LAYOUT_V1,
            destination=str(package.descriptor.provenance["weight_layout"]),
        ),
        inputs["a_scale"],
        inputs["b_scale"],
        inputs["output"],
    )
    device_copies = _copies(hip, arrays, copies)

    def launch(bundle: _DeviceArrays) -> None:
        values: list[Any] = [
            *(ctypes.c_void_p(pointer.value) for pointer in bundle.device),
            ctypes.c_int64(case.m),
            ctypes.c_int64(case.n),
            ctypes.c_int64(case.k),
        ]
        arguments = (ctypes.c_void_p * len(values))(
            *[ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in values]
        )
        grid = package.descriptor.geometry.grid
        workgroup = package.descriptor.geometry.workgroup
        assert grid is not None and workgroup is not None
        rc = hip.hipModuleLaunchKernel(
            function, *grid, *workgroup, 0, stream, arguments, None
        )
        if rc != 0:
            raise RuntimeError(f"Tessera MXFP4 launch failed rc={rc}")

    engine = _Engine(
        "tessera",
        hip,
        device_copies,
        launch,
        4,
        {
            "abi": package.descriptor.abi_id,
            "image_sha256": hashlib.sha256(package.image.payload).hexdigest(),
            "route": package.descriptor.provenance["route"],
            "weight_layout": package.descriptor.provenance["weight_layout"],
            "compiler_fingerprint": package.image.compiler_fingerprint,
            "toolchain_fingerprint": package.image.toolchain_fingerprint,
            "schedule": {
                name: package.descriptor.provenance[name]
                for name in (
                    "workload",
                    "group_m",
                    "split_k",
                    "stages",
                    "waves_per_eu",
                    "cache_modifier",
                )
            },
        },
    )
    original_close = engine.close

    def close() -> None:
        original_close()
        hip.hipModuleUnload(module)

    engine.close = close  # type: ignore[method-assign]
    return engine


def _load_radiance(path: Path):
    os.environ["RADIANCE_MXFP4_WPERM"] = "1"
    os.environ["RADIANCE_MXFP4_DECODE_MAX_M"] = "64"
    spec = importlib.util.spec_from_file_location("radiance_mxfp4_fp8", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Radiance module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _radiance_engine(
    hip: ctypes.CDLL,
    module: Any,
    case: Case,
    inputs: dict[str, np.ndarray],
    copies: int,
) -> _Engine:
    if case.k % 64:
        raise ValueError("Radiance comparison requires K divisible by 64")
    arrays = (
        inputs["a"],
        _fragment_order(inputs["packed_row_major"], case.n, case.k),
        inputs["b_scale"],
        inputs["row_reference"],
        inputs["a_scale"],
        inputs["output"],
    )
    device_copies = _copies(hip, arrays, copies)
    scratch: _DeviceArrays | None = None
    if case.m <= 64:
        scratch = _DeviceArrays(
            hip,
            (
                np.empty((4 * case.m * case.n,), dtype=np.float32),
                np.zeros(((case.n + 127) // 128 + 8,), dtype=np.int32),
            ),
        )
        module.set_decode_scratch(
            scratch.device[0].value,
            scratch.host[0].nbytes,
            scratch.device[1].value,
        )

    def launch(bundle: _DeviceArrays) -> None:
        p = bundle.device
        module.launch(
            p[0].value,
            p[1].value,
            p[2].value,
            p[3].value,
            p[4].value,
            p[5].value,
            case.m,
            case.n,
            case.k,
            0,
        )

    engine = _Engine(
        "radiance",
        hip,
        device_copies,
        launch,
        5,
        {"weight_layout": "fragment_order", "row_reference_fold": "exact_delta_le_2"},
    )
    original_close = engine.close

    def close() -> None:
        original_close()
        if scratch is not None:
            scratch.close()

    engine.close = close  # type: ignore[method-assign]
    return engine


def _libr4d_engine(
    hip: ctypes.CDLL,
    library: ctypes.CDLL,
    case: Case,
    inputs: dict[str, np.ndarray],
    copies: int,
) -> _Engine | None:
    if case.m > 64:
        return None
    function = library.r4d_gemm_mxfp4a8_nt_m64
    function.argtypes = [ctypes.c_long] * 6 + [ctypes.c_int] * 7 + [ctypes.c_long]
    function.restype = None
    arrays = (
        inputs["a"],
        inputs["a_scale"],
        _fragment_order(inputs["packed_row_major"], case.n, case.k),
        inputs["b_scale"],
        inputs["row_reference"],
        inputs["output"],
    )
    device_copies = _copies(hip, arrays, copies)
    mb = max(1, min(4, (case.m + 15) // 16))
    split_k = 8 if case.m <= 16 else 4
    while split_k > 1 and case.k % (split_k * 32):
        split_k //= 2
    wave_groups, n_per_wave = 2, 2

    def launch(bundle: _DeviceArrays) -> None:
        function(
            *(ctypes.c_long(pointer.value) for pointer in bundle.device),
            case.m,
            case.k,
            case.n,
            wave_groups,
            split_k,
            mb,
            n_per_wave,
            0,
        )

    return _Engine(
        "libr4d",
        hip,
        device_copies,
        launch,
        5,
        {
            "weight_layout": "fragment_order",
            "row_reference_fold": "exact_delta_le_2",
            "WV": wave_groups,
            "SK": split_k,
            "MB": mb,
            "NPW": n_per_wave,
        },
    )


def _measure(
    hip: ctypes.CDLL,
    engine: _Engine,
    *,
    warmup: int,
    trials: int,
    iterations: int,
) -> list[float]:
    for _ in range(warmup):
        engine.launch()
    if hip.hipDeviceSynchronize() != 0:
        raise RuntimeError(f"{engine.name} warmup synchronization failed")
    samples: list[float] = []
    for _ in range(trials):
        start, stop = ctypes.c_void_p(), ctypes.c_void_p()
        if hip.hipEventCreate(ctypes.byref(start)) != 0:
            raise RuntimeError("MXFP4 start-event creation failed")
        if hip.hipEventCreate(ctypes.byref(stop)) != 0:
            hip.hipEventDestroy(start)
            raise RuntimeError("MXFP4 stop-event creation failed")
        try:
            if hip.hipEventRecord(start, None) != 0:
                raise RuntimeError("MXFP4 start-event record failed")
            for _ in range(iterations):
                engine.launch()
            if (
                hip.hipEventRecord(stop, None) != 0
                or hip.hipEventSynchronize(stop) != 0
            ):
                raise RuntimeError("MXFP4 stop-event synchronization failed")
            elapsed = ctypes.c_float()
            if hip.hipEventElapsedTime(ctypes.byref(elapsed), start, stop) != 0:
                raise RuntimeError("MXFP4 HIP event timing failed")
            sample = float(elapsed.value) / iterations
            if not math.isfinite(sample) or sample <= 0.0:
                raise RuntimeError(f"invalid MXFP4 timing sample {sample}")
            samples.append(sample)
        finally:
            hip.hipEventDestroy(start)
            hip.hipEventDestroy(stop)
    return samples


def benchmark(
    *,
    cases: tuple[Case, ...],
    radiance_module: Path | None,
    libr4d_library: Path | None,
    copies: int,
    warmup: int,
    trials: int,
    iterations: int,
    tessera_group_m: int | None,
    tessera_split_k: int | None,
    tessera_k_step_schedule: str | None,
    radiance_revision: str | None,
    libr4d_revision: str | None,
) -> dict[str, object]:
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("matched MXFP4 production benchmark requires selected gfx1201")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        raise RuntimeError("matched MXFP4 production benchmark requires HIP")
    if radiance_module is not None and not radiance_revision:
        raise ValueError("Radiance comparison requires --radiance-revision")
    if libr4d_library is not None and not libr4d_revision:
        raise ValueError("libr4d comparison requires --libr4d-revision")
    radiance = _load_radiance(radiance_module) if radiance_module else None
    libr4d = ctypes.CDLL(str(libr4d_library)) if libr4d_library else None
    rows: list[dict[str, object]] = []
    for case in cases:
        inputs = _logical_inputs(case)
        engines = [
            _tessera_engine(
                hip,
                case,
                inputs,
                copies,
                tessera_group_m,
                tessera_split_k,
                k_step_schedule=tessera_k_step_schedule,
            )
        ]
        if radiance is not None:
            engines.append(_radiance_engine(hip, radiance, case, inputs, copies))
        if libr4d is not None:
            candidate = _libr4d_engine(hip, libr4d, case, inputs, copies)
            if candidate is not None:
                engines.append(candidate)
        try:
            outputs = {engine.name: engine.output() for engine in engines}
            sample_rows, sample_cols, expected = _sampled_exact_reference(case, inputs)
            for name, output in outputs.items():
                sampled = output[np.ix_(sample_rows, sample_cols)]
                if not np.array_equal(sampled.view(np.uint16), expected.view(np.uint16)):
                    mismatch = sampled.view(np.uint16) != expected.view(np.uint16)
                    first = tuple(int(value) for value in np.argwhere(mismatch)[0])
                    row = int(sample_rows[first[0]])
                    col = int(sample_cols[first[1]])
                    raise RuntimeError(
                        f"{case.label}: {name} fails sampled exact FP32 reference at "
                        f"[{row},{col}]: got=0x{int(output.view(np.uint16)[row, col]):04x} "
                        f"expected=0x{int(expected.view(np.uint16)[first]):04x}"
                    )
            baseline = outputs["tessera"]
            for name, output in outputs.items():
                if not np.array_equal(output.view(np.uint16), baseline.view(np.uint16)):
                    mismatch = int(np.count_nonzero(output.view(np.uint16) != baseline.view(np.uint16)))
                    first = tuple(
                        int(value)
                        for value in np.argwhere(
                            output.view(np.uint16) != baseline.view(np.uint16)
                        )[0]
                    )
                    raise RuntimeError(
                        f"{case.label}: {name} differs from Tessera in {mismatch} BF16 cells; "
                        f"first={first} tessera=0x{int(baseline.view(np.uint16)[first]):04x} "
                        f"{name}=0x{int(output.view(np.uint16)[first]):04x}"
                    )
            for engine in engines:
                samples = _measure(
                    hip,
                    engine,
                    warmup=warmup,
                    trials=trials,
                    iterations=iterations,
                )
                weight_bytes = case.n * case.k // 2 + case.n * case.k // 32
                median_ms = statistics.median(samples)
                rows.append(
                    {
                        "case": case.label,
                        "workload": case.workload,
                        "shape": [case.m, case.n, case.k],
                        "engine": engine.name,
                        "samples_ms": samples,
                        "median_ms": median_ms,
                        "min_ms": min(samples),
                        "effective_weight_gbps": weight_bytes / median_ms / 1.0e6,
                        "matched_output_sha256": hashlib.sha256(
                            outputs[engine.name].view(np.uint8)
                        ).hexdigest(),
                        "metadata": engine.metadata,
                    }
                )
        finally:
            for engine in reversed(engines):
                engine.close()
    return {
        "schema": "tessera.rocm.gfx1201_mxfp4_matched_benchmark.v1",
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "device": _selected_device_name(hip),
        "architecture": rt._rocm_live_arch(),
        "clock": "hipEventElapsedTime",
        "copies": copies,
        "warmup": warmup,
        "trials": trials,
        "iterations": iterations,
        "input_contract": (
            "matched logical E4M3/E2M1/E8M0 values; row-reference exponent delta <= 2; "
            "bit-exact BF16 agreement required before timing"
        ),
        "source": {
            "revision": _git_revision(),
            "generator_sha256": _sha256(
                ROOT / "python/tessera/compiler/rocm_mxfp4_native.py"
            ),
            "benchmark_sha256": _sha256(Path(__file__)),
        },
        "independent_comparisons": {
            "radiance": None
            if radiance_module is None
            else {
                "revision": radiance_revision,
                "binary_sha256": _sha256(radiance_module),
            },
            "libr4d": None
            if libr4d_library is None
            else {
                "revision": libr4d_revision,
                "binary_sha256": _sha256(libr4d_library),
            },
        },
        "rows": rows,
    }


def _parse_case(value: str) -> Case:
    try:
        workload, shape = value.split(":", 1)
        m, n, k = (int(part) for part in shape.lower().split("x"))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("case must be decode:MxNxK or prefill:MxNxK") from exc
    if workload not in {"decode", "prefill"} or min(m, n, k) <= 0:
        raise argparse.ArgumentTypeError("case must use decode/prefill and positive dimensions")
    return Case(workload, m, n, k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", type=_parse_case)
    parser.add_argument("--radiance-module", type=Path)
    parser.add_argument("--libr4d-library", type=Path)
    parser.add_argument("--radiance-revision")
    parser.add_argument("--libr4d-revision")
    parser.add_argument("--copies", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--tessera-group-m", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--tessera-split-k", type=int, choices=(1, 2, 4, 8))
    parser.add_argument(
        "--tessera-k-step-schedule",
        choices=("isolated_scale_group", "relaxed"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.copies, args.warmup, args.trials, args.iterations) <= 0:
        parser.error("copies, warmup, trials, and iterations must be positive")
    packet = benchmark(
        cases=tuple(args.case or CASES),
        radiance_module=args.radiance_module,
        libr4d_library=args.libr4d_library,
        copies=args.copies,
        warmup=args.warmup,
        trials=args.trials,
        iterations=args.iterations,
        tessera_group_m=args.tessera_group_m,
        tessera_split_k=args.tessera_split_k,
        tessera_k_step_schedule=args.tessera_k_step_schedule,
        radiance_revision=args.radiance_revision,
        libr4d_revision=args.libr4d_revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps(packet, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
