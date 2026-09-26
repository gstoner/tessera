#!/usr/bin/env python3
"""Record an exact-host E2E-SPINE-3 packet for the x86 AVX-512 lane.

Sync AVX512-E2E-PACKETS-2026-09-26. The two Zen 5 hosts are separate proof
lanes, so the packet architecture is pinned to the host by CPU model name and
the recorder refuses to write one host's key from the other:

* Princess-Luna, ``AMD RYZEN AI MAX+ 395`` (Strix Halo) -> ``x86_64_avx512_strix_halo``
* Tajasarus, ``AMD Ryzen 7 9800X3D`` (Granite Ridge)    -> ``x86_64_avx512_granite_ridge``

Three further refusals keep the packet comparable across the two hosts:

* the runtime kernel library must be optimized (RUNTIME-LIB-OPT-1): the
  ``runtime_library_build.json`` record of the build tree whose library the
  images embed is read, checked and stamped into ``resources.json``;
* the ``kernel_wall`` domain is timed with the x86 TSC witness
  (``profiler_x86_clock``): the TSC frequency is calibrated over separate
  intervals, every timed window is read with ``rdtscp`` and
  ``CLOCK_MONOTONIC_RAW`` on one pinned CPU, and each window becomes a
  ``tessera.profiler_timing.v1`` sample that must build (under WSL the
  builder itself refuses a TSC that disagrees with the raw clock by more
  than 5%). The recorder applies the same 5% band on bare metal;
* the source tree must be clean, so ``source_commit`` names what ran.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

#: CPU model-name prefix -> the one fleet architecture that host may record.
HOST_ARCHITECTURES: dict[str, str] = {
    "AMD RYZEN AI MAX+ 395": "x86_64_avx512_strix_halo",
    "AMD Ryzen 7 9800X3D": "x86_64_avx512_granite_ridge",
}
PIPELINE = "tessera-lower-to-x86"
AGREEMENT_BAND = 0.05
REQUIRED_LIBRARY = "tessera_x86_elementwise"


def _tensor(shape: tuple[int, ...]):
    from tessera.compiler.graph_ir import IRType

    return IRType(f"tensor<{'x'.join(map(str, shape))}xf32>", tuple(map(str, shape)), "fp32")


def _module(name: str, args: list[tuple[str, tuple[int, ...]]], op_name: str,
            result_shape: tuple[int, ...], kwargs: dict[str, Any], result: str = "o"):
    from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp

    arguments = [IRArg(arg, _tensor(shape)) for arg, shape in args]
    output = _tensor(result_shape)
    return GraphIRModule(functions=[GraphIRFunction(
        name=name, args=arguments, result_types=[output],
        body=[IROp(
            result=result, op_name=op_name, operands=[f"%{arg}" for arg, _ in args],
            operand_types=[str(arg.ir_type) for arg in arguments],
            result_type=str(output), kwargs=kwargs,
        )], return_values=[f"%{result}"],
    )])


def _softmax_module(shape: tuple[int, int]):
    return _module("fleet_softmax", [("x", shape)], "tessera.softmax", shape, {"axis": -1})


def _reduction_module(shape: tuple[int, int]):
    return _module("fleet_reduction", [("x", shape)], "tessera.sum", (shape[0],),
                   {"axis": -1, "keepdims": False})


def _matmul_module(m: int, k: int, n: int):
    return _module("fleet_matmul", [("a", (m, k)), ("b", (k, n))], "tessera.matmul",
                   (m, n), {})


def _attention_module(b: int, h: int, sq: int, sk: int, d: int, dv: int):
    return _module(
        "fleet_attention",
        [("q", (b, h, sq, d)), ("k", (b, h, sk, d)), ("v", (b, h, sk, dv))],
        "tessera.flash_attn", (b, h, sq, dv),
        {"scale": 1.0 / math.sqrt(float(d)), "causal": False},
    )


def _cholesky_module(shape: tuple[int, ...]):
    return _module("fleet_cholesky", [("matrix", shape)], "tessera.cholesky", shape, {},
                   result="result")


def _cpu_identity() -> tuple[str, list[str]]:
    text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    model = next(line.split(":", 1)[1].strip()
                 for line in text.splitlines() if line.startswith("model name"))
    flags = next(line.split(":", 1)[1].strip()
                 for line in text.splitlines() if line.startswith("flags"))
    return model, sorted(flags.split())


def host_architecture(model: str) -> str:
    """The single fleet architecture this CPU may record; refuses any other host."""
    for prefix, architecture in HOST_ARCHITECTURES.items():
        if model.strip().lower().startswith(prefix.lower()):
            return architecture
    raise RuntimeError(
        f"CPU {model!r} is not an assigned x86 AVX-512 packet host; "
        f"expected one of {sorted(HOST_ARCHITECTURES)}")


def execution_environment() -> str:
    return "wsl2" if "microsoft" in platform.release().lower() else "bare_metal"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout


def _build_dir() -> Path:
    selected = os.environ.get("TESSERA_BUILD_DIR")
    if not selected:
        return ROOT / "build"
    path = Path(selected).expanduser()
    return path if path.is_absolute() else ROOT / path


def _library_record() -> dict[str, Any]:
    from tessera.compiler import x86_native
    from tessera.compiler.runtime_library_build import is_optimized, runtime_library_build

    build = _build_dir()
    library = x86_native._library_path(x86_native.X86_AVX512_ARCHITECTURE)
    expected = build / "src/compiler/codegen/tessera_x86_backend/libtessera_x86_elementwise.so"
    if library is None or library.resolve() != expected.resolve():
        raise RuntimeError(
            f"the AVX-512 image library is {library}, not {expected}; the optimization "
            "record would describe a different library than the one measured")
    record = runtime_library_build(build, require=(REQUIRED_LIBRARY,))
    level = str(record["libraries"][REQUIRED_LIBRARY])
    if not is_optimized(level):
        raise RuntimeError(
            f"{REQUIRED_LIBRARY} is not optimized ({level!r}); RUNTIME-LIB-OPT-1 "
            "forbids recording timing from an unoptimized runtime library")
    return {**record, "measured_library": str(expected),
            "measured_library_sha256": hashlib.sha256(expected.read_bytes()).hexdigest()}


def _two_run_medians_ns(call: Callable[[], object], *, samples: int,
                        iterations: int) -> list[float]:
    call()  # First-use loader/compiler/cache effects never enter the timing sample.
    cohorts: tuple[list[float], list[float]] = ([], [])
    for sample in range(samples):
        for cohort in ((0, 1) if sample % 2 == 0 else (1, 0)):
            started = time.perf_counter_ns()
            for _ in range(iterations):
                call()
            cohorts[cohort].append((time.perf_counter_ns() - started) / iterations)
    return [float(statistics.median(values)) for values in cohorts]


def _witnessed_medians_ns(
    call: Callable[[], object], *, family: str, samples: int, iterations: int,
    calibration: dict[str, Any], digests: dict[str, str], environment: str,
    measurement_cpu: int,
) -> tuple[list[float], list[dict[str, Any]], list[float]]:
    """kernel_wall run medians from TSC-witnessed windows, plus the witnesses."""
    from tessera.compiler import profiler_x86_clock as clock
    from tessera.compiler.profiler_timing import (
        build_timing_sample, measured_clock, unavailable_clock,
    )

    def region() -> None:
        for _ in range(iterations):
            call()

    call()
    hz = float(calibration["frequency_hz"])
    cohorts: tuple[list[float], list[float]] = ([], [])
    witnesses: list[dict[str, Any]] = []
    agreement: list[float] = []
    for sample in range(samples):
        for cohort in ((0, 1) if sample % 2 == 0 else (1, 0)):
            wall_start = time.perf_counter_ns()
            window = clock.measure(region, calibration)
            wall_end = time.perf_counter_ns()
            clocks: dict[str, Any] = dict(clock.witness_clocks(calibration, window))
            clocks["host_wall_ns"] = measured_clock(
                "host_wall_ns", source="perf_counter", value=wall_end - wall_start,
                provenance={"encloses": "the rdtscp/raw snapshots around the region"})
            clocks["perf_task_clock_ns"] = unavailable_clock(
                "perf_task_clock_ns", source="perf_event_task_clock",
                reason="not_collected_by_recorder",
                provenance={"detail": "record_x86_avx512_packet.py opens no perf_event"})
            witness = build_timing_sample(
                sample_id=f"{family}-kernel_wall-s{sample:02d}-c{cohort}",
                target="x86", clocks=clocks, artifact_digests=digests,
                batch_size=iterations, warm_state="warm",
                synchronization="synchronous_c_abi_return",
                execution_environment=environment,
                resources={"family": family, "measurement_cpu": measurement_cpu},
            )
            raw_ns = float(window["raw_end_ns"] - window["raw_start_ns"])
            tsc_ns = float(window["tsc_end"] - window["tsc_start"]) * 1.0e9 / hz
            error = abs(tsc_ns - raw_ns) / raw_ns
            if error > AGREEMENT_BAND:
                raise RuntimeError(
                    f"{family}: TSC disagrees with CLOCK_MONOTONIC_RAW by {error:.2%}")
            agreement.append(error)
            witnesses.append(witness)
            cohorts[cohort].append(tsc_ns / iterations)
    return [float(statistics.median(v)) for v in cohorts], witnesses, agreement


def _stability(run_medians: list[float]) -> float:
    return (max(run_medians) - min(run_medians)) / min(run_medians) * 100.0


def _spd(batch: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((batch, n, n))
    return np.ascontiguousarray(
        raw @ raw.transpose(0, 2, 1) + n * np.eye(n), dtype=np.float32)


def _family_definitions(fixtures: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Fixture module/bindings and timing module/bindings per family."""
    from tessera.compiler.x86_breadth import package_graph_breadth
    from tessera.compiler.x86_native import (
        X86_AVX512_ARCHITECTURE, package_attention, package_matmul,
        package_reduction, package_softmax,
    )

    def arrays(fixture_id: str) -> dict[str, np.ndarray]:
        return {name: np.ascontiguousarray(value, dtype=np.float32)
                for name, value in fixtures[fixture_id]["inputs"].items()}

    def unary(packager):
        return lambda module: packager(module, pipeline_name=PIPELINE,
                                       architecture=X86_AVX512_ARCHITECTURE)

    rng = np.random.default_rng(20260926)
    definitions: list[dict[str, Any]] = []

    x = arrays("matmul-f32-2x3x2-v1")
    ta = np.ascontiguousarray(rng.standard_normal((256, 256)), dtype=np.float32)
    tb = np.ascontiguousarray(rng.standard_normal((256, 256)), dtype=np.float32)
    definitions.append({
        "family": "matmul", "fixture_id": "matmul-f32-2x3x2-v1",
        "route": "schedule_tile_x86_avx512_c_abi",
        "package": lambda module: package_matmul(module, pipeline_name=PIPELINE),
        "module": _matmul_module(2, 3, 2),
        "bindings": {"a": x["a"], "b": x["b"], "o": np.zeros((2, 2), np.float32),
                     "M": 2, "N": 2, "K": 3},
        "timing_shape": [256, 256, 256],
        "timing_module": _matmul_module(256, 256, 256),
        "timing_bindings": {"a": ta, "b": tb, "o": np.zeros((256, 256), np.float32),
                            "M": 256, "N": 256, "K": 256},
    })

    x = arrays("softmax-f32-2x2-extreme-v1")
    tx = np.linspace(-1.0, 1.0, num=64 * 256, dtype=np.float32).reshape(64, 256)
    definitions.append({
        "family": "softmax", "fixture_id": "softmax-f32-2x2-extreme-v1",
        "route": "schedule_tile_x86_avx512_c_abi", "package": unary(package_softmax),
        "module": _softmax_module((2, 2)),
        "bindings": {"x": x["x"], "o": np.zeros((2, 2), np.float32), "Rows": 2, "K": 2},
        "timing_shape": [64, 256], "timing_module": _softmax_module((64, 256)),
        "timing_bindings": {"x": tx, "o": np.zeros_like(tx), "Rows": 64, "K": 256},
    })

    x = arrays("reduce-sum-f32-axis1-v1")
    definitions.append({
        "family": "reduction", "fixture_id": "reduce-sum-f32-axis1-v1",
        "route": "schedule_tile_x86_avx512_c_abi", "package": unary(package_reduction),
        "module": _reduction_module((2, 3)),
        "bindings": {"x": x["x"], "o": np.zeros((2,), np.float32),
                     "Outer": 2, "AxisExtent": 3, "Inner": 1},
        "timing_shape": [64, 256], "timing_module": _reduction_module((64, 256)),
        "timing_bindings": {"x": tx.copy(), "o": np.zeros((64,), np.float32),
                            "Outer": 64, "AxisExtent": 256, "Inner": 1},
    })

    x = arrays("attention-f32-uniform-1x1x2x2-v1")
    tq, tk, tv = (np.ascontiguousarray(rng.standard_normal((1, 4, 128, 64)), dtype=np.float32)
                  for _ in range(3))
    definitions.append({
        "family": "attention", "fixture_id": "attention-f32-uniform-1x1x2x2-v1",
        "route": "schedule_tile_x86_avx512_c_abi",
        "package": lambda module: package_attention(module, pipeline_name=PIPELINE),
        "module": _attention_module(1, 1, 2, 2, 2, 2),
        "bindings": {"q": x["q"], "k": x["k"], "v": x["v"],
                     "o": np.zeros((1, 1, 2, 2), np.float32),
                     "B": 1, "Hq": 1, "Hkv": 1, "Sq": 2, "Sk": 2, "D": 2, "Dv": 2},
        "timing_shape": [1, 4, 128, 128, 64, 64],
        "timing_module": _attention_module(1, 4, 128, 128, 64, 64),
        "timing_bindings": {"q": tq, "k": tk, "v": tv,
                            "o": np.zeros((1, 4, 128, 64), np.float32),
                            "B": 1, "Hq": 4, "Hkv": 4, "Sq": 128, "Sk": 128,
                            "D": 64, "Dv": 64},
    })

    x = arrays("cholesky-f32-3x3-spd-v1")
    tm = _spd(8, 64, 20260927)
    definitions.append({
        "family": "linalg", "fixture_id": "cholesky-f32-3x3-spd-v1",
        "route": "x86_breadth_graph_abi_c_abi",
        "package": lambda module: package_graph_breadth(module, pipeline_name=PIPELINE),
        "module": _cholesky_module((3, 3)),
        "bindings": {"matrix": x["matrix"], "result": np.zeros((3, 3), np.float32),
                     "Batch": 1, "N": 3},
        "timing_shape": [8, 64, 64], "timing_module": _cholesky_module((8, 64, 64)),
        "timing_bindings": {"matrix": tm, "result": np.zeros_like(tm), "Batch": 8, "N": 64},
    })
    return definitions


def _direct_call(family: str, package: Any, bindings: dict[str, Any]) -> Callable[[], object]:
    """The descriptor's C ABI called directly: the kernel_wall region."""
    from tessera import runtime as rt

    library = rt._load_x86_native_image(package.image)
    function = getattr(library, package.descriptor.entry_symbol)
    function.restype = None
    fp = ctypes.POINTER(ctypes.c_float)
    i64 = ctypes.c_int64

    def ptr(name: str):
        return bindings[name].ctypes.data_as(fp)

    if family == "matmul":
        function.argtypes = [fp, fp, i64, i64, i64, fp]
        args = (ptr("a"), ptr("b"), i64(bindings["M"]), i64(bindings["N"]),
                i64(bindings["K"]), ptr("o"))
    elif family == "softmax":
        function.argtypes = [fp, i64, i64, fp]
        args = (ptr("x"), i64(bindings["Rows"]), i64(bindings["K"]), ptr("o"))
    elif family == "reduction":
        function.argtypes = [fp, i64, i64, fp, ctypes.c_int]
        args = (ptr("x"), i64(bindings["Outer"]), i64(bindings["AxisExtent"]),
                ptr("o"), ctypes.c_int(0))
    elif family == "attention":
        provenance = package.descriptor.provenance
        function.argtypes = [fp, fp, fp] + [i64] * 5 + [ctypes.c_float, ctypes.c_int, fp]
        args = (ptr("q"), ptr("k"), ptr("v"), i64(bindings["B"] * bindings["Hq"]),
                i64(bindings["Sq"]), i64(bindings["Sk"]), i64(bindings["D"]),
                i64(bindings["Dv"]), ctypes.c_float(float(provenance["scale"])),
                ctypes.c_int(1 if provenance["causal"] else 0), ptr("o"))
    elif family == "linalg":
        if package.descriptor.provenance.get("returns_status"):
            raise RuntimeError("cholesky ABI unexpectedly returns a status")
        function.argtypes = [fp, i64, i64, fp]
        args = (ptr("matrix"), i64(bindings["Batch"]), i64(bindings["N"]), ptr("result"))
    else:  # pragma: no cover - definitions are closed above
        raise RuntimeError(f"no direct ABI for {family}")
    return lambda: function(*args)


def _timing_oracle(family: str, bindings: dict[str, Any]) -> np.ndarray:
    if family == "matmul":
        return bindings["a"].astype(np.float64) @ bindings["b"].astype(np.float64)
    if family == "softmax":
        x = bindings["x"].astype(np.float64)
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)
    if family == "reduction":
        return bindings["x"].astype(np.float64).sum(axis=-1)
    if family == "attention":
        q, k, v = (bindings[n].astype(np.float64) for n in ("q", "k", "v"))
        s = np.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(q.shape[-1])
        w = np.exp(s - s.max(axis=-1, keepdims=True))
        return np.einsum("bhqk,bhkd->bhqd", w / w.sum(axis=-1, keepdims=True), v)
    return np.linalg.cholesky(bindings["matrix"].astype(np.float64))


def record(*, samples: int, iterations: int, stability_limit: float,
           allow_dirty: bool = False) -> tuple[dict, dict]:
    from tessera import runtime as rt
    from tessera.compiler import profiler_x86_clock as clock
    from tessera.compiler.e2e_fleet import load_fixture_corpus, validate_backend_report
    from tessera.compiler.x86_native import (
        X86_AVX512_ARCHITECTURE, host_supports_architecture,
    )

    model, flags = _cpu_identity()
    architecture = host_architecture(model)
    if not host_supports_architecture(X86_AVX512_ARCHITECTURE):
        raise RuntimeError("this host lacks the AVX-512 feature set the image requires")
    if not clock.invariant_tsc():
        raise RuntimeError("the TSC witness requires constant_tsc and nonstop_tsc")
    if _git("status", "--porcelain").strip() and not allow_dirty:
        raise RuntimeError("source tree is dirty; commit before recording evidence")
    source_commit = _git("rev-parse", "HEAD").strip()
    library_record = _library_record()
    environment = execution_environment()
    fixtures = load_fixture_corpus()

    measurement_cpu = clock.pin_current_cpu()
    calibration = clock.calibrate()

    fixture_rows, cache_rows, benchmark_rows, resource_rows = [], [], [], []
    witness_rows: dict[str, Any] = {}
    toolchains: set[str] = set()
    for spec in _family_definitions(fixtures):
        family, fixture_id = spec["family"], spec["fixture_id"]
        cold, warm = spec["package"](spec["module"]), spec["package"](spec["module"])
        toolchains.add(cold.image.toolchain_fingerprint)
        if (cold.image.cache_key != warm.image.cache_key
                or cold.image.image_digest != warm.image.image_digest
                or cold.descriptor.descriptor_digest != warm.descriptor.descriptor_digest):
            raise RuntimeError(f"{family} prepackaged image/descriptor is not reproducible")
        artifact = rt.RuntimeArtifact(
            metadata={"target": "x86", "architecture": X86_AVX512_ARCHITECTURE},
            native_image=cold.image, launch_descriptor=cold.descriptor,
            tile_ir=cold.tile_ir, target_ir=cold.target_ir,
        )
        bindings = spec["bindings"]
        output_name = "result" if family == "linalg" else "o"
        launch_result = rt.launch(artifact, bindings)
        if not launch_result["ok"]:
            raise RuntimeError(f"{family}: {launch_result.get('reason')}")
        if launch_result.get("execution_kind") != "native_cpu":
            raise RuntimeError(f"{family} did not execute natively: {launch_result}")
        output = bindings[output_name]
        tolerance = fixtures[fixture_id]["tolerance"]
        np.testing.assert_allclose(
            output, np.asarray(fixtures[fixture_id]["oracle"], dtype=np.float32),
            rtol=tolerance["rtol"], atol=tolerance["atol"])

        timing = spec["package"](spec["timing_module"])
        timing_artifact = rt.RuntimeArtifact(
            metadata={"target": "x86", "architecture": X86_AVX512_ARCHITECTURE},
            native_image=timing.image, launch_descriptor=timing.descriptor,
            tile_ir=timing.tile_ir, target_ir=timing.target_ir,
        )
        timing_bindings = spec["timing_bindings"]
        timing_result = rt.launch(timing_artifact, timing_bindings)
        if not timing_result["ok"] or timing_result.get("execution_kind") != "native_cpu":
            raise RuntimeError(f"{family} timing shape did not execute natively")
        timing_error = float(np.max(np.abs(
            timing_bindings[output_name].astype(np.float64)
            - _timing_oracle(family, timing_bindings))))
        if not timing_error <= 2e-3:
            raise RuntimeError(f"{family} timing shape disagrees with its oracle ({timing_error})")
        digests = {"image": timing.image.image_digest,
                   "descriptor": timing.descriptor.descriptor_digest}
        kernel_medians, witnesses, agreement = _witnessed_medians_ns(
            _direct_call(family, timing, timing_bindings), family=family,
            samples=samples, iterations=iterations, calibration=calibration,
            digests=digests, environment=environment, measurement_cpu=measurement_cpu,
        )
        run_medians = {
            "kernel_wall": kernel_medians,
            "end_to_end": _two_run_medians_ns(
                lambda: rt.launch(timing_artifact, timing_bindings),
                samples=samples, iterations=iterations),
        }
        resource = {
            "architecture": architecture,
            "entry": timing.descriptor.entry_symbol,
            "abi": timing.descriptor.abi_id,
            "image_digest": timing.image.image_digest,
            "timing_shape": spec["timing_shape"],
            "instruction_envelope": "x86-64 AVX-512 (avx512f/bw/cd/dq/vl/vnni/bf16/vpopcntdq)",
            "runtime_library_level": library_record["libraries"][REQUIRED_LIBRARY],
        }
        resource_fingerprint = hashlib.sha256(
            json.dumps(resource, sort_keys=True).encode()).hexdigest()
        for domain, medians in run_medians.items():
            benchmark_rows.append({
                "family": family, "route": spec["route"], "timing_domain": domain,
                "timing_source": ("tsc_rdtscp_witnessed_by_clock_monotonic_raw"
                                  if domain == "kernel_wall" else "perf_counter_ns"),
                "median_ns": float(statistics.median(medians)),
                "run_medians_ns": medians, "stability_limit_pct": stability_limit,
                "stable": _stability(medians) <= stability_limit,
                "selected": True, "repetitions": samples * iterations,
                "warmups": 1, "discard_first": True,
                "resource_fingerprint": resource_fingerprint,
            })
        fixture_rows.append({
            "fixture_id": fixture_id,
            "levels": {"a": "proven", "b": "proven", "c": "proven"},
            "actual": output.tolist(), "image_digest": cold.image.image_digest,
            "descriptor_digest": cold.descriptor.descriptor_digest,
        })
        state = {"compile_state": "prepackaged", "cache_key": cold.image.cache_key,
                 "image_digest": cold.image.image_digest,
                 "descriptor_digest": cold.descriptor.descriptor_digest}
        cache_rows.append({"fixture_id": fixture_id, "cold": state, "warm": dict(state)})
        resource_rows.append({"family": family, "resource_fingerprint": resource_fingerprint,
                              "timing_max_abs_error": timing_error, **resource})
        witness_rows[family] = {
            "timing_domain": "kernel_wall",
            "agreement": {"band": AGREEMENT_BAND, "min_relative_error": min(agreement),
                          "max_relative_error": max(agreement), "windows": len(agreement)},
            "samples": witnesses,
        }

    toolchain = hashlib.sha256("\n".join(sorted(toolchains)).encode()).hexdigest()
    report = {
        "schema": "tessera.e2e-backend-report.v1",
        "target": "x86", "architecture": architecture,
        "device": {"exact": True, "identity": f"{platform.node()} | {model}"},
        "source_commit": source_commit, "toolchain_fingerprint": toolchain,
        "scope": [spec["family"] for spec in _family_definitions(fixtures)],
        "required_timing_domains": ["kernel_wall", "end_to_end"],
        "fixtures": fixture_rows, "cache_proofs": cache_rows,
        "benchmarks": benchmark_rows,
    }
    validate_backend_report(report)
    resources = {
        "schema": "tessera.e2e-x86-resource-record.v1",
        "device": {"model": model, "flags": flags, "host": platform.node(),
                   "kernel_release": platform.release()},
        "execution_environment": environment,
        "measurement_cpu": measurement_cpu,
        "runtime_library_build": library_record,
        "tsc_calibration": calibration,
        "timing_witness": witness_rows,
        "rows": resource_rows,
    }
    return report, resources


def main() -> int:
    from tessera.compiler.e2e_fleet import seal_packet

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--stability-limit", type=float, default=4.0)
    parser.add_argument("--packet-dir", type=Path)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="dry runs only; a sealed packet must name a clean commit")
    args = parser.parse_args()
    if args.samples < 2 or args.iterations < 10:
        parser.error("use at least two samples and ten amortized iterations")
    if args.packet_dir and args.allow_dirty:
        parser.error("--allow-dirty cannot seal a packet")
    report, resources = record(
        samples=args.samples, iterations=args.iterations,
        stability_limit=args.stability_limit, allow_dirty=args.allow_dirty,
    )
    if args.packet_dir:
        if args.packet_dir.name != report["architecture"]:
            raise SystemExit(
                f"packet dir {args.packet_dir} does not name this host's architecture "
                f"{report['architecture']}")
        args.packet_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in (("report.json", report), ("resources.json", resources)):
            (args.packet_dir / name).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        seal_packet(args.packet_dir)
        print(f"sealed {args.packet_dir}")
    else:
        summary = {row["family"] + "/" + row["timing_domain"]: row["median_ns"]
                   for row in report["benchmarks"]}
        print(json.dumps({"architecture": report["architecture"], "medians_ns": summary,
                          "agreement": {k: v["agreement"] for k, v in
                                        resources["timing_witness"].items()}},
                         indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
