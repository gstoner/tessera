#!/usr/bin/env python3
"""Record exact-host E2E-SPINE-3 packets for the Apple fleet identities.

`apple_gpu`/`apple7` and `apple_cpu`/`apple_m1_max` are two *independent*
registrations. Exact-device evidence never transfers between them, so each
`--lane` produces its own report/resources/manifest triple under its own packet
directory. Recording one does not close the other.

Both lanes report `kernel_wall`, not `device_event`. The Apple GPU runtime does
expose a command-buffer device timer, but only the tile/dispatch lane feeds it:
after a value-descriptor launch `tessera_apple_gpu_last_dispatch_device_time_ns`
still reads `-1` and the timing source reads `0`. Claiming `device_event` from a
host clock would be a fabricated domain, so this recorder measures the native
submit boundary and records the reason in `resources.json` rather than leaving a
reader to guess why the stronger domain is absent.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]


@dataclass(frozen=True)
class FamilyPlan:
    """One family's fixture proof plus the larger shape it is timed at."""

    family: str
    fixture_id: str
    route: str
    op_name: str
    # Fixture operands are reshaped to this rank before packaging; `None` keeps
    # the corpus shape. The Apple GPU GEMM contract is batched-only, so the 2-D
    # matmul fixture is proven as a batch-1 BMM.
    batched: bool
    timing_shape: tuple[int, ...]


APPLE_GPU_PLANS = (
    FamilyPlan(
        "matmul", "matmul-f32-2x3x2-v1", "apple_gpu_bmm_f32_batch1",
        "tessera.matmul", True, (1, 256, 256, 256),
    ),
    FamilyPlan(
        "softmax", "softmax-f32-2x2-extreme-v1", "apple_gpu_msl_softmax_f32",
        "tessera.softmax", False, (64, 256),
    ),
)

# `apple_cpu` registers only ("matmul", "linalg"). Adding a softmax row here
# would be rejected as an undeclared family even though the lane executes it.
APPLE_CPU_PLANS = (
    FamilyPlan(
        "matmul", "matmul-f32-2x3x2-v1", "apple_cpu_accelerate_gemm_f32",
        "tessera.matmul", False, (256, 256, 256),
    ),
)


@dataclass(frozen=True)
class Lane:
    target: str
    architecture: str
    pipeline: str
    plans: tuple[FamilyPlan, ...]
    packet_subdir: str


LANES: dict[str, Lane] = {
    "apple_gpu": Lane(
        "apple_gpu", "apple7", "tessera-lower-to-apple_gpu", APPLE_GPU_PLANS,
        "apple_gpu/apple7",
    ),
    "apple_cpu": Lane(
        "apple_cpu", "apple_m1_max", "tessera-lower-to-apple_cpu", APPLE_CPU_PLANS,
        "apple_cpu/apple_m1_max",
    ),
}


def _tensor_type(shape: Sequence[int]):
    from tessera.compiler.graph_ir import IRType

    dims = "x".join(str(int(dim)) for dim in shape)
    return IRType(f"tensor<{dims}xf32>", tuple(str(int(d)) for d in shape), "fp32")


def _matmul_module(a_shape: Sequence[int], b_shape: Sequence[int],
                   o_shape: Sequence[int]):
    from tessera.compiler.graph_ir import (
        GraphIRFunction, GraphIRModule, IRArg, IROp,
    )

    a, b, o = _tensor_type(a_shape), _tensor_type(b_shape), _tensor_type(o_shape)
    return GraphIRModule(functions=[GraphIRFunction(
        name="fleet_matmul", args=[IRArg("a", a), IRArg("b", b)],
        result_types=[o],
        body=[IROp(
            result="o", op_name="tessera.matmul", operands=["%a", "%b"],
            operand_types=[str(a), str(b)], result_type=str(o), kwargs={},
        )],
        return_values=["%o"],
    )])


def _softmax_module(shape: Sequence[int]):
    from tessera.compiler.graph_ir import (
        GraphIRFunction, GraphIRModule, IRArg, IROp,
    )

    x = _tensor_type(shape)
    return GraphIRModule(functions=[GraphIRFunction(
        name="fleet_softmax", args=[IRArg("x", x)], result_types=[x],
        body=[IROp(
            result="o", op_name="tessera.softmax", operands=["%x"],
            operand_types=[str(x)], result_type=str(x), kwargs={"axis": -1},
        )],
        return_values=["%o"],
    )])


def _module_for(plan: FamilyPlan, shape: Sequence[int]):
    """Build the Graph IR for one family at either the fixture or timing shape."""
    if plan.op_name == "tessera.softmax":
        return _softmax_module(shape)
    if plan.batched:
        batch, m, n, k = shape
        return _matmul_module((batch, m, k), (batch, k, n), (batch, m, n))
    m, n, k = shape
    return _matmul_module((m, k), (k, n), (m, n))


def _fixture_bindings(plan: FamilyPlan, fixture: dict) -> tuple[dict, np.ndarray, tuple]:
    """Bind the corpus fixture, reshaping only where the lane's ABI demands it."""
    inputs = fixture["inputs"]
    if plan.op_name == "tessera.softmax":
        x = np.asarray(inputs["x"], dtype=np.float32)
        out = np.zeros_like(x)
        return {"x": x, "o": out}, out, tuple(x.shape)
    a = np.asarray(inputs["a"], dtype=np.float32)
    b = np.asarray(inputs["b"], dtype=np.float32)
    if plan.batched:
        a, b = a.reshape(1, *a.shape), b.reshape(1, *b.shape)
        out = np.zeros((1, a.shape[1], b.shape[2]), dtype=np.float32)
    else:
        out = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)
    return {"a": a, "b": b, "o": out}, out, (a.shape, b.shape)


def _timing_bindings(plan: FamilyPlan) -> tuple[dict, np.ndarray]:
    shape = plan.timing_shape
    if plan.op_name == "tessera.softmax":
        rows, columns = shape
        x = np.linspace(-1.0, 1.0, rows * columns, dtype=np.float32).reshape(rows, columns)
        out = np.zeros_like(x)
        return {"x": x, "o": out}, out
    if plan.batched:
        batch, m, n, k = shape
        a = np.linspace(-1.0, 1.0, batch * m * k, dtype=np.float32).reshape(batch, m, k)
        b = np.linspace(1.0, -1.0, batch * k * n, dtype=np.float32).reshape(batch, k, n)
        out = np.zeros((batch, m, n), dtype=np.float32)
    else:
        m, n, k = shape
        a = np.linspace(-1.0, 1.0, m * k, dtype=np.float32).reshape(m, k)
        b = np.linspace(1.0, -1.0, k * n, dtype=np.float32).reshape(k, n)
        out = np.zeros((m, n), dtype=np.float32)
    return {"a": a, "b": b, "o": out}, out


def _packager(lane: Lane) -> Callable[..., Any]:
    if lane.target == "apple_gpu":
        from tessera.compiler import apple_native

        return apple_native.package_native
    from tessera.compiler import apple_cpu_native

    return apple_cpu_native.package_native


def _two_run_medians_ns(
    call: Callable[[], object], *, samples: int, iterations: int,
) -> list[float]:
    """Two interleaved cohorts, so a monotonic thermal drift cannot read as stable."""
    call()  # First-use compile/cache/loader effects never enter the sample.
    cohorts: tuple[list[float], list[float]] = ([], [])
    for sample in range(samples):
        order = (0, 1) if sample % 2 == 0 else (1, 0)
        for cohort in order:
            started = time.perf_counter_ns()
            for _ in range(iterations):
                call()
            cohorts[cohort].append((time.perf_counter_ns() - started) / iterations)
    return [float(statistics.median(values)) for values in cohorts]


def _stability(run_medians: Sequence[float]) -> float:
    return (max(run_medians) - min(run_medians)) / min(run_medians) * 100.0


def _host_identity() -> tuple[str, str]:
    """Return (hardware model, CPU brand) from sysctl, refusing a silent guess."""
    def sysctl(key: str) -> str:
        result = subprocess.run(
            ["sysctl", "-n", key], check=False, capture_output=True, text=True,
        )
        value = result.stdout.strip()
        if result.returncode != 0 or not value:
            raise RuntimeError(f"cannot read host identity: sysctl -n {key} failed")
        return value

    return sysctl("hw.model"), sysctl("machdep.cpu.brand_string")


def _require_apple_silicon_host(lane: Lane) -> tuple[str, str]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError(
            f"{lane.target} packets must be recorded on an Apple Silicon macOS host; "
            f"this is {platform.system()}/{platform.machine()}"
        )
    model, brand = _host_identity()
    if lane.architecture == "apple_m1_max" and "M1 Max" not in brand:
        raise RuntimeError(
            f"apple_cpu/apple_m1_max is an exact-host identity; this host reports {brand!r}"
        )
    return model, brand


def _native_submit(package: Any, bindings: dict) -> Callable[[], object]:
    """Bind one validated native submit, the boundary `kernel_wall` measures.

    Argument splitting, contract validation, launcher resolution, and result
    packaging are hoisted out of the loop; only the descriptor-consuming submit
    is timed.
    """
    from tessera import runtime as rt

    values, contracts, scalars = rt._split_native_arguments(package.descriptor, bindings)
    package.descriptor.validate_invocation(package.image, contracts, scalars)
    rt._ensure_builtin_native_launcher(package.image.target, package.descriptor.abi_id)
    registration = rt._native_launchers.get(package.image.target)
    if registration is None:
        raise RuntimeError(
            f"no native launcher is registered for {package.image.target}; "
            "kernel_wall cannot be measured without one"
        )
    return lambda: registration.submit(
        package.image, package.descriptor, values, scalars, None,
    )


def _gpu_resource_evidence(source_sha256: str) -> dict[str, Any]:
    """Metal-native resource evidence, with absent fields carrying their reason."""
    from tessera.compiler.apple_counter_evidence import (
        apple_autotune_evidence, probe_apple_profiling_capabilities,
    )

    capabilities = probe_apple_profiling_capabilities()
    if capabilities is None:
        raise RuntimeError(
            "the Apple GPU runtime reported no profiling capability matrix; "
            "an apple_gpu packet must be recorded on a live Metal host"
        )
    from tessera import runtime as rt

    runtime = rt._load_apple_gpu_runtime()
    cache_probe = getattr(runtime, "tessera_apple_gpu_runtime_msl_cache_size", None)
    if cache_probe is not None:
        cache_probe.restype = ctypes.c_int32
    evidence = apple_autotune_evidence(
        capabilities=capabilities,
        pipeline_cache_size=int(cache_probe()) if cache_probe is not None else None,
        cold_compile=False,
        source_sha256=source_sha256,
    )
    return {
        "capability_matrix": capabilities.raw,
        "evidence": evidence,
        "device_event_available": False,
        "device_event_reason": (
            "the value-descriptor C ABI lane does not feed "
            "tessera_apple_gpu_last_dispatch_device_time_ns (it reads -1 and the "
            "timing source reads 0 after a descriptor launch); only the "
            "tile/dispatch lane populates it"
        ),
    }


def record(
    lane: Lane, *, samples: int, iterations: int, stability_limit: float,
) -> tuple[dict, dict]:
    from tessera import runtime as rt
    from tessera.compiler.e2e_fleet import (
        FLEET_REGISTRATIONS, load_fixture_corpus, validate_backend_report,
    )

    model, brand = _require_apple_silicon_host(lane)
    fixtures = load_fixture_corpus()
    registration = next(
        (r for r in FLEET_REGISTRATIONS
         if r.target == lane.target and r.architecture == lane.architecture),
        None,
    )
    if registration is None:
        raise RuntimeError(f"{lane.target}/{lane.architecture} is not a fleet registration")
    undeclared = {plan.family for plan in lane.plans} - set(registration.families)
    if undeclared:
        raise RuntimeError(
            f"{lane.target} plans claim undeclared families: {', '.join(sorted(undeclared))}"
        )

    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    package_native = _packager(lane)

    fixture_rows: list[dict] = []
    cache_rows: list[dict] = []
    benchmark_rows: list[dict] = []
    resource_rows: list[dict] = []
    toolchains: set[str] = set()

    for plan in lane.plans:
        fixture = fixtures[plan.fixture_id]
        bindings, output, operand_shapes = _fixture_bindings(plan, fixture)
        module = _module_for(
            plan,
            _fixture_shape(plan, fixture),
        )
        cold = package_native(module, pipeline_name=lane.pipeline)
        warm = package_native(module, pipeline_name=lane.pipeline)
        toolchains.add(str(cold.image.toolchain_fingerprint))
        if (cold.image.cache_key != warm.image.cache_key
                or cold.image.image_digest != warm.image.image_digest
                or cold.descriptor.descriptor_digest != warm.descriptor.descriptor_digest):
            raise RuntimeError(
                f"{plan.family} prepackaged image/descriptor is not reproducible"
            )

        artifact = rt.RuntimeArtifact(
            metadata={"target": lane.target, "architecture": lane.architecture},
            native_image=cold.image, launch_descriptor=cold.descriptor,
            tile_ir=cold.tile_ir, target_ir=cold.target_ir,
        )
        result = rt.launch(artifact, bindings)
        if not result["ok"]:
            raise RuntimeError(
                f"{plan.family} fixture launch failed: {result.get('reason')}"
            )
        oracle = np.asarray(fixture["oracle"], dtype=np.float32)
        tolerance = fixture["tolerance"]
        np.testing.assert_allclose(
            output.reshape(oracle.shape), oracle,
            rtol=float(tolerance["rtol"]), atol=float(tolerance["atol"]),
        )

        timing_package = package_native(
            _module_for(plan, plan.timing_shape), pipeline_name=lane.pipeline,
        )
        timing_bindings, _ = _timing_bindings(plan)
        timing_artifact = rt.RuntimeArtifact(
            metadata={"target": lane.target, "architecture": lane.architecture},
            native_image=timing_package.image,
            launch_descriptor=timing_package.descriptor,
            tile_ir=timing_package.tile_ir, target_ir=timing_package.target_ir,
        )
        submit = _native_submit(timing_package, timing_bindings)
        run_medians = {
            "kernel_wall": _two_run_medians_ns(
                submit, samples=samples, iterations=iterations,
            ),
            "end_to_end": _two_run_medians_ns(
                lambda: rt.launch(timing_artifact, timing_bindings),
                samples=samples, iterations=iterations,
            ),
        }
        resource_fingerprint = hashlib.sha256(json.dumps({
            "target": lane.target, "architecture": lane.architecture,
            "entry": timing_package.descriptor.entry_symbol,
            "image_digest": timing_package.image.image_digest,
            "timing_shape": list(plan.timing_shape),
            "route": plan.route,
        }, sort_keys=True).encode()).hexdigest()

        for domain, medians in run_medians.items():
            benchmark_rows.append({
                "family": plan.family, "route": plan.route,
                "timing_domain": domain,
                "median_ns": float(statistics.median(medians)),
                "run_medians_ns": medians,
                "stability_limit_pct": stability_limit,
                "stable": _stability(medians) <= stability_limit,
                "selected": True, "repetitions": samples * iterations,
                "warmups": 1, "discard_first": True,
                "resource_fingerprint": resource_fingerprint,
            })
        fixture_rows.append({
            "fixture_id": plan.fixture_id,
            "levels": {"a": "proven", "b": "proven", "c": "proven"},
            "actual": output.tolist(),
            "image_digest": cold.image.image_digest,
            "descriptor_digest": cold.descriptor.descriptor_digest,
        })
        state = {
            "compile_state": "prepackaged", "cache_key": cold.image.cache_key,
            "image_digest": cold.image.image_digest,
            "descriptor_digest": cold.descriptor.descriptor_digest,
        }
        cache_rows.append({
            "fixture_id": plan.fixture_id, "cold": state, "warm": dict(state),
        })
        resource_rows.append({
            "family": plan.family, "route": plan.route,
            "resource_fingerprint": resource_fingerprint,
            "entry": timing_package.descriptor.entry_symbol,
            "image_digest": timing_package.image.image_digest,
            "fixture_operand_shapes": [list(s) for s in operand_shapes]
            if isinstance(operand_shapes[0], tuple) else [list(operand_shapes)],
            "timing_shape": list(plan.timing_shape),
        })

    toolchain = hashlib.sha256("\n".join(sorted(toolchains)).encode()).hexdigest()
    report = {
        "schema": "tessera.e2e-backend-report.v1",
        "target": lane.target, "architecture": lane.architecture,
        "device": {"exact": True, "identity": f"{platform.node()} | {model} | {brand}"},
        "source_commit": source_commit, "toolchain_fingerprint": toolchain,
        "scope": [plan.family for plan in lane.plans],
        "required_timing_domains": ["kernel_wall", "end_to_end"],
        "fixtures": fixture_rows, "cache_proofs": cache_rows,
        "benchmarks": benchmark_rows,
    }
    validate_backend_report(report)

    resources: dict[str, Any] = {
        "schema": "tessera.e2e-apple-resource-record.v1",
        "lane": lane.target,
        "device": {"model": model, "cpu_brand": brand, "os": platform.mac_ver()[0]},
        "timing_domains": {
            "kernel_wall": (
                "descriptor-consuming native submit; argument splitting, contract "
                "validation, launcher resolution, and result packaging are hoisted "
                "out of the timed region"
            ),
            "end_to_end": "tessera.runtime.launch, including all of the above",
        },
        "rows": resource_rows,
    }
    if lane.target == "apple_gpu":
        resources["metal"] = _gpu_resource_evidence(toolchain)
    else:
        resources["metal"] = {
            "applicable": False,
            "reason": (
                "apple_cpu executes through Accelerate on the CPU; the Metal "
                "profiling capability matrix describes the GPU lane and does not "
                "apply here"
            ),
        }
    return report, resources


def _fixture_shape(plan: FamilyPlan, fixture: dict) -> tuple[int, ...]:
    """The corpus fixture's own shape, in this plan's module convention."""
    shape = tuple(int(dim) for dim in fixture["shape"])
    if plan.op_name == "tessera.softmax":
        return shape
    m, k, n = shape
    return (1, m, n, k) if plan.batched else (m, n, k)


def main(argv: Sequence[str] | None = None) -> int:
    from tessera.compiler.e2e_fleet import EVIDENCE_ROOT, seal_packet

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=sorted(LANES), required=True)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--stability-limit", type=float, default=4.0,
        help="matches the x86 lane; both Apple lanes measure well inside it, and "
             "an unstable run fails rather than being admitted at a looser bound",
    )
    parser.add_argument(
        "--packet-dir", type=Path,
        help="defaults to the registered evidence directory for the lane",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="print the report instead of sealing a packet")
    args = parser.parse_args(argv)
    if args.samples < 2 or args.iterations < 10:
        parser.error("use at least two samples and ten amortized iterations")

    lane = LANES[args.lane]
    report, resources = record(
        lane, samples=args.samples, iterations=args.iterations,
        stability_limit=args.stability_limit,
    )
    if args.dry_run:
        print(json.dumps({"report": report, "resources": resources},
                         indent=2, sort_keys=True))
        return 0

    packet_dir = args.packet_dir or (EVIDENCE_ROOT / lane.packet_subdir)
    packet_dir.mkdir(parents=True, exist_ok=True)
    (packet_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (packet_dir / "resources.json").write_text(
        json.dumps(resources, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    seal_packet(packet_dir)
    print(f"sealed {packet_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
