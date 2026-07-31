"""Record the SM120 CUDA DeltaNet-backward decision packet.

All rows execute the physical v2 CUDA package and compare every requested
gradient with the shared analytic oracle.  The recorder deliberately reports
the plain, affine, and serial-fill flag paths separately; it does not change a
selector or execution-matrix row.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

from tessera.autodiff.vjp import get_vjp
from tessera.compiler.nvidia_training import package_deltanet_backward
from tessera.runtime import (
    _nvidia_native_descriptor_device_latency,
    _nvidia_native_descriptor_resources,
    _submit_nvidia_sm120_native,
)
from tests._support.nvidia import nvidia_cuda_host_ready


_NCU_METRICS = ("lts__t_sector_hit_rate.pct", "dram__bytes.sum")
_VARIANTS = ("plain", "affine", "serial_fill")
_F32_INPUT_SCALE = 0.05


def _shape(text: str) -> tuple[int, int, int, int, int]:
    values = tuple(int(value) for value in text.split("x"))
    if len(values) != 5 or min(values) < 1:
        raise argparse.ArgumentTypeError("shapes must be positive BxHxSxDqkxDv")
    if values[3] > 8 or values[4] > 8:
        raise argparse.ArgumentTypeError("the validated v2 package requires Dqk,Dv <= 8")
    return values


def _ncu_report_spec(text: str) -> tuple[str, Path]:
    try:
        variant, raw_path = text.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("NCU report is variant=/path/report.ncu-rep") from exc
    if variant not in _VARIANTS:
        raise argparse.ArgumentTypeError(f"unknown DeltaNet variant {variant!r}")
    path = Path(raw_path)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"NCU report does not exist: {path}")
    return variant, path


def _ncu_counters(report: Path, *, ncu_bin: str) -> dict[str, float]:
    completed = subprocess.run(
        [ncu_bin, "--import", str(report), "--csv", "--page", "raw", "--print-units", "base"],
        check=False, capture_output=True, text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"NCU import failed: {completed.stderr.strip()}")
    lines = completed.stdout.splitlines()
    start = next((i for i, line in enumerate(lines)
                  if line.startswith('"ID",') and '"Kernel Name"' in line), None)
    if start is None:
        raise RuntimeError("NCU did not produce a CSV metric table")
    values: dict[str, list[float]] = {metric: [] for metric in _NCU_METRICS}
    for row in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        if "tessera_cuda_training_deltanet_backward_f32_v2" not in row.get("Kernel Name", ""):
            continue
        metric = row.get("Metric Name", "").strip()
        value = row.get("Metric Value", "").replace(",", "").replace("%", "").strip()
        if metric in values and value:
            values[metric].append(float(value))
        elif not metric:
            for metric in values:
                value = row.get(metric, "").replace(",", "").replace("%", "").strip()
                if value:
                    values[metric].append(float(value))
    missing = [metric for metric, samples in values.items() if not samples]
    if missing:
        raise RuntimeError("NCU DeltaNet counters missing: " + ", ".join(missing))
    return {
        "l2_sector_hit_rate_pct": statistics.median(values[_NCU_METRICS[0]]),
        "dram_bytes": statistics.median(values[_NCU_METRICS[1]]),
    }


def _inputs(shape: tuple[int, int, int, int, int], *, seed: int) -> dict[str, np.ndarray]:
    b, h, s, dqk, dv = shape
    rng = np.random.default_rng(seed)
    q = np.ascontiguousarray((rng.normal(size=(b, h, s, dqk)) * _F32_INPUT_SCALE).astype(np.float32))
    k = np.ascontiguousarray((rng.normal(size=q.shape) * _F32_INPUT_SCALE).astype(np.float32))
    v = np.ascontiguousarray((rng.normal(size=(b, h, s, dv)) * _F32_INPUT_SCALE).astype(np.float32))
    return {
        "q": q, "k": k, "v": v,
        "gate": np.ascontiguousarray(rng.normal(size=v.shape).astype(np.float32)),
        "beta": np.ascontiguousarray(rng.uniform(.2, .9, size=(b, h, s)).astype(np.float32)),
        "decay": np.ascontiguousarray(rng.uniform(.7, .99, size=(b, h, s)).astype(np.float32)),
        "dy": np.ascontiguousarray(rng.normal(size=v.shape).astype(np.float32)),
    }


def _variant_args(
    values: dict[str, np.ndarray], shape: tuple[int, int, int, int, int], variant: str,
) -> tuple[dict[str, np.ndarray], dict[str, int], tuple[np.ndarray, ...]]:
    b, h, s, dqk, dv = shape
    affine = variant != "plain"
    serial = variant == "serial_fill"
    buffers = {
        **values,
        "dq": np.empty_like(values["q"]), "dk": np.empty_like(values["k"]),
        "dv": np.empty_like(values["v"]), "dgate": np.empty_like(values["v"]),
        "dbeta": np.empty((b, h, s), np.float32), "ddecay": np.empty((b, h, s), np.float32),
    }
    scalars = {
        "B": b, "H": h, "S": s, "Dqk": dqk, "Dv": dv,
        "HasGate": int(affine), "HasBeta": int(affine), "HasDecay": int(affine),
        "Erase": int(serial), "Modified": int(serial),
    }
    if variant == "plain":
        expected = get_vjp("gated_deltanet")(values["dy"], values["q"], values["k"], values["v"])
    elif variant == "affine":
        expected = get_vjp("gated_deltanet")(values["dy"], values["q"], values["k"], values["v"], values["gate"], values["beta"], values["decay"])
    else:
        expected = get_vjp("modified_delta_attention")(
            values["dy"], values["q"], values["k"], values["v"], values["gate"],
            values["beta"], values["decay"], erase=True,
        )
    return buffers, scalars, expected


def _invoke(package, buffers: dict[str, np.ndarray], scalars: dict[str, int]) -> tuple[np.ndarray, ...]:
    return _submit_nvidia_sm120_native(package.image, package.descriptor, buffers, scalars, None)


def _max_error(actual: tuple[np.ndarray, ...], expected: tuple[np.ndarray, ...], variant: str) -> float:
    count = 3 if variant == "plain" else 6
    values = [float(np.max(np.abs(got - reference)))
              for got, reference in zip(actual[:count], expected, strict=True)]
    if variant == "plain":
        values.extend(float(np.max(np.abs(unused))) for unused in actual[3:])
    return max(values, default=0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", type=_shape, nargs="+", default=((1, 1, 16, 8, 8), (1, 2, 16, 8, 8)))
    parser.add_argument("--variants", choices=_VARIANTS, nargs="+", default=_VARIANTS)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--ncu-report", action="append", type=_ncu_report_spec)
    parser.add_argument("--ncu-bin", default="/usr/local/cuda/bin/ncu")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not nvidia_cuda_host_ready():
        print(json.dumps({"status": "skipped", "reason": "host WSL CUDA SM120 unavailable"}))
        return 0
    if args.profile_only and (len(args.shapes) != 1 or len(args.variants) != 1):
        raise ValueError("--profile-only requires exactly one shape and one variant")
    if not args.profile_only and (args.output is None or args.runs < 2 or args.reps < 1 or args.warmup < 0):
        raise ValueError("packet mode requires --output, two runs, positive reps, and nonnegative warmup")
    reports: dict[str, list[Path]] = {}
    for variant, report in args.ncu_report or ():
        reports.setdefault(variant, []).append(report)
    package = package_deltanet_backward()
    rows: list[dict[str, object]] = []
    for shape_index, shape in enumerate(args.shapes):
        values = _inputs(shape, seed=734_000 + shape_index)
        for variant in args.variants:
            buffers, scalars, expected = _variant_args(values, shape, variant)
            actual = _invoke(package, buffers, scalars)
            if args.profile_only:
                continue
            error = _max_error(actual, expected, variant)
            if error > 5e-3:
                raise RuntimeError(f"{shape}/{variant} max gradient error {error} exceeds 0.005")
            launch_args = {**buffers, **scalars}
            device = [float(_nvidia_native_descriptor_device_latency(
                package.image, package.descriptor, launch_args,
                reps=args.reps, warmup=args.warmup,
            )) for _ in range(args.runs)]
            end_to_end: list[float] = []
            for _ in range(args.runs):
                start = time.perf_counter()
                for _ in range(args.reps):
                    _invoke(package, buffers, scalars)
                end_to_end.append((time.perf_counter() - start) * 1e3 / args.reps)
            row: dict[str, object] = {
                "shape": list(shape), "variant": variant, "max_abs_gradient_error": error,
                "device_event_ms": device, "device_event_median_ms": statistics.median(device),
                "end_to_end_ms": end_to_end, "end_to_end_median_ms": statistics.median(end_to_end),
                "resources": _nvidia_native_descriptor_resources(package.image, package.descriptor, block_size=128),
            }
            if variant in reports:
                ncu_samples = [
                    _ncu_counters(report, ncu_bin=args.ncu_bin)
                    for report in reports[variant]
                ]
                row["ncu_samples"] = ncu_samples
                row["ncu_reports"] = [report.name for report in reports[variant]]
                row["ncu"] = {
                    metric: statistics.median(sample[metric] for sample in ncu_samples)
                    for metric in ("l2_sector_hit_rate_pct", "dram_bytes")
                }
            rows.append(row)
    if args.profile_only:
        return 0
    args.output.write_text(json.dumps({
        "schema": "tessera.nvidia.deltanet_backward.packet.v1", "work_item": "NVIDIA-DeltaNet-backward",
        "device": "nvidia:sm_120", "package": package.descriptor.entry_symbol,
        "abi_id": package.descriptor.abi_id, "bounded_dims": {"Dqk": 8, "Dv": 8},
        "method": {"timing_domains": ["cuda_event", "end_to_end"], "runs": args.runs,
                   "repetitions": args.reps, "warmup": args.warmup,
                   "ncu_metrics": list(_NCU_METRICS), "f32_input_scale": _F32_INPUT_SCALE,
                   "selector_changed": False},
        "rows": rows,
    }, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
