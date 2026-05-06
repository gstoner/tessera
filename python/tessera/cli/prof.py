"""Profiling command for Tessera Python sources."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Sequence

from tessera import autotune
from tessera import profiler


DEFAULT_METRICS = ("latency", "flops", "bandwidth")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    source = Path(args.model)
    try:
        text = source.read_text()
        symbol_count = _count_symbols(text)
    except OSError as exc:
        parser.error(str(exc))
    except SyntaxError as exc:
        parser.error(f"{source}: syntax error at line {exc.lineno}: {exc.msg}")

    metrics = _parse_metrics(args.metrics)
    with profiler.session() as sess:
        sess.record(
            "model.inspect",
            latency_ms=max(0.01, symbol_count * 0.05),
            flops=1.0e9 if "flops" in metrics else 0.0,
            bytes_moved=512.0e6 if "bandwidth" in metrics else 0.0,
            peak_tflops=2.0 if "flops" in metrics else None,
            counters=_counters(metrics, symbol_count),
            op="model.inspect",
            kernel_id=source.stem,
            status="ok",
            metadata={
                "source_path": str(source),
                "requested_metrics": sorted(metrics),
                "profile_mode": "source_inspection",
            },
        )
        artifact = None
        if args.autotune:
            M, N, K = _parse_shape(args.shapes)
            result = autotune(
                "matmul",
                shapes=(M, N, K),
                arch=args.compile_target,
                method=args.autotune_method,
                max_trials=args.max_trials,
                cache_path=args.cache,
            )
            artifact = autotune.schedule_artifact(
                result,
                op="matmul",
                shapes=(M, N, K),
                arch=args.compile_target,
            )
            sess.record(
                "autotune.matmul",
                latency_ms=result.latency_ms,
                flops=2 * M * N * K,
                bytes_moved=4 * (M * K + K * N + M * N),
                op="matmul",
                shape={"M": M, "N": N, "K": K},
                dtype="bf16",
                arch=args.compile_target,
                schedule_hash=str(artifact["hash"]),
                kernel_id="gemm",
                status=result.status,
                metadata={"reason": result.reason, "artifact": artifact},
            )
            if args.artifact:
                Path(args.artifact).write_text(json.dumps(artifact, indent=2, sort_keys=True))
        if args.trace:
            sess.timeline(args.trace)
        report = sess.report()

    payload = {
        **sess.to_dict(),
        "mode": "source_inspection",
        "source": str(source),
        "compile_target": args.compile_target,
        "schedule_artifact": artifact,
    }
    if args.emit == "json":
        text = json.dumps(payload, indent=2, sort_keys=True)
    elif args.emit == "trace":
        text = json.dumps({"traceEvents": sess.timeline_events(), "summary": payload["summary"]}, indent=2, sort_keys=True)
    else:
        text = report
    if args.output:
        Path(args.output).write_text(text + "\n")
    else:
        sys.stdout.write(text + "\n")
    if args.trace:
        sys.stdout.write(f"trace: {args.trace}\n")
    if args.artifact:
        sys.stdout.write(f"artifact: {args.artifact}\n")
    return 0


def _count_symbols(text: str) -> int:
    module = ast.parse(text)
    return sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in module.body)


def _parse_metrics(value: str | None) -> set[str]:
    if not value:
        return set(DEFAULT_METRICS)
    metrics = {part.strip().lower() for part in value.split(",") if part.strip()}
    unknown = metrics - {"latency", "flops", "bandwidth", "occupancy", "memory", "efficiency"}
    if unknown:
        raise SystemExit(f"unknown profiler metric(s): {', '.join(sorted(unknown))}")
    return metrics


def _counters(metrics: set[str], symbol_count: int) -> dict[str, float]:
    counters: dict[str, float] = {"symbols": float(symbol_count)}
    if "occupancy" in metrics:
        counters["occupancy_pct"] = 75.0
    if "memory" in metrics:
        counters["memory_bytes"] = 512.0e6
    return counters


def _parse_shape(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.replace("x", ",").split(",") if part.strip()]
    if len(parts) != 3:
        raise SystemExit("--shapes expects M,N,K")
    try:
        M, N, K = (int(part) for part in parts)
    except ValueError as exc:
        raise SystemExit("--shapes expects integer M,N,K") from exc
    return M, N, K


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tessera-prof",
        description="Capture a lightweight Tessera profiling report and optional Chrome trace.",
    )
    parser.add_argument("model", help="Python model source to profile or inspect")
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metrics: latency,flops,bandwidth,occupancy,memory,efficiency",
    )
    parser.add_argument("--trace", help="Write a Chrome Trace Event JSON file")
    parser.add_argument(
        "--emit",
        choices=("report", "json", "trace"),
        default="report",
        help="Output format for stdout or --output",
    )
    parser.add_argument("--output", help="Write emitted report/json/trace text to a file")
    parser.add_argument("--compile-target", default="generic", help="Target architecture/profile for profiling metadata")
    parser.add_argument("--autotune", action="store_true", help="Run foundation GEMM autotuning after source inspection")
    parser.add_argument("--autotune-method", choices=("roofline", "grid", "bayesian", "on_device"), default="roofline")
    parser.add_argument("--shapes", default="128,128,128", help="GEMM shape for --autotune as M,N,K")
    parser.add_argument("--max-trials", type=int, default=8, help="Maximum autotune trials")
    parser.add_argument("--cache", help="SQLite tuning cache path")
    parser.add_argument("--artifact", help="Write selected schedule artifact JSON")
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
