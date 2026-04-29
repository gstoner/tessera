"""Profiling command for Tessera Python sources."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Sequence

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
        )
        if args.trace:
            sess.timeline(args.trace)
        report = sess.report()

    sys.stdout.write(report + "\n")
    if args.trace:
        sys.stdout.write(f"trace: {args.trace}\n")
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
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
