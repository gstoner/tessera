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
from tessera.compiler.profiling_plan import (
    ModelAnalyzerSweep,
    model_analyzer_manifest,
    plan_profile,
)
from tessera.compiler.model_analyzer import run_model_analyzer_manifest, write_model_analyzer_result


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
            artifact = autotune.schedule_artifact(  # type: ignore[attr-defined]
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

    analyzer_sweep = None
    advanced_plan = None
    analyzer_result = None
    if args.advanced_plan or args.model_analyzer_manifest or args.model_analyzer_result:
        analyzer_sweep = ModelAnalyzerSweep(
            mode=args.analyzer_mode,
            batch_sizes=_parse_int_list(args.batch_sizes),
            instance_counts=_parse_int_list(args.instance_counts),
            dynamic_batching=_parse_bool_list(args.dynamic_batching),
            latency_budget_ms=args.latency_budget_ms,
            memory_budget_bytes=args.memory_budget_bytes,
        )
        plan = plan_profile(
            args.compile_target,
            features=_parse_trace_features(args.trace_features),
            model_name=source.stem,
            kernels=tuple(args.kernels),
            analyzer_sweep=analyzer_sweep,
        )
        if args.advanced_plan:
            advanced_plan = plan.to_dict()
        if args.model_analyzer_manifest or args.model_analyzer_result:
            manifest = model_analyzer_manifest(plan)
            manifest_payload = manifest.to_dict()
        if args.model_analyzer_manifest:
            Path(args.model_analyzer_manifest).write_text(manifest.to_json() + "\n")
        if args.model_analyzer_result:
            analyzer_result = run_model_analyzer_manifest(manifest_payload)
            write_model_analyzer_result(analyzer_result, args.model_analyzer_result)

    payload = {
        **sess.to_dict(),
        "mode": "source_inspection",
        "source": str(source),
        "compile_target": args.compile_target,
        "schedule_artifact": artifact,
        "advanced_profiler_plan": advanced_plan,
        "model_analyzer_result": analyzer_result,
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
    if args.model_analyzer_manifest:
        sys.stdout.write(f"model_analyzer_manifest: {args.model_analyzer_manifest}\n")
    if args.model_analyzer_result:
        sys.stdout.write(f"model_analyzer_result: {args.model_analyzer_result}\n")
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


def _parse_trace_features(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_int_list(value: str) -> tuple[int, ...]:
    try:
        out = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise SystemExit("expected a comma-separated integer list") from exc
    if not out:
        raise SystemExit("expected at least one integer")
    return out


def _parse_bool_list(value: str) -> tuple[bool, ...]:
    mapping = {"1": True, "true": True, "yes": True, "0": False, "false": False, "no": False}
    out: list[bool] = []
    for part in value.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise SystemExit("expected booleans as true,false,1,0,yes,no")
        out.append(mapping[key])
    if not out:
        raise SystemExit("expected at least one boolean")
    return tuple(out)


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
    parser.add_argument(
        "--advanced-plan",
        action="store_true",
        help="Attach compiler profiler/provider plan JSON for runtime/device tracing.",
    )
    parser.add_argument(
        "--trace-features",
        default="host_context,runtime_api,device_activity,counters,roofline,model_analyzer",
        help=(
            "Comma-separated advanced profiling features: host_context,runtime_api,"
            "device_activity,counters,intra_kernel,model_analyzer,roofline"
        ),
    )
    parser.add_argument(
        "--kernels",
        nargs="*",
        default=(),
        help="Kernel names to correlate in the advanced profiler plan.",
    )
    parser.add_argument(
        "--analyzer-mode",
        choices=("quick", "brute", "manual", "optuna"),
        default="quick",
        help="Model Analyzer style search mode for --advanced-plan.",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8",
        help="Comma-separated batch sizes for the model analyzer sweep.",
    )
    parser.add_argument(
        "--instance-counts",
        default="1",
        help="Comma-separated instance counts for the model analyzer sweep.",
    )
    parser.add_argument(
        "--dynamic-batching",
        default="false,true",
        help="Comma-separated booleans for dynamic batching sweep states.",
    )
    parser.add_argument("--latency-budget-ms", type=float, help="Optional analyzer latency budget")
    parser.add_argument("--memory-budget-bytes", type=int, help="Optional analyzer memory budget")
    parser.add_argument(
        "--model-analyzer-manifest",
        help="Write runner-facing Tessera Model Analyzer manifest JSON.",
    )
    parser.add_argument(
        "--model-analyzer-result",
        help=(
            "Write a local Tessera Model Analyzer result JSON by running the "
            "manifest search contract with estimated measurements."
        ),
    )
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
