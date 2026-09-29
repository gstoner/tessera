#!/usr/bin/env python3
"""Matched, device-clock-witnessed timing of the folded gfx1201 load schedule.

Owner ROCM-MXFP4-W4A8-1; sync GFX1201-LANES-2026-09-27. For each shape the
recorder builds, on the same lossless-fold logical inputs as every earlier
folded packet (``benchmark_gfx1201_mxfp4_production._logical_inputs``):

* ``tessera_exact_k32`` -- the exact per-K32 route, correctness oracle only;
* ``tessera_folded_v1`` -- the original folded schedule, packaged directly;
* ``tessera_folded_selected`` -- the compiled route: an authored Graph op
  lowered Graph -> Schedule -> Tile -> Target, whose Target IR carries the
  physical load schedule the materializer consumes;
* ``radiance`` -- pinned Radiance with fragment-order weights (WPERM=1);
* optionally (``--decomposition``) single-key and leave-one-out schedules.

Before any timing, the exact route must match an independent sampled FP32
dequantized reference and every engine must produce bitwise-identical BF16
output. Timing then alternates engine order across trials. Each window is
bracketed by the compiler-built device-clock marker
(``--tessera-device-clock-span``, ``llvm.readsteadycounter``) and by HIP
events on the same stream; a window is at least ``MIN_WINDOW_MS`` long and
the device clock must agree with its HIP-event witness within 5%. A plain
(unbracketed) event window per engine and trial bounds the markers' own cost.
Rows report the device clock as the primary latency (``timing_source``).

``--processes`` repeats the whole measurement in independent processes with
alternating engine order; the summary reports every process's medians, not
only a pooled value. No profiler counter is captured (Tajasarus has no
``/dev/kfd``); nothing here is a DRAM or phase measurement.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

import numpy as np  # noqa: E402

from tessera import runtime as rt  # noqa: E402
from tessera.compiler.native_device_clock import build_device_clock_marker  # noqa: E402
from tessera.compiler.profiler_timing import wall_clock_ticks_to_ns  # noqa: E402
from tessera.compiler.rocm_mxfp4_folded import (  # noqa: E402
    FOLDED_PREFILL_SCHEDULE_V1, FoldedPrefillSchedule,
)
from benchmarks.rocm import ablate_gfx1201_folded_load_schedule as ls  # noqa: E402
from benchmarks.rocm import benchmark_gfx1201_mxfp4_folded as folded_bench  # noqa: E402
from benchmarks.rocm import benchmark_gfx1201_mxfp4_production as base  # noqa: E402
from benchmarks.rocm.inspect_gfx1201_folded_prefill import (  # noqa: E402
    selected_symbol_isa_evidence,
)

SCHEMA = "tessera.rocm.gfx1201_mxfp4_folded_load_schedule.v1"
SYNC_KEY = "GFX1201-LANES-2026-09-27"
AGREEMENT_BAND = 0.05
MIN_WINDOW_MS = 5.0
PRODUCTION = ((256, 5120, 8704), (1024, 17408, 5120))
#: One-row-block prefill (M <= 256): the band still behind Radiance
#: (GFX1201-PERF-2026-09-27).
SMALL = ((128, 5120, 8704), (128, 17408, 5120), (256, 5120, 8704), (256, 17408, 5120))
#: M = 256 (one whole BM256 row block) across N at K = 5120
#: (FOUNDATION-BATCH-2-2026-09-27): the expanded E4M3 weight is N*K bytes, so
#: its per-launch DRAM stream crosses the RX 9070 XT's 64 MiB last-level cache
#: at N ~ 13k, where the packed E2M1 weight (half the bytes) is still at 32 MiB.
#: Every engine rotates three input copies, so each launch reads its weight from
#: memory, not from the previous launch's cache footprint.
NSCAN = tuple((256, n, 5120) for n in (4096, 8192, 12288, 16384, 17408, 20480, 24576))
NSCAN_DECOMPOSITION = tuple((256, n, 5120) for n in (8192, 12288, 17408))
#: Three N at M = 256, K = 5120 for a per-column slope, the per-column cost's
#: attribution probes (FOUNDATION-BATCH-3-2026-09-28).
SLOPE = tuple((256, n, 5120) for n in (8192, 12288, 16384))
SWEEP = (
    (128, 5120, 8704), (128, 17408, 5120), (256, 17408, 5120),
    (512, 5120, 8704), (512, 17408, 5120), (1024, 5120, 8704),
    (2048, 5120, 8704), (2048, 17408, 5120),
)
SOURCES = (
    "python/tessera/compiler/rocm_mxfp4_folded.py",
    "python/tessera/compiler/rocm_mxfp4_folded_carrier.py",
    "python/tessera/compiler/rocm_mxfp4_folded_frontend.py",
    "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/TileToROCM.cpp",
    "benchmarks/rocm/ablate_gfx1201_folded_load_schedule.py",
    "benchmarks/rocm/benchmark_gfx1201_mxfp4_folded.py",
    "benchmarks/rocm/benchmark_gfx1201_mxfp4_production.py",
    "benchmarks/rocm/record_gfx1201_mxfp4_folded_load_schedule.py",
)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _source_state() -> dict[str, object]:
    return {
        "source_commit": _git("rev-parse", "HEAD"),
        "worktree_dirty": bool(_git("status", "--porcelain")),
        "source_sha256": {path: base._sha256(ROOT / path) for path in SOURCES},
        "execution_environment": (
            "wsl2" if "microsoft" in platform.release().lower() else "bare_metal"
        ),
        "kernel_release": platform.release(),
    }


class DeviceClock:
    """Marker-bracketed device-clock windows with a HIP-event witness."""

    def __init__(self, hip: ctypes.CDLL, compiler: Path, llvm_bin: Path) -> None:
        from benchmarks.record_ssd_gpu import _hip_enum

        self.hip = hip
        rate = ctypes.c_int()
        if hip.hipDeviceGetAttribute(
            ctypes.byref(rate), _hip_enum("hipDeviceAttributeWallClockRate"), 0,
        ) != 0 or rate.value <= 0:
            raise RuntimeError("hipDeviceAttributeWallClockRate is unavailable")
        self.rate_khz = rate.value
        self.marker = build_device_clock_marker(
            compiler=compiler, llvm_bin=llvm_bin, backend="rocm", chip="gfx1201",
        )
        self.module = ctypes.c_void_p()
        self.function = ctypes.c_void_p()
        self.span = ctypes.c_void_p()
        self.events = [ctypes.c_void_p(), ctypes.c_void_p()]
        self.host_span = (ctypes.c_uint64 * 2)()
        self._blob = ctypes.create_string_buffer(self.marker.image, len(self.marker.image))
        try:
            self._check(hip.hipModuleLoadData(ctypes.byref(self.module), self._blob),
                        "marker module load")
            self._check(hip.hipModuleGetFunction(
                ctypes.byref(self.function), self.module, self.marker.entry.encode(),
            ), "marker entry")
            self._check(hip.hipMalloc(ctypes.byref(self.span), 16), "span allocation")
            for event in self.events:
                self._check(hip.hipEventCreate(ctypes.byref(event)), "event creation")
        except Exception:
            self.close()
            raise
        self._argv = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.byref(self.span), ctypes.c_void_p))

    @staticmethod
    def _check(status: int, what: str) -> None:
        if status != 0:
            raise RuntimeError(f"device-clock {what} failed rc={status}")

    def _marker(self) -> None:
        self._check(self.hip.hipModuleLaunchKernel(
            self.function, 1, 1, 1, 1, 1, 1, 0, None, self._argv, None,
        ), "marker launch")

    def window(self, engine: base._Engine, launches: int, *, bracketed: bool) -> dict[str, Any]:
        hip = self.hip
        self.host_span[0], self.host_span[1] = (1 << 64) - 1, 0
        host = ctypes.c_void_p(ctypes.addressof(self.host_span))
        self._check(hip.hipMemcpy(self.span, host, 16, 1), "span reset")
        self._check(hip.hipDeviceSynchronize(), "pre-window synchronize")
        start = time.perf_counter_ns()
        self._check(hip.hipEventRecord(self.events[0], None), "start event")
        if bracketed:
            self._marker()
        for _ in range(launches):
            engine.launch()
        if bracketed:
            self._marker()
        self._check(hip.hipEventRecord(self.events[1], None), "stop event")
        self._check(hip.hipEventSynchronize(self.events[1]), "stop synchronize")
        host_ms = (time.perf_counter_ns() - start) / 1e6
        elapsed = ctypes.c_float()
        self._check(hip.hipEventElapsedTime(ctypes.byref(elapsed), self.events[0],
                                            self.events[1]), "event elapsed")
        sample: dict[str, Any] = {
            "launches": launches, "bracketed": bracketed,
            "event_window_ms": float(elapsed.value), "host_window_ms": host_ms,
        }
        if bracketed:
            self._check(hip.hipMemcpy(host, self.span, 16, 2), "span read")
            begin, end = int(self.host_span[0]), int(self.host_span[1])
            if begin == (1 << 64) - 1 or end <= begin:
                raise RuntimeError(f"device-clock marker span was not written: {[begin, end]}")
            device_ms = wall_clock_ticks_to_ns(end - begin, self.rate_khz) / 1e6
            sample["device_window_ms"] = device_ms
            sample["device_event_disagreement"] = abs(device_ms - elapsed.value) / elapsed.value
        return sample

    def close(self) -> None:
        for event in self.events:
            if event.value:
                self.hip.hipEventDestroy(event)
        if self.span.value:
            self.hip.hipFree(self.span)
        if self.module.value:
            self.hip.hipModuleUnload(self.module)


def _schedule_from(receipt: dict[str, Any]) -> FoldedPrefillSchedule:
    selected = receipt["selected_schedule"]
    return FoldedPrefillSchedule(
        raster_group_m=int(selected["raster_group_m"]),
        workgroup_mode=str(selected["workgroup_mode"]),
        staging_prefetch=str(selected["staging_prefetch"]),
        epilogue=str(selected["epilogue_schedule"]),
        row_guard=str(selected["row_guard"]),
    )


def _decomposition(selected: FoldedPrefillSchedule) -> list[tuple[str, FoldedPrefillSchedule]]:
    """Single-key additions to V1 and leave-one-out removals from ``selected``."""
    defaults = FOLDED_PREFILL_SCHEDULE_V1
    keys = ("raster_group_m", "workgroup_mode", "staging_prefetch", "epilogue", "row_guard")
    changed = [key for key in keys if getattr(selected, key) != getattr(defaults, key)]
    out: list[tuple[str, FoldedPrefillSchedule]] = []
    for key in changed:
        out.append((f"v1_plus_{key}", FoldedPrefillSchedule(**{key: getattr(selected, key)})))
    if len(changed) > 1:
        for key in changed:
            fields = {name: getattr(selected, name) for name in keys}
            fields[key] = getattr(defaults, key)
            out.append((f"selected_minus_{key}", FoldedPrefillSchedule(**fields)))
    return out


def _engine_evidence(engine: base._Engine) -> dict[str, Any]:
    """Timed-image identity, selected-symbol ISA and resources of one engine."""
    return dict(engine.metadata)


def run_case(
    hip: ctypes.CDLL, clock: DeviceClock, case: base.Case, *, tessera_opt: Path,
    radiance: Any, decomposition: bool, trials: int, order_seed: int,
    diagnostics: tuple[tuple[str, tuple[str, ...]], ...] = (),
    packed: bool = False, copies: int = 3,
) -> dict[str, Any]:
    inputs = base._logical_inputs(case)
    exact = base._tessera_engine(hip, case, inputs, 3, None, None, name="tessera_exact_k32")
    engines: list[base._Engine] = []
    try:
        selected, folded = folded_bench.folded_engine(
            hip, case, inputs, copies, tessera_opt=tessera_opt,
        )
        selected.name = "tessera_folded_selected"
        engines.append(selected)
        if not folded.lossless:
            raise RuntimeError("matched timing requires lossless-fold inputs")
        receipt = selected.metadata["frontend_receipt"]
        schedule = _schedule_from(receipt)
        engines.append(ls.schedule_engine(
            hip, case, inputs, folded, FOLDED_PREFILL_SCHEDULE_V1, name="tessera_folded_v1",
            copies=copies,
        ))
        if decomposition:
            for label, variant in _decomposition(schedule):
                engines.append(ls.schedule_engine(
                    hip, case, inputs, folded, variant, name=f"tessera_{label}",
                    copies=copies,
                ))
        # Diagnostic source edits on top of the selected schedule; each must
        # still produce the exact route's BF16 bits (checked below).
        for label, edits in diagnostics:
            engines.append(ls.schedule_engine(
                hip, case, inputs, folded, schedule, edits, name=f"tessera_selected+{label}",
                copies=copies,
            ))
        if packed:
            # The opt-in packed-E2M1 candidates (manual, never selected): the
            # weight read at half the expanded bytes, decoded in the kernel.
            # Diagnostic arms for the packed-bytes hypothesis at one row block.
            from benchmarks.rocm.benchmark_gfx1201_mxfp4_packed_folded import (
                packed_folded_engine,
            )
            for flags in ({"batched_loads": True, "permute_decode": True},
                          {"batched_loads": True, "permute_decode": True,
                           "a_offset32": True}):
                engine, _ = packed_folded_engine(
                    hip, case, inputs, copies, integer_decode=False, **flags)
                engines.append(engine)
        engines.append(base._radiance_engine(hip, radiance, case, inputs, copies))
        outputs = {engine.name: engine.output() for engine in [exact, *engines]}
        rows, cols, reference = base._sampled_exact_reference(case, inputs)
        np.testing.assert_array_equal(
            outputs["tessera_exact_k32"][np.ix_(rows, cols)], reference,
            err_msg=f"{case.label}: exact K32 fails the independent oracle",
        )
        engine_by_name = {engine.name: engine for engine in engines}
        for name, output in outputs.items():
            if name != exact.name and engine_by_name[name].metadata.get("diagnostic_bound", False):
                # Output-changing attribution probes are identified by the
                # source edit, independent of the caller's display label.
                continue
            np.testing.assert_array_equal(
                output.view(np.uint16), outputs["tessera_exact_k32"].view(np.uint16),
                err_msg=f"{case.label}: {name} changed BF16 output",
            )
    finally:
        exact.close()
    try:
        for engine in engines:
            base._warmup(hip, engine, 6)
        estimates = [
            clock.window(engine, 3, bracketed=False)["event_window_ms"] / 3
            for engine in engines
        ]
        launches = max(12, math.ceil(MIN_WINDOW_MS * 1.2 / min(estimates)))
        samples: dict[str, list[dict[str, Any]]] = {engine.name: [] for engine in engines}
        for trial in range(trials):
            order = engines if (trial + order_seed) % 2 == 0 else list(reversed(engines))
            for engine in order:
                pair = (True, False) if trial % 2 == 0 else (False, True)
                for bracketed in pair:
                    samples[engine.name].append(
                        clock.window(engine, launches, bracketed=bracketed)
                    )
        result_rows = []
        for engine in engines:
            bracketed = [s for s in samples[engine.name] if s["bracketed"]]
            plain = [s for s in samples[engine.name] if not s["bracketed"]]
            device = [s["device_window_ms"] / s["launches"] for s in bracketed]
            event = [s["event_window_ms"] / s["launches"] for s in bracketed]
            plain_event = [s["event_window_ms"] / s["launches"] for s in plain]
            disagreement = [s["device_event_disagreement"] for s in bracketed]
            evidence = _engine_evidence(engine)
            result_rows.append({
                "case": case.label, "engine": engine.name,
                "timing_source": "device_wall_clock_marker",
                "witness": "hip_event",
                "median_device_ms": statistics.median(device),
                "median_event_ms": statistics.median(event),
                "median_plain_event_ms": statistics.median(plain_event),
                "marker_bracket_ratio": statistics.median(event) / statistics.median(plain_event),
                "max_device_event_disagreement": max(disagreement),
                "witness_agrees": max(disagreement) <= AGREEMENT_BAND,
                "window_ms_min": min(s["device_window_ms"] for s in bracketed),
                "device_ms_samples": device, "event_ms_samples": event,
                "plain_event_ms_samples": plain_event,
                "output_sha256": hashlib.sha256(outputs[engine.name].view(np.uint8)).hexdigest(),
                "metadata": evidence,
            })
        return {
            "case": case.label, "m": case.m, "n": case.n, "k": case.k,
            "launches_per_window": launches, "trials": trials,
            "selected_schedule": receipt["selected_schedule"],
            "frontend_receipt": receipt,
            "rows": result_rows,
        }
    finally:
        for engine in reversed(engines):
            engine.close()


def _child(args: argparse.Namespace) -> None:
    if rt._rocm_live_arch() != "gfx1201":
        raise SystemExit("folded load-schedule timing requires the selected gfx1201 device")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        raise SystemExit("folded load-schedule timing requires HIP")
    from tessera.compiler.llvm_tools import llvm_bin_dir

    llvm_bin = llvm_bin_dir()
    if llvm_bin is None:
        raise SystemExit("matched LLVM 23 tools not found (set TESSERA_LLVM_BIN)")
    radiance = base._load_radiance(args.radiance_module)
    clock = DeviceClock(hip, args.tessera_opt, llvm_bin)
    try:
        cases = []
        for m, n, k in args.shapes:
            cases.append(run_case(
                hip, clock, base.Case("prefill", m, n, k), tessera_opt=args.tessera_opt,
                radiance=radiance, decomposition=args.decomposition,
                trials=args.trials, order_seed=args.order_seed,
                diagnostics=_diagnostic_specs(args.diagnostics),
                packed=args.packed, copies=args.copies,
            ))
            print(f"completed {m}x{n}x{k}", flush=True)
        packet = {
            "process_id": os.getpid(), "order_seed": args.order_seed,
            "device": base._selected_device_name(hip),
            "architecture": rt._rocm_live_arch(),
            "wall_clock_rate_khz": clock.rate_khz,
            "marker_image_sha256": clock.marker.image_sha256,
            "cases": cases,
        }
    finally:
        clock.close()
    args.child_output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")


def _diagnostic_specs(items: list[str]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    specs = []
    for item in items:
        label, _, edits = item.partition("=")
        if not label or not edits:
            raise SystemExit(f"--diagnostics wants LABEL=edit[+edit...], got {item!r}")
        specs.append((label, tuple(edits.split("+"))))
    return tuple(specs)


def _summarize(processes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, list[float]]] = {}
    for process in processes:
        for case in process["cases"]:
            for row in case["rows"]:
                by_case.setdefault(case["case"], {}).setdefault(row["engine"], []).append(
                    row["median_device_ms"]
                )
    summary = []
    for label, engines in by_case.items():
        selected = engines["tessera_folded_selected"]
        v1 = engines["tessera_folded_v1"]
        radiance = engines["radiance"]
        summary.append({
            "case": label,
            "per_process_median_device_ms": engines,
            "selected_over_v1": [s / b for s, b in zip(selected, v1)],
            "selected_over_radiance": [s / r for s, r in zip(selected, radiance)],
            "v1_over_radiance": [b / r for b, r in zip(v1, radiance)],
            # Diagnostic engines (selected schedule + source edits), if any.
            "over_selected": {
                name: [t / s for t, s in zip(times, selected)]
                for name, times in engines.items() if name.startswith("tessera_selected+")
            },
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radiance-module", type=Path, required=True)
    parser.add_argument("--radiance-revision", required=True)
    parser.add_argument("--tessera-opt", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shapes", choices=("production", "sweep", "all", "small", "nscan", "nscan_decomposition", "slope"),
                        default="production")
    parser.add_argument("--decomposition", action="store_true")
    parser.add_argument("--packed", action="store_true",
                        help="also time the opt-in packed-E2M1 folded candidates")
    parser.add_argument("--copies", type=int, default=3,
                        help="rotating device copies of every input per engine (3: each "
                             "launch reads operands the previous launch did not; 1: "
                             "operands may stay resident in the last-level cache)")
    parser.add_argument("--diagnostics", action="append", default=[],
                        help="LABEL=edit[+edit...]: an extra engine of the selected schedule "
                             "with those diagnostic source edits (see ablate module)")
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--sync-key", default=SYNC_KEY)
    parser.add_argument("--trials", type=int, default=11)
    parser.add_argument("--diagnostic", action="store_true",
                        help="allow a dirty tree; the packet records it and is not evidence")
    parser.add_argument("--child-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--order-seed", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--child-shapes", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.copies < 1:
        raise SystemExit("--copies must be at least 1")
    if os.environ.get("RADIANCE_MXFP4_WPERM") != "1":
        raise SystemExit("matched fragment-order Radiance requires RADIANCE_MXFP4_WPERM=1")
    args.tessera_opt = args.tessera_opt.resolve()
    if not args.tessera_opt.is_file():
        raise SystemExit(f"tessera-opt not found: {args.tessera_opt}")
    if args.child_output is not None:
        args.shapes = [tuple(int(v) for v in item.split("x"))
                       for item in (args.child_shapes or "").split(",") if item]
        _child(args)
        return
    if args.output is None:
        raise SystemExit("--output is required")
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        raise SystemExit("write measurement evidence outside the source checkout")
    source = _source_state()
    if source["worktree_dirty"] and not args.diagnostic:
        raise SystemExit("source tree has uncommitted changes; commit first or pass --diagnostic")
    shapes = {"production": PRODUCTION, "sweep": SWEEP, "all": PRODUCTION + SWEEP,
              "small": SMALL, "nscan": NSCAN,
              "nscan_decomposition": NSCAN_DECOMPOSITION, "slope": SLOPE}[args.shapes]
    output.parent.mkdir(parents=True, exist_ok=True)
    processes = []
    for index in range(args.processes):
        child = output.with_name(f"{output.stem}.process{index}.json")
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--radiance-module", str(args.radiance_module),
            "--radiance-revision", args.radiance_revision,
            "--tessera-opt", str(args.tessera_opt),
            "--trials", str(args.trials), "--order-seed", str(index),
            "--child-output", str(child),
            "--child-shapes", ",".join(f"{m}x{n}x{k}" for m, n, k in shapes),
        ] + (["--decomposition"] if args.decomposition else []) + (
            ["--packed"] if args.packed else []) + ["--copies", str(args.copies)] + [
            f"--diagnostics={item}" for item in args.diagnostics]
        subprocess.run(command, cwd=ROOT, check=True, timeout=7200)
        processes.append(json.loads(child.read_text()))
        print(f"completed process {index + 1}/{args.processes}", flush=True)
    if _source_state()["source_sha256"] != source["source_sha256"]:
        raise SystemExit("source changed while collecting evidence")
    packet = {
        "schema": SCHEMA, "sync_key": args.sync_key, "work_item": "ROCM-MXFP4-W4A8-1",
        "diagnostics": args.diagnostics, "packed": args.packed, "shapes": args.shapes,
        "copies": args.copies,
        "source": source,
        "tessera_opt_sha256": base._sha256(args.tessera_opt),
        "radiance": {
            "revision": args.radiance_revision,
            "binary_sha256": base._sha256(args.radiance_module),
            "weight_layout": "fragment_order", "wperm": 1,
        },
        "timing": {
            "primary": "device_wall_clock_marker (llvm.readsteadycounter via "
                       "--tessera-device-clock-span marker kernels)",
            "witness": "hip_event on the same stream and window",
            "agreement_band": AGREEMENT_BAND, "min_window_ms": MIN_WINDOW_MS,
            "order": "alternating per trial; process order seed alternates",
            "profiler_counters": "not captured (WSL2, no /dev/kfd)",
        },
        "processes": processes,
        "summary": _summarize(processes),
    }
    output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps(packet["summary"], indent=2))


if __name__ == "__main__":
    main()
