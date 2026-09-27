#!/usr/bin/env python3
"""ROCM-SPLIT-K-1 follow-on: a split-K slice sweep over skinny shapes, timed on
the compiler-built device clock. A measurement, never an admission of a route.

For every ``--shapes`` entry (``MxNxK``) and storage the production route is
lowered once (Graph -> Schedule -> Tile through ``lower_scheduled_matmul``).
From its Tile IR the recorder builds one image per slice count in
``--slices`` (1 = the unsplit kernel):

* the slice count the production Schedule selected uses the production
  package's image byte-for-byte (row ``production: true``);
* every other count is the same Tile IR with the ``tessera.split_k`` pair
  inserted, removed or rewritten, compiled by the same
  ``rocm_native._compile_native_tile_ir`` call the packager uses, with the
  packager's K-unroll fit rule (``resolve_scheduled_matmul_k_unroll``: the
  derived unroll, or 1 when it does not divide the slice). Counts whose slices
  are not whole macro K blocks are recorded as refused, not timed. Counts below
  ``rocm_tiling.SPLIT_K_MIN_SLICE_K`` per slice ARE built: testing that
  unmeasured guard is one of the sweep's purposes.

Images are compiled once in the parent and the same bytes are timed by every
fresh worker process (``--runs``), so run-to-run spread is not compile spread.

Timing (sync ``WSL-TIMING-ADMISSION-2026-09-26``): each worker times every
variant in ``--rounds`` interleaved rounds (order reversed on odd rounds). A
round is one *plain* window (HIP events around N launches) and one
*bracketed* window (the same N launches between two launches of the
``--tessera-device-clock-span`` marker, which records the window on the
device's constant-rate counter), their order alternating. N is chosen per
shape so the fastest variant's window is at least ``--window-ms`` (the ROCm
minimum is 5 ms, ``profiler_timing.MINIMUM_DEVICE_CLOCK_WINDOW_NS``). Each
variant's windows become one ``tessera.profiler_rocm_packet.v1`` built by
``build_rocm_profiler_packet`` -- which re-derives the admission route and
the per-window refusals itself -- so a row's device-clock number is quoted
only alongside the packet that did or did not admit it. A split iteration is
BOTH launches (partial + ordered reduce); the fp32 workspace is allocated once
outside the timed loop (``runtime.launch`` allocates per call; that cost is
excluded, as in the 2026-09-26 packet).

Correctness before timing: every variant against an f64 reference, and two
launches of every variant compared bit for bit (the ordered reduction's
determinism claim). A variant that fails is recorded with its failure.

Usage (Tajasarus, toolkit env sourced, TESSERA_ROCM_CHIP=gfx1201; commit
first -- a dirty tree is refused unless ``--diagnostic``)::

    python benchmarks/rocm/record_split_k_sweep.py --output-dir \\
        benchmarks/baselines/rocm_split_k_20260927
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.rocm.record_split_k_router_gate import (  # noqa: E402
    REDUCE_THREADS, _graph_module, _Variant)

SPLIT_PAIR = re.compile(r', tessera\.split_k = (\d+) : i64, tessera\.split_k_reduction = "ordered"')
DEFAULT_SHAPES = ",".join((
    # MoE router gates (N = experts): OLMoE 64x2048, Qwen3-30B-A3B 128x2048,
    # DeepSeek-V3 256x7168, Qwen3-235B 128x4096, Mixtral 8x4096 (ragged N).
    "16x64x2048", "16x128x2048", "16x256x2048", "16x256x7168", "32x256x7168",
    "64x128x4096", "16x8x4096",
    # Occupancy-short with a small K: the 256-per-slice guard's territory.
    "16x256x256", "32x128x512", "64x64x1024", "48x96x1536",
    # Decode / MoE expert GEMMs: M <= 64, output tiles >= 32 WGPs (the rule
    # never splits these; the sweep asks whether it should).
    "16x768x2048", "16x2048x768", "64x512x2048", "32x1536x4096", "16x2048x7168",
))


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def _isa_sha256(image: bytes, llvm_bin: Path) -> str:
    """Digest of the instruction stream; an hsaco container is not
    byte-deterministic across rebuilds of an unchanged kernel."""
    with tempfile.TemporaryDirectory(prefix="splitk-isa-") as tmp:
        obj = Path(tmp) / "image.hsaco"
        obj.write_bytes(image)
        text = subprocess.run([str(llvm_bin / "llvm-objdump"), "-d", str(obj)], check=True,
                              capture_output=True, text=True, timeout=120).stdout
    body = "\n".join(line for line in text.splitlines() if str(obj) not in line)
    return hashlib.sha256(body.encode()).hexdigest()


def _with_split(tile_ir: str, slices: int) -> str:
    """The Tile IR with its split pair set to ``slices`` (1 removes it)."""
    stripped = SPLIT_PAIR.sub("", tile_ir)
    if "split_k" in stripped:
        raise RuntimeError("the Tile IR carries a split attribute this recorder cannot parse")
    if slices == 1:
        return stripped
    anchor = ", tessera.tile_k = "
    if stripped.count(anchor) != 1:
        raise RuntimeError("expected exactly one tile.matmul_kernel with tessera.tile_k")
    return stripped.replace(
        anchor, f', tessera.split_k = {slices} : i64, tessera.split_k_reduction = "ordered"'
        + anchor)


def build_variants(chip, shape, dtype, slices_axis, llvm_bin, directory):
    """Compile every slice count once; returns the manifest group."""
    from tessera.compiler import rocm_native, scheduled_matmul

    m, n, k = shape
    artifact = scheduled_matmul.lower_scheduled_matmul(
        _graph_module(m, n, k, dtype), target=f"rocm_{chip}")
    package = rocm_native.package_scheduled_matmul(artifact, pipeline_name="tessera-lower-to-rocm")
    provenance = package.descriptor.provenance
    selected = int(artifact.split_k)
    macro = tuple(provenance["macro_tile"])
    block_k = max(scheduled_matmul.rocm_gfx1201_block_k(k, dynamic_k=False), 16)
    derived_unroll = scheduled_matmul.rocm_k_unroll(
        m, n, k, arch=chip, dynamic=False, storage=artifact.storage)
    group = dict(shape=[m, n, k], dtype=dtype, selected_split_k=selected, macro_tile=list(macro),
                 block_k=block_k, derived_k_unroll=derived_unroll,
                 production_route=provenance["route"],
                 production_physical_route=provenance["physical_route"],
                 schedule_digest=artifact.schedule_digest, function=artifact.function_name,
                 variants=[])
    for slices in sorted(set(slices_axis) | {1, selected}):
        variant = dict(split_k=slices, production=slices == selected)
        group["variants"].append(variant)
        if k % (slices * block_k) != 0:
            variant["refused"] = (f"K={k} does not split into {slices} slices of whole macro "
                                  f"K blocks ({block_k})")
            continue
        tile_ir = artifact.tile_ir if slices == selected else _with_split(artifact.tile_ir, slices)
        if slices == selected:
            payload, unroll = package.image.payload, int(provenance["k_unroll"])
        else:
            unroll = derived_unroll if k % (slices * block_k * derived_unroll) == 0 else 1
            try:
                _t, _backend, payload, *_ = rocm_native._compile_native_tile_ir(
                    tile_ir, directive="tessera_rocm.wmma", family="matmul", architecture=chip,
                    staging="register", lds_waves=(2, 2), k_unroll=unroll)
            except Exception as exc:  # the refusal is the result
                variant["refused"] = str(exc)[:300]
                continue
        path = Path(directory) / f"{dtype}_{m}x{n}x{k}_s{slices}.hsaco"
        path.write_bytes(payload)
        variant.update(image=str(path), k_unroll=unroll, slice_k=k // slices,
                       image_sha256=hashlib.sha256(payload).hexdigest(),
                       isa_sha256=_isa_sha256(payload, llvm_bin),
                       semantic_sha256=hashlib.sha256(tile_ir.encode()).hexdigest(),
                       route=(provenance["route"] if slices == selected
                              else "measurement_only_slice_sweep"))
    return group


class _Clock:
    """The device-clock marker, a span buffer and two HIP events."""

    def __init__(self, hip, marker_image: bytes, entry: str):
        self.hip = hip
        self.mod, self.fn, self.span = ct.c_void_p(), ct.c_void_p(), ct.c_void_p()
        self.events = [ct.c_void_p(), ct.c_void_p()]
        self.blob = ct.create_string_buffer(marker_image)
        self.host_span = (ct.c_uint64 * 2)()
        if hip.hipModuleLoadData(ct.byref(self.mod), self.blob) != 0:
            raise RuntimeError("marker image refused by hipModuleLoadData")
        if hip.hipModuleGetFunction(ct.byref(self.fn), self.mod, entry.encode()) != 0:
            raise RuntimeError("marker entry not found")
        if hip.hipMalloc(ct.byref(self.span), ct.sizeof(self.host_span)) != 0:
            raise RuntimeError("hipMalloc(span) failed")
        self.argv_keep = [self.span]
        self.argv = (ct.c_void_p * 1)(ct.cast(ct.byref(self.span), ct.c_void_p))
        for event in self.events:
            if hip.hipEventCreate(ct.byref(event)) != 0:
                raise RuntimeError("hipEventCreate failed")

    def _marker(self):
        rc = self.hip.hipModuleLaunchKernel(self.fn, 1, 1, 1, 1, 1, 1, 0, None, self.argv, None)
        if rc != 0:
            raise RuntimeError(f"marker launch failed rc={rc}")

    def window(self, variant, launches, bracketed, rate_khz):
        """One timed window; per-launch (event_ns, device_ns | None, host_ns)."""
        from tessera.compiler.profiler_timing import wall_clock_ticks_to_ns
        hip = self.hip
        self.host_span[0], self.host_span[1] = (1 << 64) - 1, 0
        if hip.hipMemcpy(self.span, ct.addressof(self.host_span), ct.sizeof(self.host_span), 1):
            raise RuntimeError("span reset failed")
        if hip.hipDeviceSynchronize() != 0:
            raise RuntimeError("hipDeviceSynchronize failed")
        start = time.perf_counter_ns()
        if hip.hipEventRecord(self.events[0], None) != 0:
            raise RuntimeError("hipEventRecord failed")
        if bracketed:
            self._marker()
        for _ in range(launches):
            if variant.launch() != 0:
                raise RuntimeError("timed launch failed")
        if bracketed:
            self._marker()
        if hip.hipEventRecord(self.events[1], None) != 0 or hip.hipEventSynchronize(self.events[1]) != 0:
            raise RuntimeError("hipEventRecord/Synchronize failed")
        host = (time.perf_counter_ns() - start) / launches
        ms = ct.c_float()
        if hip.hipEventElapsedTime(ct.byref(ms), self.events[0], self.events[1]) != 0:
            raise RuntimeError("hipEventElapsedTime failed")
        device = None
        if bracketed:
            if hip.hipMemcpy(ct.addressof(self.host_span), self.span, ct.sizeof(self.host_span), 2):
                raise RuntimeError("span readback failed")
            lo, hi = self.host_span[0], self.host_span[1]
            if lo == (1 << 64) - 1 or hi <= lo:
                raise RuntimeError(f"device-clock marker span was not written: {[lo, hi]}")
            device = wall_clock_ticks_to_ns(hi - lo, rate_khz) / launches
        return ms.value * 1e6 / launches, device, host

    def close(self):
        for event in self.events:
            if event.value:
                self.hip.hipEventDestroy(event)
                event.value = None
        if self.span.value:
            self.hip.hipFree(self.span)
            self.span.value = None
        if self.mod.value:
            self.hip.hipModuleUnload(self.mod)
            self.mod.value = None


def _resources(hip, function):
    getter = hip.hipFuncGetAttribute
    getter.argtypes = [ct.POINTER(ct.c_int), ct.c_int, ct.c_void_p]
    out = {}
    for key, attribute in (("lds_bytes", 1), ("scratch_bytes", 3), ("vgpr", 4)):
        value = ct.c_int()
        if getter(ct.byref(value), attribute, function) != 0:
            raise RuntimeError(f"hipFuncGetAttribute({key}) failed")
        out[key] = value.value
    return out


def _packet(*, chip, identity, variant_meta, group, windows, launches, rate_khz, marker_sha,
            source, run_id, resources):
    from tessera.compiler.profiler_provider_trace import build_provider_trace_artifact
    from tessera.compiler.profiler_rocm_evidence import build_rocm_profiler_packet
    from tessera.compiler.profiler_timing import (
        build_timing_sample, device_clock_window_refusals, measured_clock, unavailable_clock)
    from tessera.compiler.ssd_performance import SSD_CALIBRATION_WINDOW_PROTOCOL

    device_ns = [w["device_ns"] for w in windows]
    event_ns = [w["event_ns"] for w in windows]
    plain_ns = [w["plain_event_ns"] for w in windows]
    host_ns = [w["host_ns"] for w in windows]
    wsl = "microsoft" in platform.release().lower()
    environment = "wsl2" if wsl else "bare_metal"
    m, n, k = group["shape"]
    sample_id = f"splitk-{group['dtype']}-{m}x{n}x{k}-s{variant_meta['split_k']}-{uuid.uuid4().hex[:12]}"
    timing = build_timing_sample(
        sample_id=sample_id, target=f"rocm_{chip}",
        clocks={
            "host_wall_ns": measured_clock("host_wall_ns", source="perf_counter",
                                           value=statistics.median(host_ns)),
            "hip_event_ns": measured_clock("hip_event_ns", source="hip_event",
                                           value=statistics.median(event_ns)),
            "device_wall_clock_ns": measured_clock(
                "device_wall_clock_ns", source="device_wall_clock",
                value=statistics.median(device_ns), instrumented=True,
                calibrated_against=("hip_event_ns",), eligible_for_promotion=True,
                provenance={"method": "compiler_built_marker_bracketing",
                            "windows": len(windows), "launches_per_window": launches,
                            "marker_image_sha256": marker_sha, "per_window_ns": device_ns,
                            "clock": "llvm.readsteadycounter", "wall_clock_rate_khz": rate_khz}),
            "profiler_activity_ns": unavailable_clock(
                "profiler_activity_ns", source="rocprofiler_activity",
                reason="ROCPROFILER_UNAVAILABLE_NO_KFD" if wsl else "NOT_CAPTURED"),
        },
        artifact_digests={"application_image": variant_meta["image_sha256"],
                          "device_clock_marker": marker_sha,
                          "tile_ir": variant_meta["semantic_sha256"]},
        batch_size=launches, warm_state="warm", synchronization="hipEventSynchronize",
        execution_environment=environment, resources={"application": resources},
        environment={"kernel_release": platform.release(), "per_window_event_ns": event_ns,
                     "per_window_plain_event_ns": plain_ns, "per_window_host_ns": host_ns,
                     "window_protocol": SSD_CALIBRATION_WINDOW_PROTOCOL, "run_id": run_id,
                     "process_id": os.getpid(), "device_identity": identity,
                     "launches_per_iteration": 2 if variant_meta["split_k"] > 1 else 1})
    image = dict(architecture=chip, kernel_name=group["function"],
                 semantic_sha256=variant_meta["semantic_sha256"],
                 image_sha256=variant_meta["image_sha256"], isa_sha256=variant_meta["isa_sha256"],
                 clock_source="hip_event", calibration_sample_id=sample_id, resources=resources)
    clean = dict(image, duration_ns=statistics.median(plain_ns), instrumented=False)
    probe = dict(image, duration_ns=statistics.median(event_ns), instrumented=True)
    capture = {
        "schema": "tessera.profiler_rocm_native_capture.v1", "provider": "rocprofiler",
        "status": "blocked", "fresh_process": True, "process": {"clean_exit": True},
        "reason": ("rocprofiler requires /dev/kfd; this WSL2 host exposes /dev/dxg only"
                   if wsl else "rocprofiler capture not requested"),
        "proof": {"dispatch_activity_seen": False, "hip_callback_seen": False,
                  "counter_records_seen": False, "pc_samples_seen": False},
        "requested": {"counters": [], "pc_sampling": False},
        "provider_trace": build_provider_trace_artifact(provider="rocprofiler", records=(),
                                                        source_status="unavailable"),
        "eligible_for_promotion": False,
    }
    packet = build_rocm_profiler_packet(timing=timing, capture=capture, uninstrumented=clean,
                                        instrumented=probe, source=source)
    return packet, device_clock_window_refusals(timing)


def worker(manifest_path: Path, rounds: int, window_ms: float):
    import ml_dtypes
    from benchmarks.calibration.calibrate_gfx1151 import _active_device_identity
    from benchmarks.record_ssd_gpu import _hip_enum
    from tessera import runtime as rt

    manifest = json.loads(manifest_path.read_text())
    chip = manifest["chip"]
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        raise RuntimeError("no usable HIP runtime on this host")
    identity = _active_device_identity(hip)
    if identity["architecture"] != chip:
        raise RuntimeError(f"manifest chip {chip} is not the live device {identity}")
    rate = ct.c_int()
    get_attribute = hip.hipDeviceGetAttribute
    get_attribute.argtypes = [ct.POINTER(ct.c_int), ct.c_int, ct.c_int]
    if get_attribute(ct.byref(rate), _hip_enum("hipDeviceAttributeWallClockRate"), 0) != 0 \
            or rate.value <= 0:
        raise RuntimeError("hipDeviceAttributeWallClockRate is not positive")
    clock = _Clock(hip, Path(manifest["marker_image"]).read_bytes(), manifest["marker_entry"])
    run_id = uuid.uuid4().hex[:12]
    rows = []
    try:
        for group in manifest["groups"]:
            m, n, k = group["shape"]
            storage = np.float16 if group["dtype"] == "fp16" else ml_dtypes.bfloat16
            rng = np.random.default_rng(1201)
            a = (rng.standard_normal((m, k)) * 0.25).astype(storage)
            b = (rng.standard_normal((k, n)) * 0.25).astype(storage)
            ref = a.astype(np.float64) @ b.astype(np.float64)
            scale = float(np.max(np.abs(ref))) + 1e-6
            loaded = []
            for meta in group["variants"]:
                row = dict(dtype=group["dtype"], shape=group["shape"], split_k=meta["split_k"],
                           production=meta["production"])
                rows.append(row)
                if "refused" in meta:
                    row["refused"] = meta["refused"]
                    continue
                variant = None
                try:
                    variant = _Variant(hip, Path(meta["image"]).read_bytes(), group["function"],
                                       a, b, m, n, k, tuple(group["macro_tile"]), meta["split_k"])
                    first = variant.result()
                    # Poison the output so the rerun must recompute it: a
                    # kernel that stopped writing would otherwise compare
                    # equal to itself.
                    if hip.hipMemset(variant.dev[2], 0xFF, 4 * m * n) != 0:
                        raise RuntimeError("hipMemset(output) failed")
                    second = variant.result()
                    row["bit_identical_reruns"] = bool(np.array_equal(
                        first.view(np.uint32), second.view(np.uint32)))
                    rel = float(np.max(np.abs(first.astype(np.float64) - ref))) / scale
                    row["relative_error_vs_f64"] = rel
                    if not math.isfinite(rel) or rel > 2e-3:
                        raise RuntimeError(f"relative error {rel:.3e} exceeds 2e-3")
                    if not row["bit_identical_reruns"]:
                        raise RuntimeError("two launches were not bit-identical")
                    row["resources"] = _resources(hip, variant.fn)
                    loaded.append((row, meta, variant, first))
                except Exception as exc:
                    row["refused"] = str(exc)[:300]
                    if variant is not None:
                        variant.close()
            base = next((x for x in loaded if x[1]["split_k"] == 1), None)
            if base is not None:
                for row, _meta, _v, out in loaded:
                    if row is not base[0]:
                        row["relative_diff_vs_unsplit"] = float(
                            np.max(np.abs(out.astype(np.float64) - base[3]))) / scale
            if not loaded:
                continue
            ramp_until = time.perf_counter() + 0.3
            while time.perf_counter() < ramp_until:
                for _r, _m, variant, _o in loaded:
                    variant.launch()
            hip.hipDeviceSynchronize()
            fastest = min(v.time_batch(50) for _r, _m, v, _o in loaded) * 1e6  # ns / iter
            launches = int(min(20000, max(100, math.ceil(window_ms * 1e6 / fastest))))
            windows = {id(row): [] for row, *_ in loaded}
            for r in range(rounds):
                order = loaded if r % 2 == 0 else list(reversed(loaded))
                for row, _meta, variant, _o in order:
                    entry = {}
                    for bracketed in ((False, True) if r % 2 == 0 else (True, False)):
                        event, device, host = clock.window(variant, launches, bracketed, rate.value)
                        if bracketed:
                            entry.update(event_ns=event, device_ns=device, host_ns=host)
                        else:
                            entry["plain_event_ns"] = event
                    windows[id(row)].append(entry)
            source = dict(source_commit=manifest["git_head"], worktree_dirty=manifest["git_dirty"])
            for row, meta, variant, _o in loaded:
                ws = windows[id(row)]
                packet, window_refusals = _packet(
                    chip=chip, identity=identity, variant_meta=meta, group=group, windows=ws,
                    launches=launches, rate_khz=rate.value, marker_sha=manifest["marker_sha256"],
                    source=source, run_id=run_id, resources=row["resources"])
                row.update(
                    launches_per_window=launches,
                    device_ns=statistics.median(w["device_ns"] for w in ws),
                    event_ns=statistics.median(w["event_ns"] for w in ws),
                    plain_event_ns=statistics.median(w["plain_event_ns"] for w in ws),
                    per_round_device_ns=[w["device_ns"] for w in ws],
                    shortest_window_ms=min(w["event_ns"] for w in ws) * launches / 1e6,
                    admission_route=packet["admission_route"],
                    eligible_for_promotion=packet["eligible_for_promotion"],
                    ineligibility_reasons=packet["ineligibility_reasons"],
                    diagnostic_gaps=packet["diagnostic_gaps"],
                    marker_overhead_ratio=packet["instrumentation_comparison"]["duration_ratio"],
                    window_refusals=window_refusals,
                    packet_sha256=packet["packet_sha256"])
                row["_packet"] = packet
                variant.close()
            if base is not None:
                base_rounds = base[0]["per_round_device_ns"]
                for row, *_ in loaded:
                    if row is base[0]:
                        continue
                    paired = [u / s for u, s in zip(base_rounds, row["per_round_device_ns"])]
                    row["speedup_vs_unsplit_device_clock"] = statistics.median(paired)
                    row["speedup_vs_unsplit_per_round"] = paired
                    row["rounds_faster"] = sum(p > 1.0 for p in paired)
    finally:
        clock.close()
    return dict(chip=chip, identity=identity, pid=os.getpid(), run_id=run_id, rows=rows)


def record(args):
    from tessera.compiler.llvm_tools import llvm_bin_dir
    from tessera.compiler.native_device_clock import build_device_clock_marker
    from tessera.compiler.scheduled_matmul import find_tessera_opt
    from tessera import runtime as rt

    head, dirty = _git("rev-parse", "HEAD"), bool(_git("status", "--porcelain"))
    if dirty and not args.diagnostic:
        raise SystemExit("source tree has uncommitted changes; commit first or pass --diagnostic")
    chip = rt._rocm_chip()
    if rt._rocm_live_arch() != chip:
        raise SystemExit(f"pinned chip {chip} is not the live device {rt._rocm_live_arch()}")
    tool, llvm_bin = find_tessera_opt(), llvm_bin_dir()
    if tool is None or llvm_bin is None:
        raise SystemExit("tessera-opt or the matched LLVM bin directory was not found")
    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",") if s]
    dtypes = [d for d in args.dtypes.split(",") if d]
    slices = [int(v) for v in args.slices.split(",") if v]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="splitk-sweep-") as directory:
        marker = build_device_clock_marker(compiler=tool, llvm_bin=llvm_bin, backend="rocm", chip=chip)
        marker_path = Path(directory) / "marker.hsaco"
        marker_path.write_bytes(marker.image)
        groups = []
        for dtype in dtypes:
            for shape in shapes:
                groups.append(build_variants(chip, shape, dtype, slices, llvm_bin, directory))
                print(f"built {dtype} {'x'.join(map(str, shape))}", flush=True)
        manifest = dict(chip=chip, git_head=head, git_dirty=dirty, marker_image=str(marker_path),
                        marker_entry=marker.entry, marker_sha256=marker.image_sha256, groups=groups)
        manifest_path = Path(directory) / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        results = []
        for run in range(args.runs):
            child = Path(directory) / f"run{run}.json"
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker",
                            str(manifest_path), "--output-dir", str(child),
                            "--rounds", str(args.rounds), "--window-ms", str(args.window_ms)],
                           check=True)
            results.append(json.loads(child.read_text()))
            print(f"completed run {run + 1}/{args.runs}", flush=True)
    if _git("rev-parse", "HEAD") != head or bool(_git("status", "--porcelain")) != dirty:
        raise SystemExit("source state changed while collecting evidence")
    with (out / "packets.jsonl").open("w") as handle:
        for run, result in enumerate(results):
            for row in result["rows"]:
                packet = row.pop("_packet", None)
                if packet is not None:
                    handle.write(json.dumps(dict(run=run, dtype=row["dtype"], shape=row["shape"],
                                                 split_k=row["split_k"], packet=packet)) + "\n")
    for group in groups:
        for variant in group["variants"]:
            variant.pop("image", None)
    summary = summarize(results)
    packet = dict(
        item="ROCM-SPLIT-K-1", sync="GFX1201-LANES-2026-09-27", host=platform.node(), chip=chip,
        git_head=head, git_dirty=dirty, tessera_opt=str(tool),
        tessera_opt_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),
        llvm_bin=str(llvm_bin), marker_sha256=marker.image_sha256,
        timing_source="device_wall_clock_ns (compiler-built marker, llvm.readsteadycounter)",
        protocol=dict(rounds=args.rounds, window_ms_target=args.window_ms, runs=args.runs,
                      fresh_process_per_run=True, interleaved=True,
                      reduce_threads=REDUCE_THREADS, workspace="allocated once, outside timing"),
        groups=groups, summary=summary, runs=results)
    (out / "sweep.json").write_text(json.dumps(packet, indent=1) + "\n")
    for line in format_summary(summary):
        print(line)


def summarize(results):
    """Per (dtype, shape, split_k): medians over runs, and the admission tally."""
    table = {}
    for result in results:
        for row in result["rows"]:
            key = (row["dtype"], "x".join(map(str, row["shape"])), row["split_k"])
            table.setdefault(key, []).append(row)
    summary = []
    for (dtype, shape, slices), rows in sorted(table.items(), key=lambda kv: (
            kv[0][0], [int(v) for v in kv[0][1].split("x")], kv[0][2])):
        entry = dict(dtype=dtype, shape=shape, split_k=slices, production=rows[0]["production"])
        refused = [r["refused"] for r in rows if "refused" in r]
        if refused:
            entry["refused"] = refused[0]
            summary.append(entry)
            continue
        entry.update(
            device_us=statistics.median(r["device_ns"] for r in rows) / 1e3,
            admitted_runs=sum(bool(r["eligible_for_promotion"]) for r in rows),
            runs=len(rows),
            routes=sorted({r["admission_route"] for r in rows}),
            reasons=sorted({x for r in rows for x in r["ineligibility_reasons"]}),
            max_relative_error_vs_f64=max(r["relative_error_vs_f64"] for r in rows),
            bit_identical_reruns=all(r["bit_identical_reruns"] for r in rows),
            shortest_window_ms=min(r["shortest_window_ms"] for r in rows))
        speed = [r["speedup_vs_unsplit_device_clock"] for r in rows
                 if "speedup_vs_unsplit_device_clock" in r]
        if speed:
            entry.update(
                speedup_median=statistics.median(speed), speedup_min_run=min(speed),
                rounds_faster=sum(r["rounds_faster"] for r in rows),
                rounds=sum(len(r["speedup_vs_unsplit_per_round"]) for r in rows),
                max_relative_diff_vs_unsplit=max(r["relative_diff_vs_unsplit"] for r in rows))
        summary.append(entry)
    return summary


def format_summary(summary):
    for e in summary:
        head = f"{e['dtype']} {e['shape']:>12} S={e['split_k']:<2}{'*' if e['production'] else ' '}"
        if "refused" in e:
            yield f"{head} refused: {e['refused'][:120]}"
            continue
        speed = (f" x{e['speedup_median']:.3f} (min run {e['speedup_min_run']:.3f},"
                 f" {e['rounds_faster']}/{e['rounds']} rounds)") if "speedup_median" in e else ""
        yield (f"{head} {e['device_us']:8.2f} us{speed} admitted {e['admitted_runs']}/{e['runs']}"
               f" rel_err {e['max_relative_error_vs_f64']:.1e}"
               f" window>={e['shortest_window_ms']:.1f}ms"
               + (f" reasons={e['reasons']}" if e["reasons"] else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shapes", default=DEFAULT_SHAPES)
    parser.add_argument("--dtypes", default="fp16,bf16")
    parser.add_argument("--slices", default="1,2,4,8,16")
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--window-ms", type=float, default=10.0)
    parser.add_argument("--diagnostic", action="store_true",
                        help="allow a dirty tree (packets record it and cannot promote)")
    args = parser.parse_args()
    if args.worker:
        result = worker(args.worker, args.rounds, args.window_ms)
        args.output_dir.write_text(json.dumps(result))
        return
    record(args)


if __name__ == "__main__":
    main()
