#!/usr/bin/env python3
"""The shared TileToROCM bounded store, before vs after, on the live chip.

ROCM-FP8-BLOCKSCALE-1 follow-up (sync ``FOUNDATION-BATCH-3-2026-09-28``). The
lane-relative bounded store (``materializeFragmentStore``,
FOUNDATION-BATCH-2-2026-09-27) is shared by every typed scheduled-matmul store
on RDNA, gfx1151 included, and was proven correct there but never timed. This
recorder times the production scheduled matmul of each shape as compiled by
two complete trees -- ``before`` (Python + ``tessera-opt`` of one checkout)
and ``after`` (another) -- paired and interleaved in one process.

Two modes, because two trees cannot share one interpreter:

* ``--compile DIR`` lowers and packages every ``--shapes`` x ``--dtypes``
  entry with whatever ``tessera`` is first on ``sys.path`` (``--tessera-root``
  puts a checkout's ``python/`` there) and whatever ``TESSERA_OPT`` names,
  writing ``<dtype>_<M>x<N>x<K>.hsaco`` plus ``manifest.json`` (symbol, macro
  tile, workgroup, route, ISA digest).
* ``--time BEFORE AFTER`` loads both manifests, skips shapes whose instruction
  streams are identical (recorded as ``isa_identical``), checks every kept
  kernel against an f64 reference and requires before/after outputs to be
  bitwise equal (same elements, same arithmetic -- only the store's index
  arithmetic changed), then times them in ``--windows`` interleaved windows
  (ABAB then BABA) per fresh process (``--runs``, alternating arm order).

Timing (sync ``WSL-TIMING-ADMISSION-2026-09-26``): each window is bracketed by
the compiler-built device-clock marker (``llvm.readsteadycounter`` at
``hipDeviceAttributeWallClockRate``) and a HIP event pair; a window is
admissible when it is >= 5 ms and the two agree within 5%. The device clock is
the reported source; the ratio is the median of per-window-pair after/before.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
P = ct.c_void_p


def _graph_module(m, n, k, dtype):
    from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType

    element = {"fp16": "f16", "bf16": "bf16", "int8": "i8"}[dtype]
    a = IRType(f"tensor<{m}x{k}x{element}>", (str(m), str(k)), dtype)
    b = IRType(f"tensor<{k}x{n}x{element}>", (str(k), str(n)), dtype)
    out = (IRType(f"tensor<{m}x{n}xi32>", (str(m), str(n)), "int32") if dtype == "int8"
           else IRType(f"tensor<{m}x{n}xf32>", (str(m), str(n)), "fp32"))
    return GraphIRModule(functions=[GraphIRFunction(
        name="rocm_bounded_store_timing", args=[IRArg("a", a), IRArg("b", b)],
        result_types=[out],
        body=[IROp(result="o", op_name="tessera.matmul", operands=["%a", "%b"],
                   operand_types=[str(a), str(b)], result_type=str(out),
                   kwargs={"activation": "none"})],
        return_values=["%o"])])


def _isa(image: bytes, llvm_bin: Path) -> tuple[str, dict]:
    with tempfile.TemporaryDirectory(prefix="b3perf-isa-") as tmp:
        obj = Path(tmp) / "k.hsaco"
        obj.write_bytes(image)
        text = subprocess.run([str(llvm_bin / "llvm-objdump"), "-d", str(obj)], check=True,
                              capture_output=True, text=True, timeout=120).stdout
        notes = subprocess.run([str(llvm_bin / "llvm-readelf"), "--notes", str(obj)], check=True,
                               capture_output=True, text=True, timeout=120).stdout
    body = [ln.split("//")[0].strip() for ln in text.splitlines()
            if ln.startswith("\t") or ln.startswith(" ")]
    body = [ln for ln in body if ln]
    census = {"instructions": len(body)}
    for key in ("vgpr_count", "sgpr_count", "vgpr_spill_count", "private_segment_fixed_size",
                "group_segment_fixed_size"):
        found = [ln for ln in notes.splitlines() if f".{key}:" in ln]
        if found:
            census[key] = int(found[0].split(":")[-1].strip())
    census["global_store"] = sum("global_store" in ln for ln in body)
    return hashlib.sha256("\n".join(body).encode()).hexdigest(), census


def compile_mode(args):
    if args.tessera_root:
        sys.path[:0] = [str(args.tessera_root), str(args.tessera_root / "python")]
    from tessera.compiler import rocm_native, scheduled_matmul
    from tessera.compiler.llvm_tools import llvm_bin_dir

    llvm_bin = llvm_bin_dir()
    out = Path(args.compile)
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for dtype in args.dtypes.split(","):
        for spec in args.shapes.split(","):
            m, n, k = (int(x) for x in spec.split("x"))
            entry = dict(dtype=dtype, shape=[m, n, k])
            entries.append(entry)
            try:
                artifact = scheduled_matmul.lower_scheduled_matmul(
                    _graph_module(m, n, k, dtype), target=f"rocm_{args.chip}")
                package = rocm_native.package_scheduled_matmul(
                    artifact, pipeline_name="tessera-lower-to-rocm")
            except Exception as exc:  # the refusal is the row
                entry["refused"] = str(exc)[:300]
                continue
            prov = package.descriptor.provenance
            path = out / f"{dtype}_{m}x{n}x{k}.hsaco"
            path.write_bytes(package.image.payload)
            isa_sha, census = _isa(package.image.payload, llvm_bin)
            if int(artifact.split_k) > 1:
                entry["refused"] = "split-K production schedule (two launches) not driven here"
                continue
            entry.update(image=str(path), symbol=package.descriptor.entry_symbol,
                         macro_tile=list(prov["macro_tile"]),
                         workgroup=list(prov.get("workgroup", [32, 1, 1])),
                         route=prov["route"], physical_route=prov.get("physical_route"),
                         staging=prov.get("staging"), isa_sha256=isa_sha, census=census,
                         image_sha256=hashlib.sha256(package.image.payload).hexdigest())
    manifest = dict(chip=args.chip, tessera_root=str(args.tessera_root or ROOT),
                    tessera_opt=os.environ.get("TESSERA_OPT"),
                    git_head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.tessera_root or ROOT,
                                            capture_output=True, text=True).stdout.strip(),
                    entries=entries)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps([(e["dtype"], e["shape"], e.get("census"), e.get("refused")) for e in entries],
                     indent=None))


class Hip:
    def __init__(self):
        root = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
        self.lib = ct.CDLL(str(root / "lib" / "libamdhip64.so"))
        self.ok(self.lib.hipInit(0))
        self.lib.hipModuleLaunchKernel.argtypes = [P, *[ct.c_uint] * 7, P, P, P]
        self.lib.hipEventElapsedTime.argtypes = [ct.POINTER(ct.c_float), P, P]
        self.lib.hipMemcpy.argtypes = [P, P, ct.c_size_t, ct.c_int]
        self.lib.hipMalloc.argtypes = [ct.POINTER(P), ct.c_size_t]
        self.lib.hipMemset.argtypes = [P, ct.c_int, ct.c_size_t]
        self.lib.hipDeviceGetAttribute.argtypes = [ct.POINTER(ct.c_int), ct.c_int, ct.c_int]

    @staticmethod
    def ok(rc):
        if rc:
            raise RuntimeError(f"HIP call failed rc={rc}")

    def upload(self, array):
        array = np.ascontiguousarray(array)
        ptr = P()
        self.ok(self.lib.hipMalloc(ct.byref(ptr), max(array.nbytes, 1)))
        self.ok(self.lib.hipMemcpy(ptr, array.ctypes.data_as(P), array.nbytes, 1))
        return ptr


def _memref(ptr, size):
    return [P(ptr.value), P(ptr.value), ct.c_int64(0), ct.c_int64(size), ct.c_int64(1)]


class Kernel:
    def __init__(self, hip, entry, a_dev, b_dev, m, n, k):
        self.hip, self.m, self.n = hip, m, n
        self.blob = ct.create_string_buffer(Path(entry["image"]).read_bytes())
        self.mod, self.fn = P(), P()
        hip.ok(hip.lib.hipModuleLoadData(ct.byref(self.mod), ct.cast(self.blob, P)))
        hip.ok(hip.lib.hipModuleGetFunction(ct.byref(self.fn), self.mod, entry["symbol"].encode()))
        self.out = P()
        hip.ok(hip.lib.hipMalloc(ct.byref(self.out), 4 * m * n))
        self.args = (_memref(a_dev, m * k) + _memref(b_dev, k * n) + _memref(self.out, m * n)
                     + [ct.c_int64(m), ct.c_int64(n), ct.c_int64(k)])
        self.argv = (P * len(self.args))(*[ct.cast(ct.byref(v), P) for v in self.args])
        bm, bn = entry["macro_tile"]
        self.grid = ((n + bn - 1) // bn, (m + bm - 1) // bm, 1)
        self.block = tuple(int(x) for x in entry["workgroup"])

    def __call__(self):
        self.hip.ok(self.hip.lib.hipModuleLaunchKernel(self.fn, *self.grid, *self.block, 0, None,
                                                       self.argv, None))

    def result(self):
        self.hip.ok(self.hip.lib.hipMemset(self.out, 0xFF, 4 * self.m * self.n))
        self()
        self.hip.ok(self.hip.lib.hipDeviceSynchronize())
        host = np.empty((self.m, self.n), np.float32)
        self.hip.ok(self.hip.lib.hipMemcpy(host.ctypes.data_as(P), self.out, host.nbytes, 2))
        return host


def time_worker(args):
    sys.path[:0] = [str(ROOT), str(ROOT / "python")]
    import ml_dtypes
    from tessera.compiler.llvm_tools import llvm_bin_dir
    from tessera.compiler.native_device_clock import build_device_clock_marker

    before = json.loads((Path(args.time[0]) / "manifest.json").read_text())
    after = json.loads((Path(args.time[1]) / "manifest.json").read_text())
    chip = after["chip"]
    marker = build_device_clock_marker(compiler=Path(os.environ["TESSERA_OPT"]),
                                       llvm_bin=llvm_bin_dir(), backend="rocm", chip=chip)
    hip = Hip()
    from benchmarks.record_ssd_gpu import _hip_enum
    rate = ct.c_int()
    hip.ok(hip.lib.hipDeviceGetAttribute(ct.byref(rate), _hip_enum("hipDeviceAttributeWallClockRate"), 0))
    mblob = ct.create_string_buffer(marker.image)
    mmod, mfn, span = P(), P(), P()
    hip.ok(hip.lib.hipModuleLoadData(ct.byref(mmod), ct.cast(mblob, P)))
    hip.ok(hip.lib.hipModuleGetFunction(ct.byref(mfn), mmod, marker.entry.encode()))
    hip.ok(hip.lib.hipMalloc(ct.byref(span), 16))
    margv = (P * 1)(ct.cast(ct.byref(span), P))
    events = [P(), P()]
    for e in events:
        hip.ok(hip.lib.hipEventCreate(ct.byref(e)))
    host_span = (ct.c_uint64 * 2)()

    def window(kernel, count):
        host_span[0], host_span[1] = (1 << 64) - 1, 0
        hip.ok(hip.lib.hipMemcpy(span, ct.addressof(host_span), 16, 1))
        hip.ok(hip.lib.hipDeviceSynchronize())
        hip.ok(hip.lib.hipEventRecord(events[0], None))
        hip.ok(hip.lib.hipModuleLaunchKernel(mfn, 1, 1, 1, 1, 1, 1, 0, None, margv, None))
        for _ in range(count):
            kernel()
        hip.ok(hip.lib.hipModuleLaunchKernel(mfn, 1, 1, 1, 1, 1, 1, 0, None, margv, None))
        hip.ok(hip.lib.hipEventRecord(events[1], None))
        hip.ok(hip.lib.hipEventSynchronize(events[1]))
        ms = ct.c_float()
        hip.ok(hip.lib.hipEventElapsedTime(ct.byref(ms), events[0], events[1]))
        hip.ok(hip.lib.hipMemcpy(ct.addressof(host_span), span, 16, 2))
        if host_span[0] == (1 << 64) - 1 or host_span[1] <= host_span[0]:
            raise RuntimeError("device-clock span not written")
        device_ns = (host_span[1] - host_span[0]) * 1e6 / rate.value
        return dict(device_ns=device_ns, event_ns=ms.value * 1e6, launches=count)

    rows = []
    by_key = {(e["dtype"], tuple(e["shape"])): e for e in before["entries"]}
    order_flip = args.worker_index % 2 == 1
    for entry_after in after["entries"]:
        key = (entry_after["dtype"], tuple(entry_after["shape"]))
        entry_before = by_key.get(key)
        row = dict(dtype=key[0], shape=list(key[1]))
        rows.append(row)
        if entry_before is None or "refused" in entry_after or "refused" in entry_before:
            row["skipped"] = "refused or missing in one tree"
            continue
        row.update(route_before=entry_before["route"], route_after=entry_after["route"],
                   census_before=entry_before["census"], census_after=entry_after["census"],
                   isa_before=entry_before["isa_sha256"], isa_after=entry_after["isa_sha256"])
        if entry_before["isa_sha256"] == entry_after["isa_sha256"]:
            row["isa_identical"] = True
            continue
        m, n, k = key[1]
        rng = np.random.default_rng(1151)
        if key[0] == "int8":
            a = rng.integers(-128, 128, size=(m, k), dtype=np.int8)
            b = rng.integers(-128, 128, size=(k, n), dtype=np.int8)
        else:
            storage = np.float16 if key[0] == "fp16" else ml_dtypes.bfloat16
            a = (rng.standard_normal((m, k)) * 0.25).astype(storage)
            b = (rng.standard_normal((k, n)) * 0.25).astype(storage)
        a_dev, b_dev = hip.upload(a), hip.upload(b)
        arms = {"before": Kernel(hip, entry_before, a_dev, b_dev, m, n, k),
                "after": Kernel(hip, entry_after, a_dev, b_dev, m, n, k)}
        outs = {name: kern.result() for name, kern in arms.items()}
        if key[0] == "int8":
            outs = {name: o.view(np.int32) for name, o in outs.items()}
        ref = a.astype(np.float64) @ b.astype(np.float64)
        scale = float(np.abs(ref).max()) + 1e-6
        row["relative_error_vs_f64"] = {name: float(np.abs(o - ref).max() / scale)
                                        for name, o in outs.items()}
        row["bitwise_equal_before_after"] = bool(np.array_equal(
            outs["before"].view(np.uint32), outs["after"].view(np.uint32)))
        if key[0] == "int8":  # exact integer contract
            row["exact_vs_int64"] = {name: bool(np.array_equal(o.astype(np.int64),
                                                               a.astype(np.int64) @ b.astype(np.int64)))
                                     for name, o in outs.items()}
        if (not row["bitwise_equal_before_after"]
                or max(row["relative_error_vs_f64"].values()) > 2e-3
                or not all(row.get("exact_vs_int64", {}).values())):
            row["skipped"] = "correctness check failed; not timed"
            continue
        names = ["after", "before"] if order_flip else ["before", "after"]
        warm = time.perf_counter() + 0.5
        while time.perf_counter() < warm:
            for kern in arms.values():
                for _ in range(10):
                    kern()
            hip.ok(hip.lib.hipDeviceSynchronize())
        probe = window(arms["before"], 50)
        count = max(50, math.ceil(1.3 * args.window_ms * 1e6 / (probe["device_ns"] / 50)))
        samples = {name: [] for name in names}
        for w in range(args.windows):
            for name in (names if w % 2 == 0 else names[::-1]):
                samples[name].append(window(arms[name], count))
        for name, ws in samples.items():
            per = [s["device_ns"] / s["launches"] for s in ws]
            agree = [abs(s["device_ns"] - s["event_ns"]) / s["event_ns"] for s in ws]
            row[name] = dict(median_us=statistics.median(per) / 1e3,
                             event_median_us=statistics.median(
                                 s["event_ns"] / s["launches"] for s in ws) / 1e3,
                             window_ms_min=min(s["device_ns"] for s in ws) / 1e6,
                             device_event_disagreement_max=max(agree),
                             admissible=min(s["device_ns"] for s in ws) >= 5e6 and max(agree) <= 0.05,
                             per_window_us=[p / 1e3 for p in per])
        pair = [a_["device_ns"] / b_["device_ns"] for a_, b_ in zip(samples["after"], samples["before"])]
        row["after_over_before_median"] = statistics.median(pair)
        row["after_over_before_range"] = [min(pair), max(pair)]
        row["launches_per_window"] = count
    return dict(pid=os.getpid(), worker_index=args.worker_index, rows=rows,
                marker_sha256=hashlib.sha256(marker.image).hexdigest(), wall_clock_khz=rate.value)


def time_mode(args):
    runs = []
    for index in range(args.runs):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            out = Path(tmp.name)
        subprocess.run([sys.executable, __file__, "--time", *args.time, "--worker-index", str(index),
                        "--windows", str(args.windows), "--window-ms", str(args.window_ms),
                        "--worker-output", str(out)], check=True, timeout=3600)
        runs.append(json.loads(out.read_text()))
        out.unlink()
    before = json.loads((Path(args.time[0]) / "manifest.json").read_text())
    after = json.loads((Path(args.time[1]) / "manifest.json").read_text())
    summary = []
    for i, row in enumerate(runs[0]["rows"]):
        entry = dict(dtype=row["dtype"], shape=row["shape"])
        for key in ("skipped", "isa_identical", "route_before", "route_after", "census_before",
                    "census_after"):
            if key in row:
                entry[key] = row[key]
        timed = [r["rows"][i] for r in runs if "after_over_before_median" in r["rows"][i]]
        if timed:
            entry["after_over_before_per_run"] = [r["after_over_before_median"] for r in timed]
            entry["before_us_per_run"] = [r["before"]["median_us"] for r in timed]
            entry["after_us_per_run"] = [r["after"]["median_us"] for r in timed]
            entry["all_admissible"] = all(r[a]["admissible"] for r in timed for a in ("before", "after"))
            entry["bitwise_equal_before_after"] = all(r["bitwise_equal_before_after"] for r in timed)
            entry["relative_error_vs_f64"] = timed[0]["relative_error_vs_f64"]
        summary.append(entry)
    record = dict(schema="tessera.rocm_bounded_store_before_after.v1",
                  sync_key="FOUNDATION-BATCH-3-2026-09-28",
                  host=platform.node(), chip=after["chip"],
                  before=dict(git_head=before["git_head"], tessera_opt=before["tessera_opt"]),
                  after=dict(git_head=after["git_head"], tessera_opt=after["tessera_opt"]),
                  timing_source="device_clock_marker (llvm.readsteadycounter), HIP event witness",
                  admission="WSL-TIMING-ADMISSION-2026-09-26: window >= 5 ms, |device-event| <= 5%",
                  summary=summary, runs=runs)
    Path(args.output).write_text(json.dumps(record, indent=1))
    for entry in summary:
        print(entry["dtype"], entry["shape"], entry.get("skipped") or
              ("isa_identical" if entry.get("isa_identical") else
               f"after/before {['%.3f' % x for x in entry.get('after_over_before_per_run', [])]} "
               f"before_us {['%.2f' % x for x in entry.get('before_us_per_run', [])]} "
               f"adm={entry.get('all_admissible')}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", type=Path)
    parser.add_argument("--tessera-root", type=Path)
    parser.add_argument("--chip", default=os.environ.get("TESSERA_ROCM_CHIP", "gfx1151"))
    parser.add_argument("--shapes", default="1024x1024x1024,1000x1000x1024,4096x4096x4096")
    parser.add_argument("--dtypes", default="fp16,bf16")
    parser.add_argument("--time", nargs=2)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--window-ms", type=float, default=6.0)
    parser.add_argument("--worker-index", type=int, default=-1)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.compile:
        compile_mode(args)
    elif args.time and args.worker_index >= 0:
        Path(args.worker_output).write_text(json.dumps(time_worker(args)))
    elif args.time:
        time_mode(args)
    else:
        parser.error("--compile or --time")


if __name__ == "__main__":
    main()
