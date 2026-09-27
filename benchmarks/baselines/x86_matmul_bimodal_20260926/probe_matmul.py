#!/usr/bin/env python3
"""X86-MATMUL-BIMODAL-1 probe: one fresh process, the recorder's matmul timing.

Mirrors record_x86_avx512_packet.py for the matmul family: same packager, same
256^3 f32 bindings (default_rng(20260926)), same direct C-ABI call, 15 samples
x 100 iterations. Timed with perf_counter_ns (the level is a ~1.45x effect).
Prints one JSON line with the per-process median and per-process variables.

Toggles (one at a time): --offset-b/--offset-a/--offset-o BYTES place the
array at that byte offset from a 4096-aligned base (default: numpy's own
allocation, i.e. the pre-fix recorder's behaviour); --cpu N pins the timed region.
"""
from __future__ import annotations

import argparse, ctypes, json, os, statistics, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("TESSERA_ROOT", Path.cwd()))
sys.path[:0] = [str(ROOT), str(ROOT / "python")]


def placed(src: np.ndarray, offset: int | None) -> np.ndarray:
    if offset is None:
        return src
    raw = np.empty(src.nbytes + 8192 + offset, dtype=np.uint8)
    base = (-raw.ctypes.data) % 4096
    view = raw[base + offset: base + offset + src.nbytes].view(np.float32).reshape(src.shape)
    view[...] = src
    return view


def smaps_for(addr: int) -> dict:
    cur = None
    with open("/proc/self/smaps") as fh:
        for line in fh:
            head = line.split()
            if "-" in head[0] and len(head) >= 5 and ":" not in head[0]:
                lo, hi = (int(x, 16) for x in head[0].split("-"))
                cur = {"lo": lo, "hi": hi} if lo <= addr < hi else None
            elif cur is not None and head[0] in ("AnonHugePages:", "Rss:", "THPeligible:"):
                cur[head[0].rstrip(":")] = head[1]
            if cur is not None and head[0] == "VmFlags:":
                cur["VmFlags"] = " ".join(head[1:])
                return {"map_lo": hex(cur["lo"]), "addr_minus_map": addr - cur["lo"],
                        **{k: v for k, v in cur.items() if k not in ("lo", "hi")}}
    return {}


def image_base() -> str:
    with open("/proc/self/maps") as fh:
        for line in fh:
            if "tessera-x86" in line or "memfd" in line:
                return line.split("-")[0]
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset-a", type=int)
    ap.add_argument("--offset-b", type=int)
    ap.add_argument("--offset-o", type=int)
    ap.add_argument("--cpu", type=int)
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--tag", default="")
    ap.add_argument("--recorder-bindings", action="store_true",
                    help="time the recorder's own matmul timing bindings (after the fix)")
    args = ap.parse_args()

    from tessera import runtime as rt
    from tessera.compiler.x86_native import package_matmul
    sys.path.insert(0, str(ROOT / "benchmarks" / "e2e_spine"))
    import record_x86_avx512_packet as rec

    rng = np.random.default_rng(20260926)
    ta = np.ascontiguousarray(rng.standard_normal((256, 256)), dtype=np.float32)
    tb = np.ascontiguousarray(rng.standard_normal((256, 256)), dtype=np.float32)
    a = placed(ta, args.offset_a)
    b = placed(tb, args.offset_b)
    o = placed(np.zeros((256, 256), np.float32), args.offset_o)
    bindings = {"a": a, "b": b, "o": o, "M": 256, "N": 256, "K": 256}
    if args.recorder_bindings:
        from tessera.compiler.e2e_fleet import load_fixture_corpus
        spec = next(d for d in rec._family_definitions(load_fixture_corpus())
                    if d["family"] == "matmul")
        bindings = spec["timing_bindings"]
        a, b, o = bindings["a"], bindings["b"], bindings["o"]
        ta, tb = a.copy(), b.copy()
    pkg = package_matmul(rec._matmul_module(256, 256, 256), pipeline_name=rec.PIPELINE)
    art = rt.RuntimeArtifact(metadata={"target": "x86", "architecture": "x86_64_avx512"},
                             native_image=pkg.image, launch_descriptor=pkg.descriptor,
                             tile_ir=pkg.tile_ir, target_ir=pkg.target_ir)
    res = rt.launch(art, bindings)
    assert res["ok"] and res.get("execution_kind") == "native_cpu", res
    err = float(np.max(np.abs(o.astype(np.float64) - ta.astype(np.float64) @ tb.astype(np.float64))))
    assert err <= 2e-3, err
    call = rec._direct_call("matmul", pkg, bindings)
    libc = ctypes.CDLL(None)
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    call()
    per, cpus = [], []
    for _ in range(args.samples):
        t0 = time.perf_counter_ns()
        for _ in range(args.iterations):
            call()
        per.append((time.perf_counter_ns() - t0) / args.iterations)
        cpus.append(libc.sched_getcpu())
    out = {
        "tag": args.tag, "host": os.uname().nodename, "pid": os.getpid(),
        "median_us": statistics.median(per) / 1e3,
        "min_us": min(per) / 1e3, "max_us": max(per) / 1e3,
        "cpus": sorted(set(cpus)),
        "a": hex(a.ctypes.data), "b": hex(b.ctypes.data), "o": hex(o.ctypes.data),
        "a_mod4096": a.ctypes.data % 4096, "b_mod4096": b.ctypes.data % 4096,
        "o_mod4096": o.ctypes.data % 4096,
        "smaps_b": smaps_for(b.ctypes.data), "smaps_a": smaps_for(a.ctypes.data),
        "smaps_o": smaps_for(o.ctypes.data),
        "image_base": image_base(),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
