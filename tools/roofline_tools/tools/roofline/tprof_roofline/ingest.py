from typing import List, Dict, Tuple
import csv, json, re
from .model import KernelSample, CommEvent

def read_kernels_csv(path: str) -> List[KernelSample]:
    samples = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(KernelSample(
                name=row.get("name","kernel"),
                flop_count=float(row["flops"]),
                dram_bytes=float(row["dram_bytes"]),
                time_ms=float(row["time_ms"]),
                dtype_key=row.get("dtype_key","fp32"),
                meta={k:v for k,v in row.items() if k not in {"name","flops","dram_bytes","time_ms","dtype_key"}},
            ))
    return samples

def read_perfetto_trace(path: str) -> Tuple[List[KernelSample], List[CommEvent]]:
    """Perfetto-like JSON with compute + comm events.
    Compute events: {type:'compute', name, flops, dram_bytes, dur_us, dtype_key}
    Comm events:    {type:'comm', name, bytes, dur_us, link}  # link in {'NVLink','PCIe','NIC',...}
    """
    with open(path, "r") as f:
        data = json.load(f)
    kernels, comms = [], []
    for ev in data.get("events", []):
        t = ev.get("type")
        if t == "compute":
            kernels.append(KernelSample(
                name=ev.get("name","compute"),
                flop_count=float(ev.get("flops", 0.0)),
                dram_bytes=float(ev.get("dram_bytes", 0.0)),
                time_ms=float(ev.get("dur_us", 0.0))/1000.0,
                dtype_key=ev.get("dtype_key","fp32"),
                meta={k:v for k,v in ev.items() if k not in {"name","flops","dram_bytes","dur_us","dtype_key","type"}},
            ))
        elif t in ("comm","copy","a2a","p2p"):
            comms.append(CommEvent(
                name=ev.get("name", t),
                bytes=float(ev.get("bytes", ev.get("size", 0.0))),
                time_ms=float(ev.get("dur_us", 0.0))/1000.0,
                link=str(ev.get("link", ev.get("bus","unknown"))),
                meta={k:v for k,v in ev.items() if k not in {"name","bytes","size","dur_us","link","bus","type"}},
            ))
    return kernels, comms

def read_nsight_compute_csv(path: str) -> List[KernelSample]:
    """Heuristic parser for Nsight Compute 'Kernel Profile' CSV export.
    Looks for bytes (dram__bytes.*), FLOP counts (flop_count_*), and duration (Duration or gpu__time_duration).
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Identify columns
    cols = rows[0].keys() if rows else []
    # Bytes
    byte_cols = [c for c in cols if re.search(r"dram__bytes(\.|$)", c)]
    # FLOPs: prefer aggregated 'flop_count_*' if available; else sum common components
    flops_cols = [c for c in cols if c.startswith("flop_count_")]
    # Duration
    dur_col = None
    for cand in ("Duration", "duration", "gpu__time_duration.sum", "gpu__time_duration"):
        if cand in cols:
            dur_col = cand
            break

    samples: List[KernelSample] = []
    for row in rows:
        name = row.get("Name") or row.get("Kernel Name") or row.get("Kernel Name Short") or row.get("name") or "kernel"
        # Bytes
        dram_bytes = 0.0
        for c in byte_cols:
            try:
                dram_bytes += float(row[c])
            except: pass
        # FLOPs
        flops = 0.0
        if flops_cols:
            for c in flops_cols:
                try: flops += float(row[c])
                except: pass
        else:
            # Try common components
            for c in ["flop_count_sp","flop_count_hp","flop_count_dp"]:
                if c in row:
                    try: flops += float(row[c])
                    except: pass
        # Time (ms)
        time_ms = 0.0
        if dur_col:
            try:
                time_ms = float(row[dur_col])
                # Heuristic: many exports have ms already; if it's too small, assume it's us
                if time_ms < 1e-3:
                    time_ms *= 1e3
            except: pass
        samples.append(KernelSample(name=name, flop_count=flops, dram_bytes=dram_bytes, time_ms=time_ms))
    return samples


#: Decision #12 stable benchmark-row fields (never removed or repurposed).
BENCHMARK_STABLE_FIELDS = (
    "backend", "op", "shape", "dtype", "latency_ms", "tflops",
    "memory_bw_gb_s", "device", "tessera_version",
)
#: The additive amendment fields. A row written before them is reported as
#: ``unknown`` -- never guessed from ``backend``, ``compiler_path`` or a name.
BENCHMARK_PROVENANCE_FIELDS = ("route", "route_source", "timing_source")
UNKNOWN = "unknown"


def _benchmark_rows(data) -> list:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("rows", "results"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError("benchmark JSON must be a list of rows or carry a 'rows'/'results' list")


def read_benchmark_json(path: str) -> List[KernelSample]:
    """Decision #12 benchmark rows (``benchmarks/run_all.py`` ``rows``, or any
    list of stable-schema rows) as kernel samples.

    FLOPs and DRAM bytes are recovered from the stable fields
    (``tflops * latency``, ``memory_bw_gb_s * latency``); a field that is
    absent or null contributes 0. ``route`` / ``route_source`` /
    ``timing_source`` land in ``meta``; an old row without them loads with
    each set to ``"unknown"``.
    """
    with open(path) as f:
        data = json.load(f)
    samples: List[KernelSample] = []
    for row in _benchmark_rows(data):
        if not isinstance(row, dict) or row.get("latency_ms") is None:
            continue
        time_ms = float(row["latency_ms"])
        seconds = time_ms * 1e-3
        tflops = row.get("tflops")
        bw = row.get("memory_bw_gb_s")
        meta = {k: v for k, v in row.items()
                if k not in {"latency_ms", "tflops", "memory_bw_gb_s"}}
        for field_name in BENCHMARK_PROVENANCE_FIELDS:
            if not meta.get(field_name):
                meta[field_name] = UNKNOWN
        shape = row.get("shape", "")
        samples.append(KernelSample(
            name=f"{row.get('op', 'kernel')}{list(shape) if isinstance(shape, (list, tuple)) else shape}",
            flop_count=float(tflops) * 1e12 * seconds if tflops is not None else 0.0,
            dram_bytes=float(bw) * 1e9 * seconds if bw is not None else 0.0,
            time_ms=time_ms,
            dtype_key=str(row.get("dtype", "fp32")),
            meta=meta,
        ))
    return samples
