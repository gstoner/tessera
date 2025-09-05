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
