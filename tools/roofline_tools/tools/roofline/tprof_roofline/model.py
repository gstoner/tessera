from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import yaml

@dataclass
class CommLink:
    name: str
    bw_GBps: float  # peak uni-directional bandwidth (GB/s)

@dataclass
class DevicePeaks:
    name: str
    hbm_bw_GBps: float
    compute_peaks_GFLOPs: Dict[str, float]
    comm_links: List[CommLink] = field(default_factory=list)

    @staticmethod
    def from_yaml(path: str) -> "DevicePeaks":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        links = []
        for L in data.get("links", []) or []:
            links.append(CommLink(name=str(L.get("name","link")), bw_GBps=float(L["bw_GBps"])))
        return DevicePeaks(
            name=data["name"],
            hbm_bw_GBps=float(data["hbm_bw_GBps"]),
            compute_peaks_GFLOPs={k: float(v) for k, v in data["compute_peaks_GFLOPs"].items()},
            comm_links=links,
        )

@dataclass
class KernelSample:
    name: str
    flop_count: float  # FLOPs
    dram_bytes: float  # bytes
    time_ms: float     # milliseconds
    dtype_key: str = "fp32"
    meta: Dict = field(default_factory=dict)

    @property
    def time_s(self) -> float:
        return self.time_ms / 1e3

    @property
    def achieved_GFLOPs(self) -> float:
        if self.time_s <= 0:
            return 0.0
        return (self.flop_count / 1e9) / self.time_s

    @property
    def achieved_GBps(self) -> float:
        if self.time_s <= 0:
            return 0.0
        return (self.dram_bytes / 1e9) / self.time_s

    @property
    def operational_intensity(self) -> float:
        if self.dram_bytes <= 0:
            return math.inf
        return (self.flop_count / 1e9) / (self.dram_bytes / 1e9)

@dataclass
class CommEvent:
    name: str
    bytes: float       # payload bytes
    time_ms: float     # milliseconds
    link: str          # e.g., "NVLink#0", "PCIe", "NIC"
    meta: Dict = field(default_factory=dict)

    @property
    def time_s(self) -> float:
        return self.time_ms / 1e3

    @property
    def achieved_GBps(self) -> float:
        if self.time_s <= 0:
            return 0.0
        return (self.bytes / 1e9) / self.time_s


def roofline_y(BW_GBps: float, intensity: float) -> float:
    return BW_GBps * intensity


def compute_bound_intersection(BW_GBps: float, compute_GFLOPs: float) -> float:
    if BW_GBps == 0:
        return math.inf
    return compute_GFLOPs / BW_GBps


@dataclass
class RooflineResult:
    device: DevicePeaks
    samples: List[KernelSample]
    dtype_key: str
    points: List[Tuple[float, float]]
    compute_peak_GFLOPs: float
    mem_bw_GBps: float

    def classify(self) -> List[str]:
        out = []
        for (oi, perf) in self.points:
            mem_cap = roofline_y(self.mem_bw_GBps, oi)
            comp_cap = self.compute_peak_GFLOPs
            cap = min(mem_cap, comp_cap)
            # memory-bound if mem roof < compute roof at this OI
            bound = "memory-bound" if mem_cap < comp_cap else "compute-bound"
            out.append(bound)
        return out

    def distance_to_knee(self) -> List[float]:
        """Return log10 distance of sample OI from the knee OI (positive = to the right of knee)."""
        knee = compute_bound_intersection(self.mem_bw_GBps, self.compute_peak_GFLOPs)
        out = []
        for (oi, _) in self.points:
            if knee <= 0 or oi <= 0:
                out.append(float("nan"))
            else:
                out.append(math.log10(oi) - math.log10(knee))
        return out
