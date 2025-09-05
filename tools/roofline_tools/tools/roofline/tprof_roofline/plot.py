from typing import Tuple, Optional, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .model import RooflineResult, roofline_y, compute_bound_intersection, CommEvent, DevicePeaks

def plot_roofline(res: RooflineResult, *, xlim: Tuple[float,float]=(1e-3, 1e3), fname: str="roofline.png", title: Optional[str]=None) -> str:
    fig = plt.figure(figsize=(8,6))
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Operational Intensity (FLOPs / Byte)")
    ax.set_ylabel("Performance (GFLOP/s)")

    I1, I2 = xlim
    xs = [I1, I2]
    ys_mem = [roofline_y(res.mem_bw_GBps, I1), roofline_y(res.mem_bw_GBps, I2)]
    ax.plot(xs, ys_mem, linewidth=2, label=f"Memory roof ({res.mem_bw_GBps:.0f} GB/s)")
    ax.hlines(res.compute_peak_GFLOPs, xmin=I1, xmax=I2, linewidth=2, linestyles="--", label=f"Compute roof ({res.dtype_key} {res.compute_peak_GFLOPs:.0f} GFLOP/s)")

    knee_x = compute_bound_intersection(res.mem_bw_GBps, res.compute_peak_GFLOPs)
    knee_y = roofline_y(res.mem_bw_GBps, knee_x)
    if I1 < knee_x < I2:
        ax.scatter([knee_x], [knee_y], marker="x")
        ax.annotate("knee", (knee_x, knee_y))

    for s, (oi, perf) in zip(res.samples, res.points):
        ax.scatter([oi], [perf])
        ax.annotate(s.name, (oi, perf), xytext=(5,5), textcoords="offset points")

    if title:
        ax.set_title(title)

    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=160)
    plt.close(fig)
    return fname

def plot_roofline_with_comm(res: RooflineResult, comms: List[CommEvent], device: DevicePeaks, *, xlim: Tuple[float,float]=(1e-3,1e3), fname: str="roofline_comm.png", title: Optional[str]=None) -> str:
    fig = plt.figure(figsize=(9,6))
    ax1 = plt.gca()
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Operational Intensity (FLOPs / Byte)")
    ax1.set_ylabel("Performance (GFLOP/s)")

    I1, I2 = xlim
    xs = [I1, I2]
    ys_mem = [roofline_y(res.mem_bw_GBps, I1), roofline_y(res.mem_bw_GBps, I2)]
    ax1.plot(xs, ys_mem, linewidth=2, label=f"Memory roof ({res.mem_bw_GBps:.0f} GB/s)")
    ax1.hlines(res.compute_peak_GFLOPs, xmin=I1, xmax=I2, linewidth=2, linestyles="--", label=f"Compute roof ({res.dtype_key} {res.compute_peak_GFLOPs:.0f} GFLOP/s)")

    # Knee
    knee_x = compute_bound_intersection(res.mem_bw_GBps, res.compute_peak_GFLOPs)
    knee_y = roofline_y(res.mem_bw_GBps, knee_x)
    if I1 < knee_x < I2:
        ax1.scatter([knee_x], [knee_y], marker="x")
        ax1.annotate("knee", (knee_x, knee_y))

    # Kernel points
    for s, (oi, perf) in zip(res.samples, res.points):
        ax1.scatter([oi], [perf])
        ax1.annotate(s.name, (oi, perf), xytext=(5,5), textcoords="offset points")

    # Right axis for GB/s (communication)
    ax2 = ax1.twinx()
    ax2.set_yscale("log")
    ax2.set_ylabel("Link Throughput (GB/s)")

    # Comm link bands (horizontal lines across x)
    for L in device.comm_links:
        ax2.hlines(L.bw_GBps, xmin=I1, xmax=I2, linestyles=":", linewidth=1.5, label=f"{L.name} ({L.bw_GBps:.0f} GB/s)")

    # Comm events as points on right axis, placed at fixed OI near left bound (for visibility)
    # We scatter them at I1 * 1.2^k to avoid overlap
    i = 0
    for ev in comms:
        oi_pos = I1 * (1.2 ** i)
        ax2.scatter([oi_pos], [ev.achieved_GBps])
        ax2.annotate(f"{ev.link}:{ev.name}", (oi_pos, ev.achieved_GBps), xytext=(5,5), textcoords="offset points")
        i += 1

    if title:
        ax1.set_title(title)

    # Build a combined legend
    h1,l1 = ax1.get_legend_handles_labels()
    h2,l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="best")

    fig.tight_layout()
    fig.savefig(fname, dpi=160)
    plt.close(fig)
    return fname
