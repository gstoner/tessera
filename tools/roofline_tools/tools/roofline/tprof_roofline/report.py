from typing import Optional, List
import base64, csv, json
from .model import RooflineResult, CommEvent
from .plot import plot_roofline_with_comm, plot_roofline

HTML_TMPL = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<title>Roofline Report</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 20px; }
table { border-collapse: collapse; width: 100%; margin-top: 1rem;}
th, td { border: 1px solid #ccc; padding: 6px 8px; text-align: left; }
th { background: #666; color: #fff; }
.figure { text-align: center; margin: 1rem auto; }
.caption { color: #555; font-size: 0.9rem; }
.tab { margin-top: 1rem; }
</style>
</head><body>
<h1>Roofline Report</h1>
<p><b>Device:</b> {device} &nbsp; <b>HBM BW:</b> {bw:.0f} GB/s &nbsp; <b>Compute roof:</b> {dtype} {comp:.0f} GFLOP/s</p>
<div class="figure">
  <img src="data:image/png;base64,{png_b64}" alt="roofline"/>
  <div class="caption">Roofline chart (with link throughput bands on right axis)</div>
</div>
<h2>Kernel Summary</h2>
<table>
<thead><tr><th>Name</th><th>Operational Intensity (FLOPs/Byte)</th><th>Achieved GFLOP/s</th><th>Achieved GB/s</th><th>Time (ms)</th><th>Class</th><th>Δlog10(OI-knee)</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
<h2>Communication Summary</h2>
<table>
<thead><tr><th>Link</th><th>Name</th><th>Bytes</th><th>Time (ms)</th><th>Achieved GB/s</th></tr></thead>
<tbody>
{comm_rows}
</tbody>
</table>
</body></html>
"""

def _b64_png(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

def export_classification(res: RooflineResult, out_csv: Optional[str], out_json: Optional[str]) -> None:
    classes = res.classify()
    dist = res.distance_to_knee()
    rows = []
    for s, (oi, perf), c, d in zip(res.samples, res.points, classes, dist):
        rows.append({
            "name": s.name,
            "operational_intensity": oi,
            "achieved_GFLOPs": perf,
            "achieved_GBps": s.achieved_GBps,
            "time_ms": s.time_ms,
            "class": c,
            "delta_log10_oi_to_knee": d,
        })
    if out_csv:
        import csv
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if out_json:
        import json
        with open(out_json, "w") as f:
            json.dump(rows, f, indent=2)

def generate_report(res: RooflineResult, comms: Optional[List[CommEvent]] = None, out_html: str="roofline_report.html", fig_path: str="roofline_comm.png") -> str:
    if comms is None:
        fig = plot_roofline(res, fname=fig_path, title=f"{res.device.name} roofline ({res.dtype_key})")
    else:
        fig = plot_roofline_with_comm(res, comms, res.device, fname=fig_path, title=f"{res.device.name} roofline+comm ({res.dtype_key})")
    b64 = _b64_png(fig)

    # Rows
    classes = res.classify()
    dist = res.distance_to_knee()
    rows = []
    for s, (oi, perf), c, d in zip(res.samples, res.points, classes, dist):
        rows.append(f"<tr><td>{s.name}</td><td>{oi:.3f}</td><td>{perf:.1f}</td><td>{s.achieved_GBps:.1f}</td><td>{s.time_ms:.3f}</td><td>{c}</td><td>{d:.3f}</td></tr>")

    # Comm rows
    comm_rows = []
    for ev in (comms or []):
        comm_rows.append(f"<tr><td>{ev.link}</td><td>{ev.name}</td><td>{ev.bytes:.3e}</td><td>{ev.time_ms:.3f}</td><td>{ev.achieved_GBps:.1f}</td></tr>")

    html = HTML_TMPL.format(
        device=res.device.name, bw=res.mem_bw_GBps, dtype=res.dtype_key, comp=res.compute_peak_GFLOPs,
        png_b64=b64, rows="\n".join(rows), comm_rows="\n".join(comm_rows)
    )
    with open(out_html, "w") as f:
        f.write(html)
    return out_html
