#!/usr/bin/env python3
import json, argparse, math, collections, time

def load_trace(path):
    with open(path, 'r') as f:
        return json.load(f)

def aggregate(trace):
    evs = trace.get("traceEvents", [])
    stacks = collections.defaultdict(list)  # tid -> stack of (name, ts_us)
    totals = collections.defaultdict(lambda: {"dur_us":0.0, "bytes":0.0, "flops":0.0})
    for e in evs:
        ph = e.get("ph")
        tid = e.get("tid", 0)
        ts = e.get("ts", 0.0)  # microseconds
        if ph == "B":
            stacks[tid].append((e.get("name","range"), ts))
        elif ph == "E":
            if stacks[tid]:
                name, t0 = stacks[tid].pop()
                totals[name]["dur_us"] += max(0.0, ts - t0)
        elif ph == "C":
            # attribute to active op on this tid if available
            target = "counter"
            if stacks[tid]:
                target = stacks[tid][-1][0]
            args = e.get("args", {})
            v = float(args.get("value", 0.0))
            ename = (e.get("name","") or "").lower()
            if "byte" in ename:
                totals[target]["bytes"] += v
            elif "flop" in ename:
                totals[target]["flops"] += v
    return totals

def make_html(totals, out_path, peak_flops=None, hbm_gbs=None):
    rows = []
    total_time = sum(v["dur_us"] for v in totals.values()) or 1.0
    for name, v in totals.items():
        b = v["bytes"]; f = v["flops"]; dur = v["dur_us"]; pct = 100.0 * dur / total_time
        intensity = (f / b) if (b > 0.0 and f > 0.0) else 0.0
        rows.append({"name": name, "ms": dur/1000.0, "pct": pct, "bytes": b, "flops": f, "intensity": intensity})
    rows.sort(key=lambda r: r["ms"], reverse=True)
    import json as _json
    data_json = _json.dumps({"rows": rows, "peak_flops": peak_flops, "hbm_gbs": hbm_gbs})
    html = f'''<!doctype html>
<html><head><meta charset="utf-8"/>
<title>Tessera Profiler Report (stub)</title>
<style>
body {{ font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; margin: 24px; }}
h1 {{ margin-top: 0; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
tbody tr:nth-child(even) {{ background: #fafafa; }}
small {{ color: #666; }}
#chart {{ width: 100%; height: 360px; border: 1px solid #ddd; margin-top: 16px; position: relative; }}
.axis {{ position: absolute; color: #666; font-size: 12px; }}
</style></head>
<body>
<h1>Tessera Profiler — Report (Roofline/Hot Ops stub)</h1>
<p><small>Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}</small></p>

<h2>Hot Ops (by time)</h2>
<table>
<thead><tr><th>Op</th><th>Time (ms)</th><th>Time %</th><th>Bytes</th><th>FLOPs</th><th>Intensity (F/B)</th></tr></thead>
<tbody></tbody>
</table>

<h2>Roofline (stub)</h2>
<div id="chart"></div>
<p><small>Inputs: peak_flops={peak_flops or "unset"} FLOP/s, hbm_gbs={hbm_gbs or "unset"} GB/s.</small></p>

<script>
const DATA = {data_json};

const tbody = document.querySelector("tbody");
for (const r of DATA.rows.slice(0, 50)) {{
  const tr = document.createElement("tr");
  function td(txt) {{ const e = document.createElement("td"); e.textContent = txt; return e; }}
  const name = document.createElement("td"); name.textContent = r.name; name.style.textAlign="left"; tr.appendChild(name);
  tr.appendChild(td(r.ms.toFixed(3)));
  tr.appendChild(td(r.pct.toFixed(1)));
  tr.appendChild(td(Math.round(r.bytes)));
  tr.appendChild(td(Math.round(r.flops)));
  tr.appendChild(td(r.intensity.toFixed(3)));
  tbody.appendChild(tr);
}}

// Scatter: x=Bytes (log), y=FLOPs (log).
const div = document.getElementById("chart");
const W = div.clientWidth, H = div.clientHeight;
const canvas = document.createElement("canvas");
canvas.width = W; canvas.height = H; div.appendChild(canvas);
const ctx = canvas.getContext("2d");
function log10(x) {{ return Math.log(x) / Math.LN10; }}
const pts = DATA.rows.filter(r => r.bytes > 0 && r.flops > 0);
const minB = Math.min(...pts.map(p => p.bytes), 1), maxB = Math.max(...pts.map(p => p.bytes), 10);
const minF = Math.min(...pts.map(p => p.flops), 1), maxF = Math.max(...pts.map(p => p.flops), 10);
const pad = 30;
function xmap(b) {{ const t=(log10(b)-log10(minB))/(log10(maxB)-log10(minB)||1); return pad + t*(W-2*pad); }}
function ymap(f) {{ const t=(log10(f)-log10(minF))/(log10(maxF)-log10(minF)||1); return H-pad - t*(H-2*pad); }}

ctx.fillStyle = "#000";
for (const p of pts) {{
  const x = xmap(p.bytes), y = ymap(p.flops);
  ctx.beginPath(); ctx.arc(x, y, 3, 0, 2*Math.PI); ctx.fill();
}}

// Rooflines
if (DATA.peak_flops) {{
  const y = ymap(DATA.peak_flops);
  ctx.strokeStyle="#a00"; ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W-pad, y); ctx.stroke();
}}
if (DATA.hbm_gbs) {{
  const f1 = DATA.hbm_gbs * minB, f2 = DATA.hbm_gbs * maxB;
  ctx.strokeStyle="#0a0"; ctx.beginPath(); ctx.moveTo(xmap(minB), ymap(f1)); ctx.lineTo(xmap(maxB), ymap(f2)); ctx.stroke();
}}
</script>
</body></html>'''
    with open(out_path, "w") as f:
        f.write(html)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="traceEvents JSON (Chrome/Perfetto)")
    ap.add_argument("--out", dest="out", required=True, help="HTML report path")
    ap.add_argument("--peak-flops", type=float, default=None, help="Device peak FLOP/s (e.g., 2.0e14)")
    ap.add_argument("--hbm-gbs", type=float, default=None, help="HBM bandwidth GB/s (e.g., 3000)")
    args = ap.parse_args()
    trace = load_trace(args.inp)
    totals = aggregate(trace)
    make_html(totals, args.out, peak_flops=args.peak_flops, hbm_gbs=args.hbm_gbs)

if __name__ == "__main__":
    main()
