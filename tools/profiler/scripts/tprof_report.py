#!/usr/bin/env python3
import argparse
import collections
import json
import os
import time


def load_trace(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate(trace):
    events = trace.get("traceEvents", [])
    stacks = collections.defaultdict(list)
    totals = collections.defaultdict(lambda: {"dur_us": 0.0, "bytes": 0.0, "flops": 0.0})
    for event in events:
        phase = event.get("ph")
        tid = event.get("tid", 0)
        ts = event.get("ts", 0.0)
        if phase == "B":
            stacks[tid].append((event.get("name", "range"), ts))
        elif phase == "E":
            if stacks[tid]:
                name, start_ts = stacks[tid].pop()
                totals[name]["dur_us"] += max(0.0, ts - start_ts)
        elif phase == "C":
            target = stacks[tid][-1][0] if stacks[tid] else "counter"
            args = event.get("args", {})
            value = float(args.get("value", 0.0))
            event_name = (event.get("name", "") or "").lower()
            if "byte" in event_name:
                totals[target]["bytes"] += value
            elif "flop" in event_name:
                totals[target]["flops"] += value
    return totals


def _coerce_peak_value(value):
    try:
        return float(value.replace("_", ""))
    except ValueError:
        return value.strip("\"'")


def _fallback_parse_yaml(peaks_path):
    out = {}
    current_key = None
    with open(peaks_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped == "devices:":
                current_key = None
                continue
            if stripped.endswith(":") and not stripped.startswith("-"):
                current_key = stripped[:-1].strip()
                out[current_key] = {}
            elif ":" in stripped and stripped.split(":", 1)[1].lstrip().startswith("{"):
                key, value = stripped.split(":", 1)
                fields = {}
                for item in value.strip().strip("{}").split(","):
                    if ":" not in item:
                        continue
                    field, raw = item.split(":", 1)
                    fields[field.strip()] = _coerce_peak_value(raw.strip())
                out[key.strip()] = fields
            elif ":" in stripped and current_key:
                key, value = stripped.split(":", 1)
                out[current_key][key.strip()] = _coerce_peak_value(value.strip())
    return out


def load_peaks(peaks_path):
    if not peaks_path:
        return {}
    try:
        import yaml  # type: ignore

        with open(peaks_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if "devices" in data and isinstance(data["devices"], dict):
            return data["devices"]
        return data
    except Exception:
        return _fallback_parse_yaml(peaks_path)


def select_peaks(peaks_map, arch, env_arch):
    for key in (arch, env_arch):
        if key and key in peaks_map:
            peak = peaks_map[key]
            return peak.get("peak_flops"), peak.get("hbm_gbs")
    for peak in peaks_map.values():
        return peak.get("peak_flops"), peak.get("hbm_gbs")
    return None, None


def make_html(totals, out_path, peak_flops=None, hbm_gbs=None):
    rows = []
    total_time = sum(value["dur_us"] for value in totals.values()) or 1.0
    for name, value in totals.items():
        bytes_moved = value["bytes"]
        flops = value["flops"]
        duration = value["dur_us"]
        intensity = flops / bytes_moved if bytes_moved > 0.0 and flops > 0.0 else 0.0
        rows.append(
            {
                "name": name,
                "ms": duration / 1000.0,
                "pct": 100.0 * duration / total_time,
                "bytes": bytes_moved,
                "flops": flops,
                "intensity": intensity,
            }
        )
    rows.sort(key=lambda row: row["ms"], reverse=True)
    data_json = json.dumps({"rows": rows, "peak_flops": peak_flops, "hbm_gbs": hbm_gbs})
    html = f'''<!doctype html>
<html><head><meta charset="utf-8"/>
<title>Tessera Profiler Report</title>
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
<h1>Tessera Profiler - Report</h1>
<p><small>Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}</small></p>

<h2>Hot Ops (by time)</h2>
<table>
<thead><tr><th>Op</th><th>Time (ms)</th><th>Time %</th><th>Bytes</th><th>FLOPs</th><th>Intensity (F/B)</th></tr></thead>
<tbody></tbody>
</table>

<h2>Roofline</h2>
<div id="chart"></div>
<p><small>Peaks: peak_flops={peak_flops or "unset"} FLOP/s, hbm_gbs={hbm_gbs or "unset"} GB/s.</small></p>

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
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True, help="traceEvents JSON")
    parser.add_argument("--out", dest="out", required=True, help="HTML report path")
    parser.add_argument("--peak-flops", type=float, default=None, help="Device peak FLOP/s")
    parser.add_argument("--hbm-gbs", type=float, default=None, help="HBM bandwidth GB/s")
    parser.add_argument("--peaks", type=str, default=None, help="YAML with device peaks")
    parser.add_argument("--arch", type=str, default=None, help="Architecture key from YAML")
    args = parser.parse_args()

    peak_flops = args.peak_flops
    hbm_gbs = args.hbm_gbs
    if args.peaks and (peak_flops is None or hbm_gbs is None):
        selected_flops, selected_hbm = select_peaks(
            load_peaks(args.peaks), args.arch, os.environ.get("TPROF_ARCH")
        )
        peak_flops = peak_flops if peak_flops is not None else selected_flops
        hbm_gbs = hbm_gbs if hbm_gbs is not None else selected_hbm

    make_html(aggregate(load_trace(args.inp)), args.out, peak_flops=peak_flops, hbm_gbs=hbm_gbs)


if __name__ == "__main__":
    main()
