#!/usr/bin/env python3
import argparse
import collections
import json
import os
import time


def load_trace(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_context(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_statuses(paths):
    out = []
    for path in paths or []:
        with open(path, "r", encoding="utf-8") as f:
            out.append(json.load(f))
    return out


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
        elif phase == "X":
            name = event.get("name", "range")
            totals[name]["dur_us"] += max(0.0, float(event.get("dur", 0.0)))
            args = event.get("args", {})
            if "bytes" in args and args["bytes"] is not None:
                totals[name]["bytes"] += float(args["bytes"])
            if "flops" in args and args["flops"] is not None:
                totals[name]["flops"] += float(args["flops"])
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


def aggregate_categories(trace):
    categories = collections.Counter()
    providers = collections.defaultdict(collections.Counter)
    provider_dropped = collections.Counter()
    correlations = {"correlation_id": set(), "launch_id": set(), "probe_name": set()}
    for event in trace.get("traceEvents", []):
        cat = str(event.get("cat", "unknown"))
        categories[cat] += 1
        args = event.get("args", {})
        provider = "unknown"
        if isinstance(args, dict):
            provider = str(args.get("provider") or "unknown")
            if cat != "metadata":
                provider_dropped[provider] += int(args.get("dropped_records") or 0)
            for key in correlations:
                if args.get(key) is not None:
                    correlations[key].add(str(args[key]))
        providers[provider][cat] += 1
    return {
        "categories": dict(sorted(categories.items())),
        "provider_categories": {
            provider: dict(sorted(counts.items()))
            for provider, counts in sorted(providers.items())
        },
        "provider_dropped_records": dict(sorted(provider_dropped.items())),
        "correlation_summary": {
            key: len(values) for key, values in sorted(correlations.items())
        },
    }


def aggregate_provider_status(trace, sidecars=None):
    statuses = []
    for status in sidecars or []:
        if isinstance(status, dict):
            statuses.append(status)
    for event in trace.get("traceEvents", []):
        if event.get("cat") != "provider_status":
            continue
        args = event.get("args", {})
        if isinstance(args, dict):
            statuses.append({
                "provider": args.get("provider"),
                "target": args.get("target"),
                "status": args.get("status"),
                "diagnostics": args.get("diagnostics", {}),
            })
    deduped = {}
    for status in statuses:
        provider = status.get("provider") or "unknown"
        deduped[provider] = status
    return [deduped[key] for key in sorted(deduped)]


def aggregate_context(context):
    if not context:
        return None
    samples = context.get("samples", [])
    summary = context.get("bottleneck_summary") or {}
    bottlenecks = summary.get("bottlenecks") or {}
    if not bottlenecks and samples:
        counts = collections.Counter(str(s.get("bottleneck", "unknown")) for s in samples)
        bottlenecks = dict(sorted(counts.items()))
    dominant = summary.get("dominant_bottleneck")
    if dominant is None and bottlenecks:
        dominant = max(bottlenecks.items(), key=lambda item: item[1])[0]
    return {
        "schema": context.get("schema"),
        "target": context.get("target"),
        "provider": context.get("provider"),
        "source_status": context.get("source_status"),
        "sample_count": context.get("sample_count", len(samples)),
        "dominant_bottleneck": dominant,
        "bottlenecks": bottlenecks,
    }


def build_report_summary(trace, totals, peak_flops=None, hbm_gbs=None, context=None, provider_statuses=None):
    rows = _rows_from_totals(totals)
    category_summary = aggregate_categories(trace)
    return {
        "schema": "tessera.profiler_report_summary.v1",
        "hot_ops": rows,
        "roofline": {
            "peak_flops": peak_flops,
            "hbm_gbs": hbm_gbs,
            "points": [
                {
                    "name": row["name"],
                    "bytes": row["bytes"],
                    "flops": row["flops"],
                    "arithmetic_intensity_flops_per_byte": row["intensity"],
                }
                for row in rows
                if row["bytes"] > 0.0 and row["flops"] > 0.0
            ],
        },
        "context": context,
        "provider_statuses": provider_statuses or [],
        **category_summary,
    }


def _rows_from_totals(totals):
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
    return rows


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


def make_html(totals, out_path, peak_flops=None, hbm_gbs=None, context=None, provider_statuses=None):
    rows = _rows_from_totals(totals)
    data_json = json.dumps({
        "rows": rows,
        "peak_flops": peak_flops,
        "hbm_gbs": hbm_gbs,
        "context": context,
        "provider_statuses": provider_statuses or [],
    })
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
.muted {{ color: #666; }}
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

<section id="system-context" hidden>
<h2>System Context</h2>
<p class="muted" id="context-meta"></p>
<table>
<thead><tr><th>Bottleneck</th><th>Samples</th></tr></thead>
<tbody id="context-body"></tbody>
</table>
</section>

<section id="provider-status" hidden>
<h2>Provider Status</h2>
<table>
<thead><tr><th>Provider</th><th>Target</th><th>Status</th><th>Diagnostics</th></tr></thead>
<tbody id="provider-status-body"></tbody>
</table>
</section>

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

if (DATA.context) {{
  document.getElementById("system-context").hidden = false;
  const c = DATA.context;
  document.getElementById("context-meta").textContent =
    `target=${{c.target || "unknown"}}, provider=${{c.provider || "unknown"}}, status=${{c.source_status || "unknown"}}, dominant=${{c.dominant_bottleneck || "unknown"}}`;
  const cbody = document.getElementById("context-body");
  for (const [name, count] of Object.entries(c.bottlenecks || {{}})) {{
    const tr = document.createElement("tr");
    const label = document.createElement("td"); label.textContent = name; label.style.textAlign = "left"; tr.appendChild(label);
    const value = document.createElement("td"); value.textContent = String(count); tr.appendChild(value);
    cbody.appendChild(tr);
  }}
}}

if (DATA.provider_statuses && DATA.provider_statuses.length) {{
  document.getElementById("provider-status").hidden = false;
  const body = document.getElementById("provider-status-body");
  for (const status of DATA.provider_statuses) {{
    const tr = document.createElement("tr");
    function td(txt, alignLeft=false) {{
      const e = document.createElement("td");
      e.textContent = txt;
      if (alignLeft) e.style.textAlign = "left";
      return e;
    }}
    tr.appendChild(td(status.provider || "unknown", true));
    tr.appendChild(td(status.target || "unknown", true));
    tr.appendChild(td(status.status || "unknown", true));
    tr.appendChild(td(JSON.stringify(status.diagnostics || {{}}), true));
    body.appendChild(tr);
  }}
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
    parser.add_argument("--context-json", type=str, default=None, help="tessera.profiler_context.v1 JSON")
    parser.add_argument(
        "--provider-status-json",
        action="append",
        default=[],
        help="tessera.profiler_provider_status.v1 JSON. Can be repeated.",
    )
    parser.add_argument("--summary-json", type=str, default=None, help="Write machine-readable report summary JSON.")
    args = parser.parse_args()

    peak_flops = args.peak_flops
    hbm_gbs = args.hbm_gbs
    if args.peaks and (peak_flops is None or hbm_gbs is None):
        selected_flops, selected_hbm = select_peaks(
            load_peaks(args.peaks), args.arch, os.environ.get("TPROF_ARCH")
        )
        peak_flops = peak_flops if peak_flops is not None else selected_flops
        hbm_gbs = hbm_gbs if hbm_gbs is not None else selected_hbm

    trace = load_trace(args.inp)
    totals = aggregate(trace)
    context = aggregate_context(load_context(args.context_json))
    provider_statuses = aggregate_provider_status(trace, load_statuses(args.provider_status_json))
    make_html(
        totals,
        args.out,
        peak_flops=peak_flops,
        hbm_gbs=hbm_gbs,
        context=context,
        provider_statuses=provider_statuses,
    )
    if args.summary_json:
        payload = build_report_summary(
            trace,
            totals,
            peak_flops=peak_flops,
            hbm_gbs=hbm_gbs,
            context=context,
            provider_statuses=provider_statuses,
        )
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
