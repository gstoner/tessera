
#!/usr/bin/env python3
import argparse, json, html, datetime, os, math
import yaml

def make_roofline_svg(results, peaks):
    # Simple log-log axes roofline with compute peak and memory roof.
    W, H = 800, 500
    pad = 60
    # Gather points with AI (I) and perf (P)
    pts = []
    for r in results.get("rows", []):
        flops = r.get("throughput_flops")
        bps = r.get("bytes_per_s")
        if not flops or not bps: 
            # Try derive I if present
            I = r.get("arithmetic_intensity")
            if I and flops: pts.append((I, flops, r.get("bench_id")))
            continue
        I = flops / max(bps, 1e-12)
        pts.append((I, flops, r.get("bench_id")))

    if not pts: 
        return "<p>No points for roofline.</p>"

    # Axes ranges
    xs = [p[0] for p in pts] + [peaks["compute_peak_flops"]/peaks["memory_peak_bytes_per_s"]]
    ys = [p[1] for p in pts] + [peaks["compute_peak_flops"], peaks["memory_peak_bytes_per_s"]*max(xs)]
    xmin, xmax = max(min(xs)/10, 1e-3), max(xs)*10
    ymin, ymax = max(min(ys)/10, 1e6), max(ys)*10
    def X(v): return pad + (math.log10(v) - math.log10(xmin)) / (math.log10(xmax)-math.log10(xmin)) * (W-2*pad)
    def Y(v): return H-pad - (math.log10(v) - math.log10(ymin)) / (math.log10(ymax)-math.log10(ymin)) * (H-2*pad)

    ridge = peaks["compute_peak_flops"] / peaks["memory_peak_bytes_per_s"]
    # Build SVG
    lines = [f"<svg width='{W}' height='{H}' xmlns='http://www.w3.org/2000/svg'>"]
    # axes
    lines.append(f"<line x1='{pad}' y1='{H-pad}' x2='{W-pad}' y2='{H-pad}' stroke='black'/>")
    lines.append(f"<line x1='{pad}' y1='{H-pad}' x2='{pad}' y2='{pad}' stroke='black'/>")
    # memory roof: y = B * x up to ridge
    x1, y1 = xmin, peaks['memory_peak_bytes_per_s']*xmin
    x2, y2 = ridge, peaks['memory_peak_bytes_per_s']*ridge
    lines.append(f"""<polyline fill='none' stroke='gray' stroke-width='2'
        points='{X(x1)},{Y(y1)} {X(x2)},{Y(y2)}'/>""")
    # compute roof: y = P_peak for x >= ridge
    lines.append(f"""<polyline fill='none' stroke='gray' stroke-width='2'
        points='{X(ridge)},{Y(peaks['compute_peak_flops'])} {X(xmax)},{Y(peaks['compute_peak_flops'])}'/>""")
    # ridge marker
    lines.append(f"""<line x1='{X(ridge)}' y1='{Y(ymin)}' x2='{X(ridge)}' y2='{Y(ymax)}' stroke='#ddd' stroke-dasharray='4,4'/>""")
    # labels
    lines.append(f"""<text x='{W/2}' y='{H-20}' text-anchor='middle' font-size='12'>Arithmetic Intensity (FLOPs / Byte)</text>""")
    lines.append(f"""<text x='{20}' y='{20}' text-anchor='start' font-size='12' transform='rotate(-90 20,20)'>Performance (FLOP/s)</text>""")
    # points
    for (I, P, name) in pts:
        lines.append(f"""<circle cx='{X(I)}' cy='{Y(P)}' r='4' fill='black'/>
            <title>{html.escape(name or 'bench')}\nI={I:.3g}, P={P:.3g}</title>""")
    # legend
    lines.append(f"""<text x='{pad+10}' y='{pad+15}' font-size='12'>Peak Compute: {peaks['compute_peak_flops']:.3g} FLOP/s</text>""")
    lines.append(f"""<text x='{pad+10}' y='{pad+30}' font-size='12'>Peak Memory BW: {peaks['memory_peak_bytes_per_s']:.3g} B/s</text>""")
    lines.append("</svg>")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--html", required=True)
    ap.add_argument("--peaks", default=None, help="YAML file with compute_peak_flops and memory_peak_bytes_per_s")
    args = ap.parse_args()

    with open(args.results, "r") as f:
        R = json.load(f)
    peaks = None
    if args.peaks and os.path.exists(args.peaks):
        with open(args.peaks,"r") as pf:
            peaks = yaml.safe_load(pf)

    ts = datetime.datetime.fromtimestamp(R.get("timestamp", 0))
    rows = R.get("rows", [])

    def td(v):
        return html.escape(str(v))

    body = []
    body.append(f"<h1>Tessera Benchmark Report</h1>")
    body.append(f"<p>Suite: {td(R.get('suite',{}).get('name',''))} — {td(ts)}</p>")

    # Roofline (optional)
    if peaks:
        body.append("<h2>Roofline</h2>")
        body.append(make_roofline_svg(R, peaks))

    # Table of results
    body.append("<h2>Results</h2>")
    body.append("<table border='1' cellpadding='6' cellspacing='0'>")
    body.append("<tr style='background:#eee;font-weight:bold'><td>bench_id</td><td>ok</td><td>skip_reason</td><td>metrics</td></tr>")
    for r in rows:
        metrics = {k:v for k,v in r.items() if k not in ['bench_id','variant','ok','skip_reason'] and not isinstance(v, dict)}
        body.append(f"<tr><td>{td(r.get('bench_id'))}</td><td>{td(r.get('ok'))}</td><td>{td(r.get('skip_reason'))}</td><td>{td(metrics)}</td></tr>")
    body.append("</table>")

    # Trace link
    body.append("<h2>Trace</h2>")
    body.append("<p>If generated, open <code>trace.json</code> in Perfetto UI.</p>")

    os.makedirs(os.path.dirname(args.html), exist_ok=True)
    with open(args.html, "w") as f:
        f.write("<html><head><meta charset='utf-8'><title>Tessera Bench</title></head><body>")
        f.write("\n".join(body))
        f.write("</body></html>")
    print(f"Wrote {args.html}")

if __name__ == "__main__":
    main()
