#!/usr/bin/env python3
import argparse, base64, csv, html, io, json, pathlib

try:
  import pandas as pd
except ImportError:
  pd = None

try:
  import matplotlib.pyplot as plt
except ImportError:
  plt = None

HTML_TMPL = r"""
<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Tessera Operator Benchmarks Report</title>
<style>
body{{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:20px;}}
h1{{margin-bottom:0}}
small{{color:#666}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{border:1px solid #ddd;padding:6px;text-align:left}}
th{{background:#eee}}
.section{{margin-top:24px}}
</style>
</head><body>
<h1>Operator Benchmarks</h1>
<small>Generated from CSV/JSON operator benchmark output</small>
<div class="section">
<h2>Summary</h2>
<p>Rows: {rows}, Ops: {ops}</p>
<p>Telemetry: {telemetry}</p>
</div>
<div class="section">
<h2>Charts</h2>
{charts}
</div>
<div class="section">
<h2>Raw CSV (head)</h2>
{table}
</div>
</body></html>
"""

def img_tag(fig):
  buf = io.BytesIO()
  fig.savefig(buf, format="png", bbox_inches="tight")
  data = base64.b64encode(buf.getvalue()).decode("ascii")
  return f'<img src="data:image/png;base64,{data}">'

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("csv")
  ap.add_argument("html_out")
  ap.add_argument("--json", default=None, help="Optional results.json emitted by scripts/opbench.py")
  args = ap.parse_args()

  rows = _read_rows(args.csv)
  telemetry = "not available"
  json_path = pathlib.Path(args.json) if args.json else pathlib.Path(args.csv).with_name("results.json")
  if json_path.exists():
    payload = json.loads(json_path.read_text())
    summary = payload.get("telemetry_summary", {})
    telemetry = f"{summary.get('schema', '')}, events={summary.get('event_count', 0)}, bottlenecks={summary.get('bottlenecks', {})}"
  charts=""
  # Example: for matmul, plot gflops vs M (averaged over others)
  ops = sorted({row.get("op", "") for row in rows if row.get("op")})
  if pd is not None and plt is not None:
    df = pd.read_csv(args.csv)
    if "matmul" in df["op"].unique():
      d = df[df["op"]=="matmul"]
      if "M" in d.columns:
        xs = sorted(d["M"].unique())
        ys = [d[d["M"]==x]["gflops"].astype(float).mean() for x in xs]
        fig = plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("M"); plt.ylabel("gflops"); plt.title("Matmul: GFLOP/s vs M")
        charts += img_tag(fig)
    table = df.head().to_html(index=False)
  else:
    charts = "<p>Install pandas and matplotlib for charts.</p>"
    table = _rows_to_table(rows[:10])
  html = HTML_TMPL.format(rows=len(rows), ops=", ".join(ops), telemetry=telemetry, charts=charts, table=table)
  with open(args.html_out,"w") as f: f.write(html)
  print("Wrote", args.html_out)

def _read_rows(path):
  with open(path, newline="") as f:
    return list(csv.DictReader(f))


def _rows_to_table(rows):
  if not rows:
    return "<p>No rows.</p>"
  keys = list(rows[0].keys())
  out = ["<table><thead><tr>"]
  out.extend(f"<th>{html.escape(key)}</th>" for key in keys)
  out.append("</tr></thead><tbody>")
  for row in rows:
    out.append("<tr>")
    out.extend(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in keys)
    out.append("</tr>")
  out.append("</tbody></table>")
  return "".join(out)


if __name__ == "__main__":
  main()
