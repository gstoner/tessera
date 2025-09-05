#!/usr/bin/env python3
import argparse, pandas as pd, matplotlib.pyplot as plt, base64, io

HTML_TMPL = r"""
<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Tessera Operator Benchmarks Report</title>
<style>
body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:20px;}
h1{margin-bottom:0}
small{color:#666}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{border:1px solid #ddd;padding:6px;text-align:left}
th{background:#eee}
.section{margin-top:24px}
</style>
</head><body>
<h1>Operator Benchmarks</h1>
<small>Generated from CSV</small>
<div class="section">
<h2>Summary</h2>
<p>Rows: {rows}, Ops: {ops}</p>
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
  args = ap.parse_args()

  df = pd.read_csv(args.csv)
  charts=""
  # Example: for matmul, plot gflops vs M (averaged over others)
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
  html = HTML_TMPL.format(rows=len(df), ops=", ".join(sorted(df["op"].unique())), charts=charts, table=table)
  with open(args.html_out,"w") as f: f.write(html)
  print("Wrote", args.html_out)

if __name__ == "__main__":
  main()
