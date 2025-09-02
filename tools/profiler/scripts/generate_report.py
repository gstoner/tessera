
import argparse, csv, os, html

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--csv", required=True)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  rows = []
  with open(args.csv, newline="") as f:
    for row in csv.DictReader(f):
      rows.append(row)

  html_rows = "\n".join(
    f"<tr><td>{html.escape(r['arch'])}</td><td>{html.escape(r['op'])}</td>"
    f"<td>{r['M']}</td><td>{r['N']}</td><td>{r['K']}</td>"
    f"<td>{r['iters']}</td><td>{float(r['ms_avg']):.3f}</td><td>{float(r['tflops']):.3f}</td></tr>"
    for r in rows
  )

  page = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Tessera Microbench Report</title>
<style>
body{{font-family:system-ui,Arial,sans-serif;margin:24px}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccc;padding:8px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
</style></head><body>
<h1>Tessera Microbench Report</h1>
<p>Source CSV: {{csv}}</p>
<table>
<tr><th>arch</th><th>op</th><th>M</th><th>N</th><th>K</th><th>iters</th><th>ms_avg</th><th>TFLOP/s</th></tr>
{html_rows}
</table>
</body></html>'''
  page = page.replace("{{csv}}", html.escape(args.csv))

  with open(os.path.join(args.out, "index.html"), "w") as f:
    f.write(page)

if __name__ == "__main__":
  main()
