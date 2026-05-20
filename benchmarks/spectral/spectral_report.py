# SPDX-License-Identifier: MIT
import argparse
import csv
import os
import tempfile
from collections import defaultdict


def _load_pyplot():
    """Return matplotlib.pyplot when available, otherwise None.

    The benchmark report should stay useful on lean CI machines.  When
    matplotlib is absent (or cannot create its font cache), we still emit
    an HTML table and record why charts were skipped.
    """

    os.environ.setdefault(
        "MPLCONFIGDIR",
        os.path.join(tempfile.gettempdir(), "tessera_mplconfig"),
    )
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt, None
    except Exception as exc:
        return None, str(exc)

def read_rows(csv_path):
    rows = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def to_num(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def group_key(row):
    return (row["op"], row["device"], row["backend"], row["dtype"])

def size_to_x(shape_str):
    s = shape_str.lower()
    if "x" in s:
        h,w = s.split("x")
        return int(h)*int(w)
    return int(s)

def main():
    ap = argparse.ArgumentParser(description="Generate charts & HTML for spectral perf results")
    ap.add_argument("--results", type=str, default="results/results.csv")
    ap.add_argument("--outdir", type=str, default="results/report")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = read_rows(args.results)
    if not rows:
        print("No rows found in CSV.")
        return
    plt, chart_warning = _load_pyplot()

    groups = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)

    charts = []
    for key, items in groups.items():
        items = sorted(items, key=lambda r: size_to_x(r["shape"]))
        xs = [size_to_x(r["shape"]) for r in items]
        gflops = [to_num(r["gflops"]) for r in items]
        gbs = [to_num(r["gbs"]) for r in items]
        time_ms = [to_num(r["time_ms"]) for r in items]

        op, device, backend, dtype = key

        if plt is not None:
            try:
                # GFLOPs chart
                plt.figure()
                plt.plot(xs, gflops, marker="o")
                plt.xlabel("Problem size (N or H*W)")
                plt.ylabel("GFLOP/s")
                plt.title(f"{op} — {device}/{backend} — {dtype} (GFLOP/s)")
                gf_path = os.path.join(args.outdir, f"{op}_{device}_{backend}_{dtype}_gflops.png".replace("/","-"))
                plt.savefig(gf_path, bbox_inches="tight", dpi=140)
                plt.close()
                charts.append(("GFLOP/s", gf_path))

                # GB/s chart
                plt.figure()
                plt.plot(xs, gbs, marker="o")
                plt.xlabel("Problem size (N or H*W)")
                plt.ylabel("GB/s")
                plt.title(f"{op} — {device}/{backend} — {dtype} (GB/s)")
                gb_path = os.path.join(args.outdir, f"{op}_{device}_{backend}_{dtype}_gbs.png".replace("/","-"))
                plt.savefig(gb_path, bbox_inches="tight", dpi=140)
                plt.close()
                charts.append(("GB/s", gb_path))

                # Time chart
                plt.figure()
                plt.plot(xs, time_ms, marker="o")
                plt.xlabel("Problem size (N or H*W)")
                plt.ylabel("Time (ms)")
                plt.title(f"{op} — {device}/{backend} — {dtype} (ms)")
                tm_path = os.path.join(args.outdir, f"{op}_{device}_{backend}_{dtype}_time_ms.png".replace("/","-"))
                plt.savefig(tm_path, bbox_inches="tight", dpi=140)
                plt.close()
                charts.append(("Time (ms)", tm_path))
            except Exception as exc:
                chart_warning = f"chart generation skipped after matplotlib error: {exc}"
                plt = None

    # Minimal HTML
    html_path = os.path.join(args.outdir, "index.html")
    with open(html_path, "w") as f:
        f.write("<html><head><meta charset='utf-8'><title>Spectral Perf Report</title></head><body>\n")
        f.write("<h1>Tessera Spectral Operators — Performance Report</h1>\n")
        f.write(f"<p>Charts generated from <code>{args.results}</code></p>\n")
        if chart_warning:
            f.write(f"<p><strong>Charts skipped:</strong> {chart_warning}</p>\n")
        # Group charts by op/device/backend/dtype
        for key, items in groups.items():
            op, device, backend, dtype = key
            f.write(f"<h2>{op} — {device}/{backend} — {dtype}</h2>\n")
            base = f"{op}_{device}_{backend}_{dtype}".replace("/","-")
            if plt is not None:
                for metric in ["gflops", "gbs", "time_ms"]:
                    img = os.path.join(args.outdir, f"{base}_{metric}.png")
                    rel = os.path.basename(img)
                    if os.path.exists(img):
                        f.write(f"<div><img src='{rel}' alt='{rel}' style='max-width:980px;'></div>\n")
            f.write("<table border='1' cellspacing='0' cellpadding='4'>\n")
            f.write("<tr><th>shape</th><th>time_ms</th><th>gflops</th><th>gbs</th><th>runtime_status</th><th>compiler_path</th></tr>\n")
            for row in sorted(items, key=lambda r: size_to_x(r["shape"])):
                f.write(
                    "<tr>"
                    f"<td>{row.get('shape', '')}</td>"
                    f"<td>{row.get('time_ms', '')}</td>"
                    f"<td>{row.get('gflops', '')}</td>"
                    f"<td>{row.get('gbs', '')}</td>"
                    f"<td>{row.get('runtime_status', '')}</td>"
                    f"<td>{row.get('compiler_path', '')}</td>"
                    "</tr>\n"
                )
            f.write("</table>\n")
        f.write("</body></html>\n")

    print(f"Report written to: {html_path}")


if __name__ == "__main__":
    main()
