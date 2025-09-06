#!/usr/bin/env python3
import csv, sys, os, math, json

def load_csv(path):
    rows = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def render_html(rows, out):
    # Minimal inline HTML (no external deps) with a simple table and a canvas placeholder.
    with open(out, "w") as f:
        f.write("<!doctype html><meta charset='utf-8'><title>EBT Roofline</title>\n")
        f.write("<h1>EBT Roofline Report</h1>\n")
        f.write("<p>Rows: %d</p>\n" % len(rows))
        f.write("<table border='1' cellpadding='4' cellspacing='0'>\n")
        if rows:
            f.write("<tr>" + "".join(f"<th>{k}</th>" for k in rows[0].keys()) + "</tr>\n")
        for r in rows:
            f.write("<tr>" + "".join(f"<td>{r[k]}</td>" for k in rows[0].keys()) + "</tr>\n")
        f.write("</table>\n")
        f.write("<p>(Hook your charting here; this is a minimal placeholder.)</p>\n")

def main():
    if len(sys.argv) < 3:
        print("usage: roofline.py reports/roofline.csv reports/roofline.html")
        sys.exit(2)
    rows = load_csv(sys.argv[1])
    render_html(rows, sys.argv[2])

if __name__ == '__main__':
    main()
