#!/usr/bin/env python3
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from tprof_roofline.model import DevicePeaks, analyze
from tprof_roofline.ingest import read_kernels_csv, read_perfetto_trace, read_nsight_compute_csv
from tprof_roofline.report import generate_report, export_classification
from tprof_roofline.report_multi import generate_multi
from tprof_roofline.plot import plot_roofline_with_comm

def main():
    ap = argparse.ArgumentParser(description="tprof-roofline-v2: Roofline analysis (compute + comm)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Single report
    one = sub.add_parser("one", help="Generate one report from CSV/Perfetto/Nsight Compute")
    one.add_argument("--peaks", required=True, help="Device peaks YAML")
    one.add_argument("--input", required=True, help="Input path")
    one.add_argument("--fmt", choices=["csv","perfetto","nsight"], default="csv")
    one.add_argument("--dtype", default="fp32")
    one.add_argument("--outdir", default="roofline_out")
    one.add_argument("--export-csv", default=None)
    one.add_argument("--export-json", default=None)

    # Multi-device
    multi = sub.add_parser("multi", help="Generate a multi-device tabbed HTML")
    multi.add_argument("--pairs", required=True, help="JSON list of {peaks:..., input:..., fmt:..., dtype:...}")
    multi.add_argument("--outdir", default="roofline_multi_out")

    args = ap.parse_args()

    if args.cmd == "one":
        device = DevicePeaks.from_yaml(args.peaks)
        if args.fmt == "csv":
            kernels = read_kernels_csv(args.input)
            comms = []
        elif args.fmt == "nsight":
            kernels = read_nsight_compute_csv(args.input)
            comms = []
        else:
            kernels, comms = read_perfetto_trace(args.input)

        res = analyze(kernels, device=device, dtype_key=args.dtype)
        os.makedirs(args.outdir, exist_ok=True)
        fig_path = os.path.join(args.outdir, "roofline_comm.png")
        out_html = os.path.join(args.outdir, "roofline_report.html")
        generate_report(res, comms, out_html=out_html, fig_path=fig_path)

        if args.export_csv or args.export_json:
            export_classification(res, args.export_csv, args.export_json)

        print(f"Wrote {out_html} and {fig_path}")
    else:
        pairs = json.loads(args.pairs)
        os.makedirs(args.outdir, exist_ok=True)
        figs = []
        for i, p in enumerate(pairs):
            device = DevicePeaks.from_yaml(p["peaks"])
            fmt = p.get("fmt","csv")
            dtype = p.get("dtype","fp32")
            if fmt == "csv":
                kernels = read_kernels_csv(p["input"]); comms = []
            elif fmt == "nsight":
                kernels = read_nsight_compute_csv(p["input"]); comms = []
            else:
                kernels, comms = read_perfetto_trace(p["input"])
            res = analyze(kernels, device, dtype_key=dtype)
            fig = os.path.join(args.outdir, f"roofline_{i}.png")
            plot_roofline_with_comm(res, comms, device, fname=fig, title=f"{device.name} ({dtype})")
            figs.append((device.name, fig))
        out_html = os.path.join(args.outdir, "roofline_multi.html")
        generate_multi(figs, out_html=out_html)
        print(f"Wrote {out_html}")

if __name__ == "__main__":
    main()
