import argparse, csv, math, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def load_df(csv_path):
    return pd.read_csv(csv_path)

def chart_tflops_vs_S(df, D_values, out_dir):
    imgs = []
    for D in D_values:
        sub = df[df['D']==D].sort_values('S')
        if sub.empty: continue
        plt.figure()
        for dp in sorted(sub['dropout_p'].unique()):
            for cz in sorted(sub['causal'].unique()):
                curve = sub[(sub['dropout_p']==dp) & (sub['causal']==cz)]
                if curve.empty: continue
                lbl = f"D={D}, drop={dp}, causal={cz}"
                plt.plot(curve['S'], curve['tflops_p50'], marker='o', label=lbl)
        plt.xlabel("Sequence length S")
        plt.ylabel("TFLOP/s (p50)")
        plt.title(f"FlashAttn tiled: TFLOP/s vs S (D={D})")
        plt.legend()
        img = Path(out_dir)/f"tflops_vs_S_D{D}.png"
        plt.savefig(img, bbox_inches='tight'); plt.close()
        imgs.append(str(img))
    return imgs

def chart_roofline(df, out_dir):
    plt.figure()
    plt.scatter(df['ai_est'], df['tflops_p50'])
    plt.xlabel("Arithmetic intensity (FLOP/byte)")
    plt.ylabel("TFLOP/s (p50)")
    plt.title("FlashAttn tiled: Roofline-ish scatter")
    img = Path(out_dir)/"roofline_scatter.png"
    plt.savefig(img, bbox_inches='tight'); plt.close()
    return str(img)

def write_markdown(df, imgs, roofline_img, out_md):
    with open(out_md, "w") as f:
        f.write("# Tessera FlashAttention Tiled — Perf Sweep Report\n\n")
        f.write("## Summary\n")
        f.write(f"- Rows: **{len(df)}**\n\n")
        f.write("## Charts\n")
        for p in imgs:
            f.write(f"![chart]({p})\n\n")
        f.write(f"![roofline]({roofline_img})\n")

def write_html(df, imgs, roofline_img, out_html):
    html = ["<html><head><meta charset='utf-8'><title>Tessera FlashAttn Sweep</title></head><body>"]
    html.append("<h1>Tessera FlashAttention Tiled — Perf Sweep Report</h1>")
    html.append(f"<p>Rows: <b>{len(df)}</b></p>")
    html.append("<h2>Charts</h2>")
    for p in imgs:
        html.append(f"<div><img src='{p}' style='max-width:100%'></div>")
    html.append(f"<div><img src='{roofline_img}' style='max-width:100%'></div>")
    html.append("</body></html>")
    Path(out_html).write_text("\n".join(html), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="runs/flashattn_sweep.csv")
    ap.add_argument("--outdir", type=str, default="runs")
    args = ap.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    df = load_df(args.csv)
    D_values = sorted(df['D'].unique())
    imgs = chart_tflops_vs_S(df, D_values, args.outdir)
    roof = chart_roofline(df, args.outdir)
    out_md = Path(args.outdir)/"flashattn_sweep_report.md"
    out_html = Path(args.outdir)/"flashattn_sweep_report.html"
    write_markdown(df, imgs, roof, out_md)
    write_html(df, imgs, roof, out_html)
    print("Wrote:", out_md, out_html)

if __name__ == "__main__":
    main()
