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

def chart_mem_io(csv_path, out_dir):
    import pandas as pd, matplotlib.pyplot as plt
    df = pd.read_csv(csv_path)
    plt.figure(); plt.plot(df['bytes']/1e6, df['gbps_p50'], marker='o')
    plt.xlabel("Bytes (MB)"); plt.ylabel("D2D memcpy GB/s (p50)"); plt.title("Device Memory Copy Bandwidth")
    img = Path(out_dir)/"mem_io_bw.png"; plt.savefig(img, bbox_inches='tight'); plt.close(); return str(img)

def chart_pcie_io(csv_path, out_dir):
    import pandas as pd, matplotlib.pyplot as plt
    df = pd.read_csv(csv_path)
    imgs = []
    for direction in ['H2D','D2H']:
        plt.figure()
        for pinned in [0,1]:
            sub = df[(df['direction']==direction) & (df['pinned']==pinned)]
            if sub.empty: continue
            lbl = f"{direction} - {'pinned' if pinned else 'pageable'}"
            plt.plot(sub['bytes']/1e6, sub['gbps_p50'], marker='o', label=lbl)
        plt.xlabel("Bytes (MB)"); plt.ylabel("GB/s (p50)"); plt.title(f"PCIe {direction} Bandwidth")
        plt.legend()
        img = Path(out_dir)/f"pcie_{direction}_bw.png"; plt.savefig(img, bbox_inches='tight'); plt.close(); imgs.append(str(img))
    return imgs

def chart_nccl(csv_path, out_dir):
    import pandas as pd, matplotlib.pyplot as plt
    df = pd.read_csv(csv_path)
    imgs = []
    for op in df['op'].unique():
        sub = df[df['op']==op].sort_values('bytes')
        plt.figure()
        plt.plot(sub['bytes']/1e6, sub['gbps_eff'], marker='o')
        plt.xlabel("Bytes (MB)"); plt.ylabel("Effective GB/s (p50)"); plt.title(f"NCCL {op} Effective BW")
        img = Path(out_dir)/f"nccl_{op}_bw.png"; plt.savefig(img, bbox_inches='tight'); plt.close(); imgs.append(str(img))
    return imgs

def maybe_add_io_sections(df, out_dir, md_lines, html_lines):
    mem_csv = Path(out_dir)/"mem_io.csv"
    pcie_csv = Path(out_dir)/"pcie_io.csv"
    nccl_csv = Path(out_dir)/"nccl_collectives.csv"
    md_lines.append("\n## I/O & Collectives\n")
    html_lines.append("<h2>I/O & Collectives</h2>")
    if mem_csv.exists():
        mimg = chart_mem_io(mem_csv, out_dir)
        md_lines.append(f"![mem]({mimg})\n")
        html_lines.append(f"<div><img src='{mimg}' style='max-width:100%'></div>")
    if pcie_csv.exists():
        pimgs = chart_pcie_io(pcie_csv, out_dir)
        for p in pimgs:
            md_lines.append(f"![pcie]({p})\n")
            html_lines.append(f"<div><img src='{p}' style='max-width:100%'></div>")
    if nccl_csv.exists():
        nimgs = chart_nccl(nccl_csv, out_dir)
        for p in nimgs:
            md_lines.append(f"![nccl]({p})\n")
            html_lines.append(f"<div><img src='{p}' style='max-width:100%'></div>")

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

    md_lines = ["# Tessera FlashAttention Tiled — Perf Sweep Report\n", "## Summary\n", f"- Rows: **{len(df)}**\n", "## Charts\n"]
    for p in imgs: md_lines.append(f"![chart]({p})\n")
    md_lines.append(f"![roofline]({roof})\n")

    html_lines = ["<html><head><meta charset='utf-8'><title>Tessera FlashAttn Sweep</title></head><body>",
                  "<h1>Tessera FlashAttention Tiled — Perf Sweep Report</h1>",
                  f"<p>Rows: <b>{len(df)}</b></p>",
                  "<h2>Charts</h2>"]
    for p in imgs: html_lines.append(f"<div><img src='{p}' style='max-width:100%'></div>")
    html_lines.append(f"<div><img src='{roof}' style='max-width:100%'></div>")

    maybe_add_io_sections(df, args.outdir, md_lines, html_lines)

    Path(out_md).write_text("\n".join(md_lines), encoding="utf-8")
    Path(out_html).write_text("\n".join(html_lines+['</body></html>']), encoding="utf-8")
    print("Wrote:", out_md, out_html)

if __name__ == "__main__":
    main()
