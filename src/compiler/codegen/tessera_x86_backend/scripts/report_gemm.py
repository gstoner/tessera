import argparse, csv, os, io, base64
import pandas as pd
import matplotlib.pyplot as plt

def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV from bench_gemm")
    ap.add_argument("--out", default="bench_report.html")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    bf16 = df[df["dtype"]=="bf16"].copy()
    i8   = df[df["dtype"]=="int8"].copy()

    img_bf16 = ""
    img_int8 = ""

    # Chart 1: Throughput vs size (GFLOPs) for BF16
    if not bf16.empty:
        bf16 = bf16.copy()
        bf16["shape"] = bf16.apply(lambda r: f'{int(r["M"])}x{int(r["N"])}x{int(r["K"])}', axis=1)
        plt.figure()
        plt.plot(bf16["shape"], bf16["gflops"], marker='o')
        plt.title("BF16 GEMM: Throughput (GFLOP/s)")
        plt.xlabel("M x N x K")
        plt.ylabel("GFLOP/s")
        img_bf16 = fig_to_base64()
        plt.close()

    # Chart 1b: INT8
    if not i8.empty:
        i8 = i8.copy()
        i8["shape"] = i8.apply(lambda r: f'{int(r["M"])}x{int(r["N"])}x{int(r["K"])}', axis=1)
        plt.figure()
        plt.plot(i8["shape"], i8["gflops"], marker='o')
        plt.title("INT8 GEMM: Throughput (GFLOP/s)")
        plt.xlabel("M x N x K")
        plt.ylabel("GFLOP/s")
        img_int8 = fig_to_base64()
        plt.close()

    # Chart 2: Roofline-like scatter
    plt.figure()
    if not bf16.empty:
        plt.scatter(bf16["intensity"], bf16["gflops"], label="BF16", marker='o')
    if not i8.empty:
        plt.scatter(i8["intensity"], i8["gflops"], label="INT8", marker='x')
    plt.title("Arithmetic Intensity vs Throughput")
    plt.xlabel("FLOPs / Byte (approx)")
    plt.ylabel("GFLOP/s")
    plt.legend()
    img_roof = fig_to_base64()
    plt.close()

    html_tpl = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Tessera x86 Backend GEMM Benchmark Report</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; padding: 24px; }}
h1,h2 {{ margin: 0.2em 0; }}
.card {{ border: 1px solid #eee; border-radius: 10px; padding: 16px; margin-bottom: 16px; }}
code {{ background: #f6f8fa; padding: 2px 6px; border-radius: 4px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
th {{ background: #fafafa; }}
.small {{ color: #666; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>Tessera x86 Backend — GEMM Micro-Bench Report</h1>
<p class="small">FLOPs ≈ 2·M·N·K; bytes ≈ |A|+|B|+|C| (beta=0).</p>

<div class="card">
  <h2>Throughput: BF16</h2>
  {bf16_img}
</div>

<div class="card">
  <h2>Throughput: INT8</h2>
  {int8_img}
</div>

<div class="card">
  <h2>Arithmetic Intensity vs Throughput</h2>
  <img src="data:image/png;base64,{roof_img}"/>
</div>

<div class="card">
  <h2>Raw Data</h2>
  <pre>{csv_dump}</pre>
</div>

</body>
</html>
"""
    html = html_tpl.format(
        bf16_img = f"<img src='data:image/png;base64,{img_bf16}'/>" if img_bf16 else "<p>No BF16 data.</p>",
        int8_img = f"<img src='data:image/png;base64,{img_int8}'/>" if img_int8 else "<p>No INT8 data.</p>",
        roof_img = img_roof,
        csv_dump = df.to_csv(index=False)
    )
    with open(args.out, "w") as f:
        f.write(html)
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
