import argparse, io, base64, pandas as pd, matplotlib.pyplot as plt

def fig_to_b64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    import base64
    return base64.b64encode(buf.read()).decode("ascii")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="gpu_bench_report.html")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # Plots
    imgs = {}
    for name, sub in df.groupby("dtype"):
        plt.figure()
        labels = [f"{int(m)}x{int(n)}x{int(k)}" for m,n,k in zip(sub.M, sub.N, sub.K)]
        plt.plot(labels, sub.tflops, marker="o")
        plt.title(f"{name.upper()} throughput (TFLOP/s)")
        plt.xlabel("M x N x K")
        plt.ylabel("TFLOP/s")
        imgs[name] = fig_to_b64()
        plt.close()

    # HTML
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>GPU Bench Report</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; padding: 24px; }}
.card {{ border:1px solid #eee; border-radius:10px; padding:16px; margin-bottom:16px; }}
</style></head><body>
<h1>Tessera GPU Benchmark</h1>
<div class="card"><h2>FP16 WMMA vs BF16 WMMA vs BF16 WGMMA vs INT8 IMMA</h2></div>
<div class="card"><h3>FP16/WMMA</h3>{('<img src="data:image/png;base64,'+imgs.get('fp16','')+'"/>') if 'fp16' in imgs else '<p>No fp16 data.</p>'}</div>
<div class="card"><h3>BF16 (WMMA)</h3>{('<img src="data:image/png;base64,'+imgs.get('bf16','')+'"/>') if 'bf16' in imgs else '<p>No bf16 data.</p>'}</div>
<div class="card"><h3>INT8 (IMMA)</h3>{('<img src="data:image/png;base64,'+imgs.get('int8','')+'"/>') if 'int8' in imgs else '<p>No int8 data.</p>'}</div>
<div class="card"><h2>Raw CSV</h2><pre>{df.to_csv(index=False)}</pre></div>
</body></html>"""
    with open(args.out, "w") as f:
        f.write(html)
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
