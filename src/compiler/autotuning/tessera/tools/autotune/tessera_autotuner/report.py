
import os, json, pandas as pd

def load_results(out_dir: str) -> pd.DataFrame:
    p = os.path.join(out_dir, "results.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    cfgs = df["config"].apply(lambda s: json.loads(s))
    cfg_df = pd.json_normalize(cfgs)
    return pd.concat([df.drop(columns=["config"]), cfg_df], axis=1)

def save_html_report(out_dir: str, top_k: int = 20):
    df = load_results(out_dir)
    key = "tflops" if "tflops" in df.columns else "latency_ms"
    asc = (key == "latency_ms")
    sdf = df.sort_values(key, ascending=asc).head(top_k)
    html = "<html><head><meta charset='utf-8'><title>Tessera Autotune Report</title></head><body>"
    html += "<h1>Tessera Autotune Report</h1>"
    html += f"<p>Total trials: {len(df)}</p>"
    html += sdf.to_html(index=False)
    html += "</body></html>"
    with open(os.path.join(out_dir, "report.html"), "w") as f:
        f.write(html)
