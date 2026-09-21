# tools/postnas_driver.py
"""Small PostNAS driver that runs a tiny search and logs throughput/accuracy.

This uses the scaffold's PostNASSearch and a dummy eval function that
returns approximate throughput & an accuracy proxy (random/stub).
Replace `eval_fn` with your real validation harness.

Run (pseudo):
    python tools/postnas_driver.py
Outputs:
    - logs/postnas_results.json
    - logs/postnas_results.csv
"""
import os, time, json, csv, random
from tessera_jetnemotron.postnas_pipeline import PostNASConfig, PostNASSearch
from tessera_jetnemotron.transformer_block import Transformer, TransformerConfig

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def make_model(n_layers=8, d_model=256, n_heads=8):
    cfg = TransformerConfig(
        d_model=d_model, n_heads=n_heads, head_dim=d_model//n_heads,
        mlp_hidden=4*d_model, n_layers=n_layers,
        attn_types=["full"]*n_layers, dropout_p=0.0,
        dtype="fp8_e4m3", accum="fp32"
    )
    return Transformer(cfg)

def eval_fn(model) -> dict:
    # Simple proxy: measure elapsed time for a couple of forward passes with synthetic data
    import numpy as np
    from tessera import Tensor
    B,S,D = 1, 128, model.cfg.d_model
    x = Tensor(np.random.randn(B,S,D).astype(np.float32))
    t0 = time.perf_counter()
    for _ in range(4):
        _y, _ = model(x, causal=True, streaming=False)
    dt = time.perf_counter() - t0
    throughput = (B*S*4) / (dt + 1e-9)  # tokens/sec (synthetic)
    # Stub accuracy proxy (replace with validation metric)
    acc = 0.5 + 0.5*random.random()
    return {"throughput": throughput, "mmlu": acc}

def main():
    model = make_model()
    cfg = PostNASConfig(
        search_layers=list(range(model.cfg.n_layers)),
        keep_full_attn_budget=2,
        jetblock_space={"feature_map":["elu1","rf"], "conv_ks":[5,7], "gate":["token","head"]},
        schedule_space={"block":[(128,128)], "warps":[4,8], "stages":[2,3]},
        metric="throughput_vs_mmlu")
    driver = PostNASSearch(model, cfg, eval_fn=eval_fn)
    best = driver.search(budget=6)

    # Save results
    json_path = os.path.join(LOG_DIR, "postnas_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    csv_path = os.path.join(LOG_DIR, "postnas_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["throughput","mmlu","full_layers","jetblock"]) 
        w.writerow([best['scores']['throughput'], best['scores']['mmlu'], 
                    sorted(list(best['full_layers'])), best['jetblock']])
    print("Saved:", json_path, "and", csv_path)

if __name__ == "__main__":
    main()
