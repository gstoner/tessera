
import os, json, argparse, functools
from .spaces import SearchSpace
from .search import GridSearch, RandomSearch
from .hyperband import Hyperband
from .evaluate import Autotuner, SyntheticGEMMWorkload
from .utils import Objective

ALGOS = ("grid","random","hyperband")

def build_space(spec: dict) -> SearchSpace:
    sp = SearchSpace()
    for p in spec.get("params", []):
        sp.add(p["name"], p["kind"], p["values"])
    return sp

def build_workload(spec: dict):
    wl = spec.get("workload", {})
    kind = wl.get("kind","synthetic_gemm")
    if kind == "synthetic_gemm":
        return SyntheticGEMMWorkload(
            wl.get("M",4096), wl.get("N",4096), wl.get("K",4096),
            device=wl.get("device","cuda"),
            dtype=wl.get("dtype","bf16"),
            iters=wl.get("iters",30),
            warmup=wl.get("warmup",5),
        )
    raise ValueError(f"Unknown workload kind: {kind}")

def main():
    ap = argparse.ArgumentParser(description="Tessera Autotune")
    ap.add_argument("--config", required=True)
    ap.add_argument("-o","--out", default="runs/autotune_run")
    ap.add_argument("--algo", choices=ALGOS, default="random")
    ap.add_argument("--max-trials", type=int, default=64)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--objective", type=str, default="max:tflops")
    # Hyperband specific
    ap.add_argument("--hb-min-budget", type=int, default=5)
    ap.add_argument("--hb-max-budget", type=int, default=30)
    ap.add_argument("--hb-eta", type=int, default=3)
    args = ap.parse_args()

    spec = json.load(open(args.config))
    space = build_space(spec)
    workload = build_workload(spec)
    objective = Objective.parse(args.objective)

    os.makedirs(args.out, exist_ok=True)
    tuner = Autotuner(objective, out_dir=args.out)

    if args.algo == "grid":
        candidates = GridSearch(space).candidates()
        best_cfg, best_metrics = tuner.run(candidates, workload)
    elif args.algo == "random":
        candidates = RandomSearch(space, max_trials=args.max_trials, seed=args.seed).candidates()
        best_cfg, best_metrics = tuner.run(candidates, workload, max_trials=args.max_trials)
    else:
        # Hyperband: sampler draws random configs from the space
        def sampler(k): 
            return list(space.random(k, seed=args.seed))
        hb = Hyperband(sampler, max_budget=args.hb_max_budget, min_budget=args.hb_min_budget, eta=args.hb_eta, seed=args.seed)
        best_cfg, best_metrics = tuner.run(hb.candidates(), workload)

    with open(os.path.join(args.out, "best.json"), "w") as f:
        json.dump({"best_config": best_cfg, "best_metrics": best_metrics}, f, indent=2)
    print("Best config:", best_cfg)
    print("Best metrics:", best_metrics)

if __name__ == "__main__":
    main()
