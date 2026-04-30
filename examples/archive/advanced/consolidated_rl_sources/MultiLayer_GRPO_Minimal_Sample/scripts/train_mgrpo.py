
#!/usr/bin/env python3
import argparse, yaml, random
from mgrpo.trainer import MGRPOTrainer, MGRPOConfig
from mgrpo.dataset import load_jsonl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    tcfg = MGRPOConfig(**cfg["trainer"])
    trainer = MGRPOTrainer(tcfg)

    rows = load_jsonl(cfg["data"]["path"])
    n_steps = cfg.get("num_steps", 3)
    batch_size = cfg.get("batch_size", 2)

    for step in range(n_steps):
        batch = random.sample(rows, batch_size)
        stats = trainer.train_step(batch)
        scalars = " ".join([f"{k}={v:.4f}" for k,v in stats.items() if isinstance(v, (int,float))])
        print(f"step {step}: {scalars}")

if __name__ == "__main__":
    main()
