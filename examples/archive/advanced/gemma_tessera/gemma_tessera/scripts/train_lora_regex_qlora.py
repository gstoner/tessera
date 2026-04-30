#!/usr/bin/env python3
"""
Minimal LoRA trainer with regex targeting + QLoRA-sim and paged-decode eval.
- If HF datasets aren't available, falls back to random text data.
- Logs CSV to logs/train.csv and uses NVTX if present.
"""
import argparse, os, math, csv, time, contextlib
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import nvtx
except Exception:
    class _NVTX:
        def range(self, *_args, **_kwargs): return contextlib.nullcontext()
    nvtx = _NVTX()

from tessera_gemma.configs import GemmaConfig
from tessera_gemma.model_tessera import TesseraGemmaForCausalLM
from tessera_gemma.peft.lora import apply_lora_regex, apply_lora, apply_qlora_sim, lora_state_dict, merge_lora, freeze_by_regex, param_groups_with_adapter_lrmult
from tessera_gemma.utils.decoding import greedy_generate_paged

class RandomText(Dataset):
    def __init__(self, vocab, n=1000, T=256, device="cpu"):
        self.vocab=vocab; self.n=n; self.T=T; self.device=device
    def __len__(self): return self.n
    def __getitem__(self, idx):
        x = torch.randint(0, self.vocab, (self.T,), device=self.device)
        return x[:-1], x[1:]

def log_csv(path, row: dict, header=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header or sorted(row.keys()))
        if new: w.writeheader()
        w.writerow(row)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--freeze", type=str, default="embed,norm", help="comma list: substrings to freeze (e.g., embed,norm)")
    ap.add_argument("--adapter_lr", type=str, default="attn=2.0,mlp=1.0,head=1.5", help="adapter LR multipliers as comma list (name=mult)")
    ap.add_argument("--rank_attn", type=int, default=16)
    ap.add_argument("--rank_mlp", type=int, default=8)
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--log", default="logs/train.csv")
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = GemmaConfig(vocab_size=32000, hidden_size=512, intermediate_size=1536,
                      num_hidden_layers=6, num_attention_heads=8, num_kv_heads=2, max_position_embeddings=4096)
    model = TesseraGemmaForCausalLM(cfg).to(device)

    # Apply LoRA via regex rules
    rules = [
        {'pattern': r'.*attn.*(qkv|proj).*', 'name': 'attn', 'rank': args.rank_attn, 'alpha': args.rank_attn*2, 'dropout': 0.0},
        {'pattern': r'.*mlp.*(wi|wo).*', 'name': 'mlp', 'rank': args.rank_mlp, 'alpha': args.rank_mlp*2, 'dropout': 0.0},
        {'pattern': r'.*lm_head.*', 'name': 'head', 'rank': args.rank_mlp, 'alpha': args.rank_mlp*2, 'dropout': 0.0},
    ]
    created = apply_lora_regex(model, rules)
    print(f"Created {created} LoRA adapters via regex rules")

    # QLoRA simulation (Int4 on base weights) if requested
    if args.qlora:
        wrapped = apply_qlora_sim(model, patterns=("qkv","proj","wi","wo"))
        print(f"Applied QLoRA-sim wrappers to {wrapped} Linear layers")

    ds = RandomText(cfg.vocab_size, n=2000, T=256, device=device)
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    # Freeze by regex
    freeze_pats = tuple([s for s in args.freeze.split(',') if s])
    frozen = freeze_by_regex(model, freeze_pats)
    print(f"Frozen params by regex {freeze_pats}: {frozen}")

    # Adapter LR multipliers
    alr = {}
    for kv in [s for s in args.adapter_lr.split(',') if s]:
        if '=' in kv:
            k,v = kv.split('=',1); alr[k.strip()] = float(v)
    print(f"Adapter LR multipliers: {alr}")
    pg = param_groups_with_adapter_lrmult(model, base_lr=args.lr, adapter_lr_mult=alr)
    opt = torch.optim.AdamW(pg)

    step = 0
    for x,y in dl:
        if step >= args.steps: break
        with nvtx.range("forward"):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        with nvtx.range("backward"):
            opt.zero_grad(set_to_none=True); loss.backward()
        with nvtx.range("step"):
            opt.step()

        if step % 10 == 0:
            print(f"step {step} loss {loss.item():.4f}")
        log_csv(args.log, {"step": step, "loss": float(loss.item())})

        if args.eval_every>0 and step>0 and step % args.eval_every == 0:
            with torch.no_grad():
                prompt = torch.randint(0, cfg.vocab_size, (1, 32), device=device)
                with nvtx.range("eval_paged_decode"):
                    _ = greedy_generate_paged(model, prompt, max_new_tokens=32)

        step += 1

    torch.save(lora_state_dict(model), "lora_adapter.pt")
    merged = merge_lora(model)
    print(f"Saved LoRA adapters and merged {merged} adapters for export")

if __name__ == "__main__":
    main()
