#!/usr/bin/env python3
import argparse, torch, math
from torch.utils.data import DataLoader, Dataset
from tessera_gemma.configs import GemmaConfig
from tessera_gemma.model_tessera import TesseraGemmaForCausalLM
from tessera_gemma.peft.lora import apply_lora, lora_state_dict, load_lora_state_dict, merge_lora, unmerge_lora

class ToyDataset(Dataset):
    def __init__(self, vocab, n=1024, T=128, device="cpu"):
        self.vocab=vocab; self.n=n; self.T=T; self.device=device
    def __len__(self): return self.n
    def __getitem__(self, idx):
        x = torch.randint(0, self.vocab, (self.T,), device=self.device)
        return x[:-1], x[1:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = GemmaConfig(vocab_size=32000, hidden_size=512, intermediate_size=1536,
                      num_hidden_layers=4, num_attention_heads=8, num_kv_heads=2, max_position_embeddings=2048)
    model = TesseraGemmaForCausalLM(cfg).to(device)
    wrapped = apply_lora(model, rank=args.rank, alpha=args.alpha, dropout=0.0)
    print(f"Applied LoRA to {wrapped} Linear layers")

    ds = ToyDataset(cfg.vocab_size, n=256, T=128, device=device)
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    model.train()
    for step, batch in enumerate(dl):
        if step >= args.steps: break
        x, y = batch
        x = x.to(device); y = y.to(device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 25 == 0:
            print(f"step {step}  loss {loss.item():.4f}")

    # Save LoRA-only state
    state = lora_state_dict(model)
    torch.save(state, "lora_adapter.pt")
    print("Saved LoRA adapters to lora_adapter.pt")

    # (optional) Merge LoRA into base weights for export
    merged = merge_lora(model); print(f"Merged {merged} LoRA layers into base weights")

if __name__ == "__main__":
    main()
