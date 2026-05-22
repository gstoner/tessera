#!/usr/bin/env python3
import argparse, os, math, csv
import torch

def try_hf_tokenize(texts, tokenizer_id):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_id)
        enc = tok(texts, return_tensors='pt', padding=False, truncation=False)
        return enc['input_ids']
    except Exception:
        return None

def naive_tokenize(lines, vocab_size):
    # whitespace split, hash to ids
    ids = []
    for ln in lines:
        toks = ln.strip().split()
        if not toks: continue
        row = [abs(hash(t)) % vocab_size for t in toks]
        if len(row)>1:
            ids.append(torch.tensor(row, dtype=torch.long))
    return ids

def perplexity(model, id_seqs, seq_len=256, device='cpu'):
    model.eval()
    losses = []
    with torch.no_grad():
        for ids in id_seqs:
            # chop into segments
            for i in range(0, len(ids)-1, seq_len):
                seg = ids[i:i+seq_len+1]
                if len(seg) < 2: continue
                x = seg[:-1].unsqueeze(0).to(device)
                y = seg[1:].unsqueeze(0).to(device)
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                losses.append(loss.item())
    if not losses: return float('inf')
    mean_loss = sum(losses)/len(losses)
    return math.exp(mean_loss), mean_loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--text", required=True, help="path to a local .txt file")
    ap.add_argument("--tokenizer_id", default=None, help="HF tokenizer id/path; if omitted, fallback to naive whitespace hashing")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--log", default="logs/eval.csv")
    args = ap.parse_args()

    from tessera_gemma.configs import GemmaConfig
    from tessera_gemma.model_tessera import TesseraGemmaForCausalLM

    device = torch.device(args.device)
    cfg = GemmaConfig(vocab_size=32000, hidden_size=512, intermediate_size=1536,
                      num_hidden_layers=6, num_attention_heads=8, num_kv_heads=2, max_position_embeddings=4096)
    model = TesseraGemmaForCausalLM(cfg).to(device)

    with open(args.text, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if args.tokenizer_id is not None:
        ids = try_hf_tokenize(lines, args.tokenizer_id)
        if ids is not None:
            id_seqs = [row for row in ids]
        else:
            id_seqs = naive_tokenize(lines, cfg.vocab_size)
    else:
        id_seqs = naive_tokenize(lines, cfg.vocab_size)

    ppl, xent = perplexity(model, id_seqs, seq_len=args.seq_len, device=device)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    with open(args.log, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text","ppl","xent"])
        if f.tell()==0: w.writeheader()
        w.writerow({"text": os.path.basename(args.text), "ppl": ppl, "xent": xent})
    print(f"PPL: {ppl:.3f}  (xent={xent:.4f})  logged to {args.log}")

if __name__ == "__main__":
    main()
