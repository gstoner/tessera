# examples/train_tiny.py
"""Tiny training loop with loss, backward, gradient clipping, gradient accumulation,
mixed-precision scaler, and a tiny validation loop (perplexity).

This script trains a small mixed-attention Transformer on a synthetic next-token
prediction task. It demonstrates:
 - forward() with mixed full-attn / JetBlock layers
 - loss computation (cross-entropy)
 - backward (autodiff)
 - optimizer step (AdamW-like)
 - gradient clipping by global norm
 - **gradient accumulation**
 - **mixed-precision scaler** (loss scaling)
 - **validation loop** with perplexity

Run (pseudo):
    python examples/train_tiny.py
"""
import math, time, os, argparse
from dataclasses import dataclass

# Tessera-like surface (modeled)
from tessera import Tensor, autodiff, jit
from tessera.stdlib import rmsnorm_safe
from tessera_jetnemotron.transformer_block import Transformer, TransformerConfig
from tessera_jetnemotron.utils.autocast import Autocast
from tessera_jetnemotron.utils.checkpoint import save_checkpoint, load_checkpoint
from tessera_jetnemotron.tools.schedule_cache import export_schedule_cache

# ----------------------
# Synthetic datasets
# ----------------------
class ToyDataset:
    def __init__(self, vocab_size=2048, seq_len=128, num_batches=200, seed=0):
        import numpy as np
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_batches = num_batches
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        import numpy as np
        for _ in range(self.num_batches):
            B = 8
            x = self.rng.integers(0, self.vocab_size, size=(B, self.seq_len), endpoint=False, dtype=np.int32)
            y = self.rng.integers(0, self.vocab_size, size=(B, self.seq_len), endpoint=False, dtype=np.int32)
            # toy embeddings
            emb = self.rng.standard_normal(size=(self.vocab_size, 256)).astype(np.float32)
            x_emb = emb[x]  # (B,S,D=256)
            yield Tensor(x_emb), Tensor(y)

def make_dataloaders(vocab_size=2048, seq_len=128, train_batches=60, val_batches=10):
    train_ds = ToyDataset(vocab_size=vocab_size, seq_len=seq_len, num_batches=train_batches, seed=0)
    val_ds   = ToyDataset(vocab_size=vocab_size, seq_len=seq_len, num_batches=val_batches, seed=1234)
    return train_ds, val_ds

# ----------------------
# Simple AdamW optimizer
# ----------------------
class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr; self.betas = betas; self.eps = eps; self.wd = weight_decay
        self.m = {id(p): Tensor.zeros_like(p) for p in self.params}
        self.v = {id(p): Tensor.zeros_like(p) for p in self.params}
        self.t = 0

    def step(self):
        b1, b2 = self.betas
        self.t += 1
        for p in self.params:
            if not hasattr(p, 'grad') or p.grad is None: continue
            g = p.grad
            pid = id(p)
            self.m[pid] = b1*self.m[pid] + (1-b1)*g
            self.v[pid] = b2*self.v[pid] + (1-b2)*(g*g)
            m_hat = self.m[pid] / (1 - b1**self.t)
            v_hat = self.v[pid] / (1 - b2**self.t)
            # decoupled weight decay
            p -= self.lr * (m_hat / (v_hat.sqrt() + self.eps) + self.wd * p)

    def zero_grad(self):
        for p in self.params:
            if hasattr(p, 'grad') and p.grad is not None:
                p.grad.fill_(0.0)

# ----------------------
# Loss & helpers
# ----------------------
def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    # logits: (B,S,V), targets: (B,S)
    # numerically stable softmax-xent
    m = logits.max(dim=-1, keepdim=True)
    z = (logits - m).exp()
    logZ = z.sum(dim=-1, keepdim=True).log()
    nll = -(logits.gather(dim=-1, index=targets[...,None]) - m - logZ).squeeze(-1)
    return nll.mean()

def global_grad_norm(params):
    total = 0.0
    for p in params:
        if hasattr(p, 'grad') and p.grad is not None:
            total += (p.grad * p.grad).sum().item()
    return math.sqrt(max(total, 1e-30))

def clip_grad_norm_(params, max_norm=1.0):
    gnorm = global_grad_norm(params)
    if gnorm <= max_norm:
        return gnorm
    scale = max_norm / (gnorm + 1e-12)
    for p in params:
        if hasattr(p, 'grad') and p.grad is not None:
            p.grad *= scale
    return gnorm

# ----------------------
# Mixed-precision GradScaler (loss scaling)
# ----------------------
class GradScaler:
    def __init__(self, init_scale=2.**14, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._good_steps = 0

    def scale_loss(self, loss: Tensor) -> Tensor:
        return loss * self.scale

    def unscale_(self, params):
        inv = 1.0 / self.scale
        for p in params:
            if hasattr(p, 'grad') and p.grad is not None:
                p.grad *= inv

    def _has_overflow(self, params) -> bool:
        for p in params:
            if hasattr(p, 'grad') and p.grad is not None:
                # Simple overflow heuristic
                if not p.grad.isfinite().all():  # placeholder API
                    return True
        return False

    def update(self, overflow: bool):
        if overflow:
            self.scale = max(self.scale * self.backoff_factor, 1.0)
            self._good_steps = 0
        else:
            self._good_steps += 1
            if self._good_steps % self.growth_interval == 0:
                self.scale *= self.growth_factor

# ----------------------
# Model head
# ----------------------
class LMHead:
    def __init__(self, d_model, vocab_size):
        from tessera import nn
        self.proj = nn.Linear(d_model, vocab_size, dtype="bf16", accum="fp32")
        self.parameters = self.proj.parameters

    def __call__(self, x: Tensor):
        return self.proj(x)  # (B,S,V)


# ----------------------
# LR Schedulers
# ----------------------
class LRSchedulerBase:
    def step(self): ...
    def state_dict(self): return {"_": 0}
    def load_state_dict(self, st): return self

class CosineLR(LRSchedulerBase):
    def __init__(self, optimizer, max_steps, min_lr=0.0, base_lr=None, warmup_steps=0):
        self.opt = optimizer
        self.t = 0
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.base_lr = base_lr if base_lr is not None else optimizer.lr
        self.warmup_steps = warmup_steps

    def step(self):
        self.t += 1
        if self.t <= self.warmup_steps and self.warmup_steps > 0:
            lr = self.base_lr * self.t / self.warmup_steps
        else:
            progress = (self.t - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            lr = self.min_lr + 0.5*(self.base_lr - self.min_lr)*(1 + math.cos(math.pi*progress))
        self.opt.lr = lr

    def state_dict(self): return {"t": self.t, "base_lr": self.base_lr}
    def load_state_dict(self, st): self.t = st.get("t", 0); self.base_lr = st.get("base_lr", self.base_lr); return self

class ConstWithWarmup(LRSchedulerBase):
    def __init__(self, optimizer, base_lr, warmup_steps=100):
        self.opt = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.t = 0

    def step(self):
        self.t += 1
        if self.t <= self.warmup_steps:
            lr = self.base_lr * self.t / self.warmup_steps
        else:
            lr = self.base_lr
        self.opt.lr = lr

    def state_dict(self): return {"t": self.t, "base_lr": self.base_lr}
    def load_state_dict(self, st): self.t = st.get("t", 0); self.base_lr = st.get("base_lr", self.base_lr); return self

# ----------------------
# Train/Val steps (JIT + autodiff)
# ----------------------
@autodiff
@jit
def forward_loss(model, head, x, y):
    logits, _ = model(x, causal=True, streaming=False)
    logits = head(logits)
    return cross_entropy(logits, y)

def validate(model, head, val_loader):
    total_loss = 0.0
    total_tokens = 0
    for x, y in val_loader:
        with Tensor.no_grad():  # pretend API for eval mode
            loss = forward_loss(model, head, x, y)
        total_loss += loss.item() * (y.shape[0] * y.shape[1])
        total_tokens += (y.shape[0] * y.shape[1])
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

def main():
    parser = argparse.ArgumentParser(description="Tiny Tessera Trainer (Jet-Nemotron scaffold)")
    parser.add_argument("--vocab", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=5e-5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--sched", type=str, choices=["cosine","const"], default="cosine")
    parser.add_argument("--train-batches", type=int, default=80)
    parser.add_argument("--val-batches", type=int, default=12)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=32, help="save checkpoint every N optimizer steps")
    parser.add_argument("--resume", type=str, default="", help="path to checkpoint json to resume from")
    parser.add_argument("--sched-cache-dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "logs", "sched_cache"))
    args = parser.parse_args()

    # Hyperparams
    D = args.d_model
    vocab = args.vocab
    grad_clip = args.grad_clip
    accum_steps = args.accum_steps
    train_batches = args.train_batches
    val_batches = args.val_batches

    # Model
    cfg = TransformerConfig(
        d_model=D, n_heads=args.heads, head_dim=D//args.heads, mlp_hidden=4*D,
        n_layers=args.layers, attn_types=["full" if i % 4 == 0 else "jet" for i in range(args.layers)],
        dropout_p=0.0, dtype="fp8_e4m3", accum="fp32"
    )
    model = Transformer(cfg)
    head = LMHead(D, vocab_size=vocab)
    opt = AdamW(model.parameters() + head.parameters(), lr=args.lr)
    scaler = GradScaler()

    # Resume
    if args.resume:
        try:
            ckpt = load_checkpoint(args.resume, model=model, head=head, optimizer=opt, scaler=smpl_object if False else scaler, scheduler=scheduler, schedule_cache_dir=args.sched_cache_dir)
            print(f"[resume] loaded checkpoint from {args.resume} (ir_hash={ckpt.get('ir_hash')})")
        except Exception as e:
            print(f"[resume] failed to load: {e}")
    # Choose one scheduler
    if args.sched == "cosine":
        scheduler = CosineLR(opt, max_steps=train_batches, min_lr=args.min_lr, base_lr=args.lr, warmup_steps=args.warmup)
    else:
        scheduler = ConstWithWarmup(opt, base_lr=args.lr, warmup_steps=args.warmup)

    # Data
    train_loader, val_loader = make_dataloaders(vocab_size=vocab, seq_len=args.seq_len,
                                                train_batches=train_batches, val_batches=val_batches)

    # Loop
    t0 = time.perf_counter()
    opt.zero_grad()
    running_tokens = 0
    for step, (x, y) in enumerate(train_loader, 1):
        # Forward with autocast
        with Autocast(enabled=True, compute_dtype="fp16", accum_dtype="fp32"):
            loss = forward_loss(model, head, x, y)
        # Mixed precision: loss scaling
        scaled = scaler.scale_loss(loss / accum_steps)
        scaled.backward()  # accumulate scaled grads

        running_tokens += x.shape[0] * x.shape[1]

        if step % accum_steps == 0:
            # Unscale grads and clip
            scaler.unscale_(model.parameters() + head.parameters())
            gnorm = clip_grad_norm_(model.parameters() + head.parameters(), max_norm=grad_clip)

            # Check overflow (skip step if overflow)
            overflow = False  # placeholder; set True if your backend flags overflow
            if not overflow:
                opt.step()
            opt.zero_grad()
            scaler.update(overflow)

            scheduler.step()
            # Train throughput logging
            dt = time.perf_counter() - t0
            tput = running_tokens / (dt + 1e-9)
            print(f"step {step:04d}  loss {loss.item():.4f}  gnorm {gnorm:.2f}  tokens/s {tput:,.0f}  scale {scaler.scale:.1f}  lr {opt.lr:.2e}")
            # Save checkpoint (scaffold) every N accumulated steps
            if step % args.save_every == 0:
                ckpt_path = os.path.join(os.path.dirname(__file__), "..", "logs", "ckpt_step%04d.json" % step)
                os.makedirs(os.path.join(os.path.dirname(__file__), "..", "logs"), exist_ok=True)
                irh = save_checkpoint(ckpt_path, model=model, head=head, optimizer=opt, scaler=scaler,
                                      scheduler=scheduler, schedule_cache_dir=os.path.join(os.path.dirname(__file__), "..", "logs", "sched_cache"))
                print(f"[ckpt] saved {ckpt_path}  ir_hash={irh}")
            t0 = time.perf_counter()
            running_tokens = 0

        # Periodic validation
        if step % max(8*accum_steps, 32) == 0:
            val_loss, val_ppl = validate(model, head, val_loader)
            print(f"[val] loss {val_loss:.4f}  ppl {val_ppl:.2f}")

if __name__ == "__main__":
    main()
