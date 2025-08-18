# Tessera Unified Specification

This document consolidates all Tessera specifications into a single source, including the Programming Model (Master Reference Manual) and the ABI Runtime Specification. Diagram references are included so that they can be rendered into PDF/HTML.

---

# Volume I: Programming Model (Master Reference Manual)

... [content unchanged from previous version] ...

---

# Volume II: ABI Runtime Specification

... [content unchanged from previous version] ...

---

## Appendix A: C ABI Bindings

```c
typedef struct tessera_context_t* tessera_context_handle;

typedef struct tessera_memory_t* tessera_memory_handle;
typedef struct tessera_operator_t* tessera_operator_handle;

typedef struct {
  uint32_t bm, bn, bk;
  uint32_t warps;
  uint32_t stages;
  uint32_t flags;
} tessera_launch_desc;

int tessera_context_create(tessera_context_handle* out);
int tessera_context_destroy(tessera_context_handle ctx);

int tessera_memory_alloc(tessera_context_handle ctx, size_t bytes, tessera_memory_handle* out);
int tessera_memory_free(tessera_memory_handle mem);

int tessera_launch_tile(tessera_operator_handle op, const tessera_launch_desc* desc);
```

---

## Appendix B: Calling Convention

Tile entry functions follow a fixed ABI:

- **Parameters**: passed in registers.
- **Spillover**: uses tile-local memory.
- **Return values**: placed in designated registers or tile-local scratch.

---

## Appendix C: Example ABI Usage

Mapping a HuggingFace GPT model into Tessera runtime:

1. Create context with `tessera_context_create`.
2. Allocate memory with `tessera_memory_alloc`.
3. Upload operator graph into `.tessera.ops`.
4. Launch tiles using descriptors tuned for `(arch, shape, dtype)`.
5. Synchronize streams with events.

---

## Appendix D: Binary Format Layout

**Figure ABI.3: Binary Format Layout**

Tessera binaries follow an ELF-like structure:

- `.tessera.ops` — serialized operator IR.
- `.tessera.meta` — metadata and tuning records.
- `.tessera.strtab` — string table.

Versioning encoded in ELF notes via `TESSERA_ABI_VERSION`. ABI compatibility requires bumping version on breaking changes.

---

## Appendix E: Tessera Code Examples

### Example 1: Running GPT-OSS-120B

```tessera
// Load pretrained GPT-OSS-120B model
let ctx = tessera.context_create()
let mem = tessera.memory_alloc(ctx, size=1TB)

// Load operator graph from GPT-OSS-120B Tessera export
let gpt_ops = tessera.load("gpt-oss-120b.tessera.ops")

// Configure tile descriptors
let desc = tessera.launch_desc(bm=128, bn=128, bk=64, warps=8, stages=4)

// Launch autoregressive decode step
for step in 0..N:
    tessera.launch_tile(gpt_ops.decode, desc)
    tessera.synchronize()
```

This code demonstrates how to run autoregressive decoding for GPT-OSS-120B using Tessera’s ABI, with tile shapes tuned for large model throughput.

---

### Example 2: HuggingFace Transformers in Tessera

```tessera
// Example: BERT-like encoder using Tessera IR
let ctx = tessera.context_create()

let input = tessera.tensor(shape=[B, L, D], dtype=f16)
let attention = tessera.operator("flash_attention")
let ff = tessera.operator("mlp_ffn")

// Forward pass
let x = input
for layer in 0..num_layers:
    let attn_out = attention(x)
    let ff_out = ff(attn_out)
    x = ff_out

// Classification head
let logits = tessera.linear(x, out_dim=num_classes)
```

Here, HuggingFace Transformer layers (attention + feed-forward) are expressed in Tessera’s operator graph, compiled and lowered into Tile IR for GPU execution.

---

### Example 3: HuggingFace + Tessera SFT + RLHF

```tessera
// Load pretrained model
let model = tessera.load("transformer_base.tessera.ops")

// Inputs
let logits = tessera.invoke(model, x)

// Supervised loss
let loss_sft = tessera.cross_entropy(logits, y)

// Reward model
let reward = tessera.policy_eval(logits, y)

// Combined loss
let loss = tessera.add(loss_sft, lambda * reward)
```

This example shows how Tessera fuses supervised fine-tuning (cross-entropy) with reinforcement learning reward shaping, directly mapping to the examples in Chapter 6.

---

End of Specification.



---

## Appendix E: Integration Examples — GPT‑OSS‑120B & Hugging Face Transformers *(Informative / Code Samples)*

> The following **Tessera code examples** illustrate end‑to‑end flows for (1) a large open model ("GPT‑OSS‑120B") and (2) importing models from **Hugging Face Transformers**. The code is **illustrative Tessera‑style pseudocode** matching the concepts in this spec (operators, tiles, schedules, ABI dispatch). Adjust names/types to your implementation.

### E.1 GPT‑OSS‑120B — Inference Graph & Runtime Execution

```python
# tessera_pseudo.py
from tessera import op, graph, schedule as sch, runtime as rt

# --- Model metadata (placeholder) ---
D_MODEL   = 8192
N_HEAD    = 64
D_HEAD    = D_MODEL // N_HEAD
N_LAYERS  = 96
VOCAB     = 128000
POLICY    = op.numeric(dtype="bf16", accum="f32")

# --- Operator library (weights preloaded from checkpoints) ---
TokEmb   = op.embedding(VOCAB, D_MODEL, policy=POLICY)
PosEnc   = op.rotary(d_model=D_MODEL)

def DecoderLayer(i):
    Q_proj = op.linear(D_MODEL, D_MODEL, name=f"Q{i}", policy=POLICY)
    K_proj = op.linear(D_MODEL, D_MODEL, name=f"K{i}", policy=POLICY)
    V_proj = op.linear(D_MODEL, D_MODEL, name=f"V{i}", policy=POLICY)
    O_proj = op.linear(D_MODEL, D_MODEL, name=f"O{i}", policy=POLICY)
    FFN_1  = op.linear(D_MODEL, 4*D_MODEL, name=f"FF1_{i}", policy=POLICY)
    FFN_2  = op.linear(4*D_MODEL, D_MODEL, name=f"FF2_{i}", policy=POLICY)
    LN_1   = op.layernorm(D_MODEL, eps=1e-5)
    LN_2   = op.layernorm(D_MODEL, eps=1e-5)
    Attn   = op.flash_attention(num_heads=N_HEAD, head_dim=D_HEAD,
                                causal=True, policy=POLICY)
    return LN_1, Q_proj, K_proj, V_proj, Attn, O_proj, LN_2, FFN_1, FFN_2

# --- Build the operator graph ---
with graph.module("gpt_oss_120b") as G:
    x_tokens = graph.input("tokens", shape=(1, None))
    x = TokEmb(x_tokens)
    x = PosEnc(x)
    kv_cache = op.kvcache_init(layers=N_LAYERS, heads=N_HEAD, dim=D_HEAD)

    for i in range(N_LAYERS):
        LN1, Qp, Kp, Vp, Attn, Opj, LN2, F1, F2 = DecoderLayer(i)
        h = LN1(x)
        Q = Qp(h); K = Kp(h); V = Vp(h)
        ctx, kv_cache = Attn(Q, K, V, kv_cache=kv_cache, layer=i)
        x = x + Opj(ctx)                    # residual
        h2 = LN2(x)
        x = x + F2(op.gelu(F1(h2)))        # MLP + residual

    logits = op.linear(D_MODEL, VOCAB, name="LMHead", policy=POLICY)(x)
    G.output("logits", logits)

# --- Scheduling & compilation ---
# Provide default schedules; autotuner can overwrite
sch.set_default("flash_attention", bm=128, bn=128, bk=64, warps=8, stages=3)
sch.set_default("matmul", bm=256, bn=128, bk=64, warps=8, stages=2)

bundle = graph.compile(G, targets=["cuda_sm90"], deterministic=True)

# --- Runtime execution ---
ctx = rt.create_context(target="cuda_sm90")
rt.load_bundle(ctx, bundle)

state = rt.allocate_state(ctx, G)   # allocs kv-cache, temps
inputs = {"tokens": rt.tensor_from_list([1, 2, 3, 4])}
outputs = rt.run(ctx, G, inputs, state)
print(outputs["logits"].shape)
```

**Notes**

- `op.flash_attention` lowers to the Tile dialect with fused softmax (see Chapter 6).
- `op.kvcache_init` materializes the state object; ABI allocates buffers in `.tessera.meta`.
- Deterministic mode enforces stable reductions and reproducible generation.

---

### E.2 GPT‑OSS‑120B — Generation Loop with Speculative Decoding (Test‑Time Compute)

```python
from tessera import graph, runtime as rt

G_small = graph.load_module("gpt_oss_6b.tsr")   # small draft model
G_large = graph.load_module("gpt_oss_120b.tsr") # verifier model

ctx = rt.create_context(target="cuda_sm90")
rt.load_bundle(ctx, [G_small.bundle, G_large.bundle])

kv_s, kv_l = rt.allocate_state(ctx, G_small), rt.allocate_state(ctx, G_large)

prompt = rt.tensor_from_list([1,2,3,4])
y = []
for step in range(256):
    draft = rt.run(ctx, G_small, {"tokens": prompt}, kv_s, stream=0)["logits"].argmax()
    ok = rt.verify(ctx, G_large, {"tokens": prompt}, draft, kv_l, stream=1)
    if ok:
        y.append(draft); prompt = rt.concat(prompt, draft)
    else:
        corr = rt.refine(ctx, G_large, {"tokens": prompt}, draft, kv_l, stream=1)
        y.append(corr); prompt = rt.concat(prompt, corr)
```

---

### E.3 Hugging Face Transformers → Tessera Import

```python
# hf_import.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from tessera import importer, graph, op

model_id = "gpt2"  # or any HF causal LM
hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="bfloat16")
tok      = AutoTokenizer.from_pretrained(model_id)

# Convert HF modules → Tessera operators
spec = importer.Spec(
    map_attention = importer.flash_attention(head_dim_from_hf=True),
    map_linear    = importer.linear_fuse_bias_act(activation="gelu"),
    numerics      = op.numeric(dtype="bf16", accum="f32"),
    deterministic = True,
)

G = importer.from_huggingface(hf_model, spec)
G = importer.attach_tokenizer(G, tok.vocab)

# Optional: apply TOD rewrites (e.g., FFT-conv for some blocks)
G = graph.apply_rewrites(G, ["fft_conv", "factorized_low_rank"])

bundle = graph.compile(G, targets=["cuda_sm80","cuda_sm90"])  # multi-arch
```

**Importer mapping highlights**

- **Attention**: `HFAttention → op.flash_attention` (with KVCache + causal mask).
- **Linear/MLP**: `nn.Linear → op.linear` (bias/activation fused when possible).
- **LayerNorm/RMSNorm**: maps to `op.layernorm` with policy from Numerics.
- **Embedding/LMHead**: `nn.Embedding/Linear` map to Tessera equivalents.

---

### E.4 Hugging Face — SFT (Supervised Fine‑Tuning) + RL (RLHF) in Tessera

```python
from tessera import train, op, graph

# Assume G is compiled from the HF importer
loss_sft = train.loss.cross_entropy()
reward   = train.loss.rlhf(policy="ppo_clip", ref_model=None, beta=0.01)

@graph.training_step(module=G)
def step(batch):
    logits = graph.forward(G, batch["input_ids"])
    l_sft  = loss_sft(logits, batch["labels"])         # stable CE with log-sum-exp
    r      = reward(logits, batch["preference_labels"]) # PPO/score function path
    l_tot  = op.add(l_sft, op.scale(r, 0.2))
    grads  = graph.backward(l_tot)
    return grads

trainer = train.Trainer(optimizer=train.AdamW(lr=2e-5),
                        numerics=op.numeric(dtype="bf16", accum="f32"),
                        deterministic=True)
trainer.fit(step, dataloader)
```

**Training notes**

- Tessera’s AD rules for operators (TOD) apply automatically during `graph.backward`.
- Deterministic mode fixes reduction order and RNG seeds for reproducibility.
- RLHF reward ca
