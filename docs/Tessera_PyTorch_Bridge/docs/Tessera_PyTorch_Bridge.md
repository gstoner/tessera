# Tessera PyTorch Conversion & Compatibility Guide

## Overview

Tessera provides a **conversion and compatibility bridge** for PyTorch models, allowing developers to:

- **Auto-convert PyTorch models** (via `torch.export` / FX) into Tessera’s Graph IR.
- **Fallback to PyTorch ops** using compatibility islands, with zero-copy tensor exchange.
- **Progressively migrate workloads**: start with hybrid execution, move toward full Tessera execution.

This enables smooth adoption of Tessera without requiring complete rewrites.

---

## Conversion Workflow

### Graph Capture

1. Use `torch.export(model, args)` for stable, functional graphs (recommended).
2. Fall back to `torch.fx.symbolic_trace` for dynamic control flow.

Captured graphs are lowered to **Tessera Graph IR** and then scheduled (Graph → Schedule → Tile → Target IR).

---

## User API

```python
import torch
import tessera as ts

# A. Auto-convert a model
tessera_model = ts.from_pytorch(pytorch_model, example_inputs=(x0,), strict=False)

# B. Wrap PyTorch modules for hybrid execution
wrapped = ts.compat.wrap(pytorch_model, allow_fallback=True)

# C. Register custom op mappings
@ts.compat.register_aten("aten.my_custom_op")
def lower_my_custom_op(node, ctx):
    return ts.ops.my_custom_kernel(*ctx.get_inputs(node), **node.kwargs)

# D. Zero-copy tensor exchange
t_x   = ts.from_dlpack(torch.utils.dlpack.to_dlpack(x))
x2    = torch.utils.dlpack.from_dlpack(ts.to_dlpack(t_x))
```

---

## Operator Mapping

Tessera maintains a mapping table for ATen ops:

| PyTorch Op                | Tessera Equivalent            |
|---------------------------|-------------------------------|
| `aten.mm`, `bmm`, `addmm` | `ts.ops.matmul`               |
| `aten.conv*`              | `ts.nn.conv{1,2,3}d`          |
| `aten.layer_norm`         | `ts.nn.layer_norm`            |
| `aten.softmax`            | `ts.nn.softmax` / fused attn  |
| `aten.dropout`            | `ts.nn.dropout`               |
| `aten._scaled_dot_product_attention` | `ts.ops.flash_attention` |
| Elementwise ops (`add`, `mul`, etc.) | `ts.ops.*`             |

---

## Compatibility Islands

Tessera allows embedding **PyTorch islands** inside Tessera models:

```python
class CompatModule(ts.Module):
    def __init__(self, torch_mod):
        super().__init__()
        self.torch_mod = torch_mod.eval()

    def forward(self, *tensors):
        torch_args = [torch.utils.dlpack.from_dlpack(ts.to_dlpack(t)) for t in tensors]
        out = self.torch_mod(*torch_args)
        def wrap(o):
            return ts.from_dlpack(torch.utils.dlpack.to_dlpack(o)) if torch.is_tensor(o) else o
        if isinstance(out, (tuple, list)):
            return type(out)(wrap(o) for o in out)
        return wrap(out)
```

- **Forward-only support** for inference.
- For training, backward passes can be delegated to PyTorch.

---

## Training Migration

- **State dicts**: `torch_model.state_dict()` → Tessera parameter store.
- **Optimizers**: Map AdamW/SGD → `ts.opt.adamw/sgd` with copied momentum states.
- **Autodiff**: Capture backward graphs with `capture_backward=True`.

---

## Distributed Migration

- Start with PyTorch **DDP** outside, Tessera inside.
- Upgrade to Tessera’s `dist.Mesh` for tensor/pipeline/data parallelism.
- Provide shims for checkpoint/load interop.

---

## Best Practices

1. **Start hybrid**: allow fallbacks (`strict=False`), measure hotspots.
2. **Promote critical ops**: replace attention, MLP, and normalization first.
3. **Enable autotuning**: cache best schedule per shape/arch.
4. **Tighten**: switch to `strict=True` once coverage is sufficient.

---

## Limitations

- Dynamic shapes/control flow require PyTorch fallback or annotations.
- Custom CUDA extensions remain in PyTorch until explicitly ported.
- Crossing boundaries too often can reduce performance—use partitioner to coalesce fallback regions.

---

## Example: Mini-Transformer Block

```python
import torch, torch.nn as nn
import tessera as ts

class Block(nn.Module):
    def __init__(self, d, nhead, mlp=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, mlp*d), nn.GELU(), nn.Linear(mlp*d, d)
        )
    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        a,_ = self.attn(h,h,h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        h = self.ln2(x)
        x = x + self.ffn(h)
        return x

pt = Block(512, 8).cuda().eval()
x  = torch.randn(8, 1024, 512, device="cuda")

tess = ts.from_pytorch(pt, example_inputs=(x,), strict=False)
tess.attn = ts.compat.wrap(pt.attn, allow_fallback=True)

tx = ts.from_dlpack(torch.utils.dlpack.to_dlpack(x))
ty = tess(tx)
y  = torch.utils.dlpack.from_dlpack(ts.to_dlpack(ty))
```

---

## Summary

The Tessera PyTorch bridge provides:
- **Automatic model conversion** (`from_pytorch`)
- **Compatibility islands** for fallback
- **Interop via DLPack** for zero-copy exchange
- **Gradual migration path** from hybrid execution to full Tessera
