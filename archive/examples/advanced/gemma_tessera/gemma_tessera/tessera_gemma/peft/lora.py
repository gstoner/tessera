
import torch
import torch.nn as nn
import re
from typing import Optional, Dict, Any, Iterable, Tuple, List

# ----------------- LoRA core with multi-adapter support -----------------

class LoRAAdapter(nn.Module):
    def __init__(self, in_f: int, out_f: int, rank: int, alpha: float, dropout: float, name: str):
        super().__init__()
        self.name = name
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.dropout_p = float(dropout)
        self.lora_A = nn.Parameter(torch.zeros(in_f, self.rank))
        self.lora_B = nn.Parameter(torch.zeros(self.rank, out_f))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        self.scaling = self.alpha / max(1, self.rank)
        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_d = self.dropout(x)
        return (x_d @ self.lora_A) @ self.lora_B * self.scaling

class LoRALinear(nn.Module):
    """
    nn.Linear wrapper with support for multiple named LoRA adapters.
    W_eff(x) = base(x) + sum_{a enabled in merge_order} adapter_a(x)
    """
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.adapters: Dict[str, LoRAAdapter] = {}
        self.merge_order: List[str] = []   # order to merge when 'merge()' is called
        self.merged = False

    @property
    def in_features(self): return self.base.in_features
    @property
    def out_features(self): return self.base.out_features

    def add_adapter(self, name: str, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        if name in self.adapters:
            raise ValueError(f"Adapter '{name}' already exists")
        ad = LoRAAdapter(self.base.in_features, self.base.out_features, rank, alpha, dropout, name)
        # Move to same device/dtype
        ad.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        self.adapters[name] = ad
        self.merge_order.append(name)
        return ad

    def enable_adapter(self, name: str, enabled: bool = True):
        self.adapters[name].enabled = enabled

    def set_merge_order(self, names: List[str]):
        for n in names:
            if n not in self.adapters:
                raise ValueError(f"Unknown adapter '{n}'")
        self.merge_order = list(names)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x) if not self.merged else self.base(x)
        if not self.merged:
            for name in self.merge_order:
                ad = self.adapters[name]
                if ad.enabled:
                    y = y + ad(x)
        return y

    def merge(self) -> int:
        if self.merged: return 0
        count = 0
        with torch.no_grad():
            W = self.base.weight
            for name in self.merge_order:
                ad = self.adapters[name]
                if ad.enabled:
                    W.add_((ad.lora_A @ ad.lora_B).T * ad.scaling)
                    count += 1
        self.merged = True
        return count

    def unmerge(self) -> int:
        if not self.merged: return 0
        count = 0
        with torch.no_grad():
            W = self.base.weight
            for name in reversed(self.merge_order):
                ad = self.adapters[name]
                if ad.enabled:
                    W.sub_((ad.lora_A @ ad.lora_B).T * ad.scaling)
                    count += 1
        self.merged = False
        return count

# ----------------- Regex targeting and per-module ranks -----------------

def _ensure_lora_linear(module: nn.Module, attr_name: str) -> LoRALinear:
    child = getattr(module, attr_name)
    if isinstance(child, LoRALinear):
        return child
    elif isinstance(child, nn.Linear):
        ll = LoRALinear(child)
        setattr(module, attr_name, ll)
        return ll
    else:
        return None  # not linear

def apply_lora_regex(
    model: nn.Module,
    rules: Iterable[Dict[str, Any]],
) -> int:
    """
    rules: list of { 'pattern': r'.*qkv.*', 'name': 'qkv', 'rank': 8, 'alpha': 16.0, 'dropout': 0.0 }
    Applies adapters to Linear modules whose full-qualified name matches regex 'pattern'.
    If multiple rules match a module, all adapters are added (compose).
    Returns number of adapter instances created.
    """
    created = 0
    modules = dict(model.named_modules())
    for fq_name, mod in list(modules.items()):
        parent_name = fq_name.rsplit('.', 1)[0] if '.' in fq_name else ''
        leaf_name = fq_name.split('.')[-1]
        parent = modules.get(parent_name, None) if parent_name else model

        for rule in rules:
            if re.fullmatch(rule['pattern'], fq_name):
                if isinstance(mod, nn.Linear) or isinstance(mod, LoRALinear):
                    ll = _ensure_lora_linear(parent, leaf_name)
                    name = rule.get('name', 'lora')
                    rank = int(rule.get('rank', 8))
                    alpha = float(rule.get('alpha', 16.0))
                    dropout = float(rule.get('dropout', 0.0))
                    ll.add_adapter(name=name, rank=rank, alpha=alpha, dropout=dropout)
                    created += 1
    return created

# Back-compat: simple apply by substr patterns
def apply_lora(model: nn.Module, *, patterns: Optional[Iterable[str]] = ("qkv","proj","wi","wo","lm_head"), rank: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> int:
    rules = [{'pattern': fr'.*{re.escape(p)}.*', 'name': p, 'rank': rank, 'alpha': alpha, 'dropout': dropout} for p in patterns]
    return apply_lora_regex(model, rules)

def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            for aname, ad in mod.adapters.items():
                out[f"{name}.adapters.{aname}.A"] = ad.lora_A.detach().clone()
                out[f"{name}.adapters.{aname}.B"] = ad.lora_B.detach().clone()
    return out

def load_lora_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            # find all keys for this module
            prefix = f"{name}.adapters."
            for k, v in list(state.items()):
                if k.startswith(prefix) and (k.endswith(".A") or k.endswith(".B")):
                    # k = "{name}.adapters.{adname}.A"
                    parts = k.split(".")
                    adname = parts[-2]
                    if adname not in mod.adapters:
                        # create adapter with default rank deduced from tensor shape
                        rank = v.shape[0] if k.endswith(".B") else v.shape[1]
                        mod.add_adapter(adname, rank=rank, alpha=rank*2.0, dropout=0.0)
            # now copy tensors
            for adname, ad in mod.adapters.items():
                akey = f"{name}.adapters.{adname}.A"
                bkey = f"{name}.adapters.{adname}.B"
                if akey in state and bkey in state:
                    with torch.no_grad():
                        ad.lora_A.copy_(state[akey].to(ad.lora_A.device).to(ad.lora_A.dtype))
                        ad.lora_B.copy_(state[bkey].to(ad.lora_B.device).to(ad.lora_B.dtype))

def merge_lora(model: nn.Module) -> int:
    c = 0
    for m in model.modules():
        if isinstance(m, LoRALinear):
            c += m.merge()
    return c

def unmerge_lora(model: nn.Module) -> int:
    c = 0
    for m in model.modules():
        if isinstance(m, LoRALinear):
            c += m.unmerge()
    return c

# ----------------- QLoRA / Int4 simulation (Q4_0 per-out-channel) -----------------

def quantize_int4_per_out_channel(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate Q4_0 with per-output-channel scales.
    Returns (qint8 in [-8,7], scales float32 with shape [out,1]).
    """
    # weight: (out, in)
    out, inn = weight.shape
    w = weight.detach().to(torch.float32)
    max_abs = w.abs().amax(dim=1, keepdim=True) + 1e-12
    scale = max_abs / 7.0
    q = torch.clamp(torch.round(w / scale), -8, 7).to(torch.int8)
    return q, scale.squeeze(1)  # scale per out

def dequant_int4_per_out_channel(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # q: (out, in) int8 with values in [-8,7], scale: (out,)
    return (q.to(torch.float32) * scale.unsqueeze(1))

class QLinearSim(nn.Module):
    """Simulate an Int4 (Q4_0) quantized Linear layer in forward()."""
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.enabled = True
        # buffers to store quantized form
        q, s = quantize_int4_per_out_channel(self.base.weight.data)
        self.register_buffer("qweight", q, persistent=False)
        self.register_buffer("scale", s, persistent=False)

    def reinit_quant(self):
        q, s = quantize_int4_per_out_channel(self.base.weight.data)
        self.qweight.copy_(q); self.scale.copy_(s)

    def set_enabled(self, enabled: bool = True):
        self.enabled = enabled

    @property
    def in_features(self): return self.base.in_features
    @property
    def out_features(self): return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return self.base(x)
        w = dequant_int4_per_out_channel(self.qweight, self.scale).to(x.dtype).to(x.device)
        y = torch.nn.functional.linear(x, w, bias=self.base.bias)
        return y

def apply_qlora_sim(model: nn.Module, patterns: Iterable[str] = ("qkv","proj","wi","wo")) -> int:
    """
    Wrap target Linear or LoRALinear.base with QLinearSim to simulate Int4. Returns number of wrapped layers.
    """
    count = 0
    for name, mod in list(model.named_modules()):
        parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
        leaf_name = name.split('.')[-1]
        parent = dict(model.named_modules()).get(parent_name, None) if parent_name else model
        if isinstance(mod, LoRALinear):
            if any(p in name for p in patterns):
                setattr(mod, "base", QLinearSim(mod.base))
                count += 1
        elif isinstance(mod, nn.Linear):
            if any(p in name for p in patterns):
                setattr(parent, leaf_name, QLinearSim(mod))
                count += 1
    return count


# ----------------- Freezing policies & LR multipliers -----------------

def freeze_by_regex(model: nn.Module, patterns=('embed','norm')) -> int:
    \"\"\"Freeze parameters whose module path matches any of the substrings in patterns.
    Returns count of frozen tensors.\"\"\"
    pats = tuple(patterns)
    count = 0
    for name, p in model.named_parameters():
        if any(s in name for s in pats):
            p.requires_grad = False
            count += 1
    return count

def param_groups_with_adapter_lrmult(model: nn.Module, base_lr: float, adapter_lr_mult: Dict[str, float] | None = None):
    \"\"\"Create optimizer param groups where LoRA adapter params receive per-adapter LR multipliers.
    - Base (non-LoRA) params get lr=base_lr (only if requires_grad).
    - For each LoRALinear.adapters[name], its parameters get lr=base_lr * adapter_lr_mult.get(name, 1.0).
    \"\"\"
    adapter_lr_mult = adapter_lr_mult or {}
    base_params = []
    groups = []
    visited = set()

    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            for ad_name, ad in mod.adapters.items():
                mult = float(adapter_lr_mult.get(ad_name, 1.0))
                if ad.lora_A.requires_grad or ad.lora_B.requires_grad:
                    groups.append({'params': [ad.lora_A, ad.lora_B], 'lr': base_lr * mult})
                    visited.add(ad.lora_A); visited.add(ad.lora_B)

    for p in model.parameters():
        if p.requires_grad and p not in visited:
            base_params.append(p)
    if base_params:
        groups.append({'params': base_params, 'lr': base_lr})
    return groups
