"""
tessera_gemma/peft/lora.py — LoRA and QLoRA adapters for Tessera-Gemma.

Implements:
  • LoRAAdapter       — core low-rank Δ = B·A·x·scaling
  • LoRALinear        — nn.Linear wrapper with multiple named adapters
  • apply_lora        — apply by substring pattern (simple API)
  • apply_lora_regex  — apply by full-regex rule dicts (flexible API)
  • QLinearSim        — simulated Int4 (Q4_0) quantisation
  • apply_qlora_sim   — wrap target layers with QLinearSim
  • freeze_by_regex   — freeze base params matching a pattern list
  • param_groups_with_adapter_lrmult — per-adapter learning-rate groups
  • lora_state_dict / load_lora_state_dict — checkpoint helpers
  • merge_lora / unmerge_lora — bake / unbake adapters into weights
"""

from __future__ import annotations

import re
import torch
import torch.nn as nn
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Core adapter module
# ---------------------------------------------------------------------------

class LoRAAdapter(nn.Module):
    """Single LoRA low-rank adapter: Δ = dropout(x) @ A @ B * (alpha/rank).

    Weights are initialised as:
      A: kaiming_uniform (same as nn.Linear default)
      B: zeros  →  adapter starts as identity (Δ = 0)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float,
        name: str,
    ) -> None:
        super().__init__()
        self.name = name
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.rank)

        self.lora_A = nn.Parameter(torch.empty(in_features, self.rank))
        self.lora_B = nn.Parameter(torch.zeros(self.rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros(*x.shape[:-1], self.lora_B.shape[1],
                               device=x.device, dtype=x.dtype)
        return (self.dropout(x) @ self.lora_A) @ self.lora_B * self.scaling

    def extra_repr(self) -> str:
        return f"name={self.name}, rank={self.rank}, alpha={self.alpha}"


# ---------------------------------------------------------------------------
# nn.Linear wrapper
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """nn.Linear wrapped with support for multiple named LoRA adapters.

    Forward: y = base(x) + Σ_{enabled adapters} adapter(x)
    """

    def __init__(self, base: nn.Linear) -> None:
        super().__init__()
        self.base = base
        self.adapters: Dict[str, LoRAAdapter] = {}
        self.merge_order: List[str] = []
        self.merged = False

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    # --- adapter management ------------------------------------------------

    def add_adapter(
        self,
        name: str,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> LoRAAdapter:
        if name in self.adapters:
            raise ValueError(f"Adapter '{name}' already exists on this layer")
        ad = LoRAAdapter(
            self.base.in_features, self.base.out_features,
            rank=rank, alpha=alpha, dropout=dropout, name=name,
        )
        ad.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        self.adapters[name] = ad
        self.merge_order.append(name)
        return ad

    def enable_adapter(self, name: str, enabled: bool = True) -> None:
        self.adapters[name].enabled = enabled

    def set_merge_order(self, names: List[str]) -> None:
        for n in names:
            if n not in self.adapters:
                raise ValueError(f"Unknown adapter '{n}'")
        self.merge_order = list(names)

    # --- forward -----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if not self.merged:
            for name in self.merge_order:
                ad = self.adapters[name]
                if ad.enabled:
                    y = y + ad(x)
        return y

    # --- merge / unmerge ---------------------------------------------------

    def merge(self) -> int:
        """Bake all enabled adapters into base weights. Returns count merged."""
        if self.merged:
            return 0
        count = 0
        with torch.no_grad():
            for name in self.merge_order:
                ad = self.adapters[name]
                if ad.enabled:
                    delta = (ad.lora_A @ ad.lora_B).T * ad.scaling  # (out, in)
                    self.base.weight.add_(delta)
                    count += 1
        self.merged = True
        return count

    def unmerge(self) -> int:
        """Subtract adapter deltas from base weights. Returns count unmerged."""
        if not self.merged:
            return 0
        count = 0
        with torch.no_grad():
            for name in reversed(self.merge_order):
                ad = self.adapters[name]
                if ad.enabled:
                    delta = (ad.lora_A @ ad.lora_B).T * ad.scaling
                    self.base.weight.sub_(delta)
                    count += 1
        self.merged = False
        return count


# ---------------------------------------------------------------------------
# Application helpers
# ---------------------------------------------------------------------------

def _ensure_lora_linear(
    parent: nn.Module, attr_name: str
) -> Optional[LoRALinear]:
    """Wrap `parent.attr_name` in LoRALinear if it's an nn.Linear."""
    child = getattr(parent, attr_name)
    if isinstance(child, LoRALinear):
        return child
    if isinstance(child, nn.Linear):
        ll = LoRALinear(child)
        setattr(parent, attr_name, ll)
        return ll
    return None


def apply_lora_regex(
    model: nn.Module,
    rules: Iterable[Dict[str, Any]],
) -> int:
    """Apply LoRA adapters to modules matching regex rules.

    Args:
        model: The model to patch.
        rules: Iterable of dicts with keys:
               - "pattern": regex matched against the full module path
               - "name":    adapter name (default "lora")
               - "rank":    LoRA rank (default 8)
               - "alpha":   LoRA alpha (default 16.0)
               - "dropout": dropout on A projection (default 0.0)

    Returns:
        Number of adapter instances created.
    """
    created = 0
    named_mods = dict(model.named_modules())
    for fq_name, mod in list(named_mods.items()):
        parent_name = fq_name.rsplit(".", 1)[0] if "." in fq_name else ""
        leaf_name   = fq_name.split(".")[-1]
        parent = named_mods.get(parent_name, None) if parent_name else model

        for rule in rules:
            if not re.fullmatch(rule["pattern"], fq_name):
                continue
            if not isinstance(mod, (nn.Linear, LoRALinear)):
                continue
            ll = _ensure_lora_linear(parent, leaf_name)
            if ll is None:
                continue
            ll.add_adapter(
                name=rule.get("name", "lora"),
                rank=int(rule.get("rank", 8)),
                alpha=float(rule.get("alpha", 16.0)),
                dropout=float(rule.get("dropout", 0.0)),
            )
            created += 1
    return created


def apply_lora(
    model: nn.Module,
    *,
    patterns: Iterable[str] = ("q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"),
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> int:
    """Convenience wrapper: apply LoRA to layers whose path contains any
    of the given substrings."""
    rules = [
        {
            "pattern": f".*{re.escape(p)}",
            "name": p,
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
        }
        for p in patterns
    ]
    return apply_lora_regex(model, rules)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only LoRA adapter tensors — use for checkpoint saving."""
    out: Dict[str, torch.Tensor] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            for aname, ad in mod.adapters.items():
                out[f"{name}.adapters.{aname}.A"] = ad.lora_A.detach().clone()
                out[f"{name}.adapters.{aname}.B"] = ad.lora_B.detach().clone()
    return out


def load_lora_state_dict(
    model: nn.Module,
    state: Dict[str, torch.Tensor],
) -> None:
    """Restore LoRA adapter tensors from a saved state dict."""
    for name, mod in model.named_modules():
        if not isinstance(mod, LoRALinear):
            continue
        prefix = f"{name}.adapters."
        for k, v in list(state.items()):
            if not k.startswith(prefix):
                continue
            parts = k[len(prefix):].split(".")
            if len(parts) < 2:
                continue
            adname, mat = parts[0], parts[1]
            if adname not in mod.adapters:
                rank = v.shape[1] if mat == "A" else v.shape[0]
                mod.add_adapter(adname, rank=rank, alpha=rank * 2.0)
        for adname, ad in mod.adapters.items():
            a_key = f"{name}.adapters.{adname}.A"
            b_key = f"{name}.adapters.{adname}.B"
            if a_key in state:
                with torch.no_grad():
                    ad.lora_A.copy_(
                        state[a_key].to(ad.lora_A.device).to(ad.lora_A.dtype)
                    )
            if b_key in state:
                with torch.no_grad():
                    ad.lora_B.copy_(
                        state[b_key].to(ad.lora_B.device).to(ad.lora_B.dtype)
                    )


def merge_lora(model: nn.Module) -> int:
    """Merge all LoRA adapters into base weights. Returns total count."""
    return sum(
        m.merge()
        for m in model.modules()
        if isinstance(m, LoRALinear)
    )


def unmerge_lora(model: nn.Module) -> int:
    """Unmerge all LoRA adapters from base weights. Returns total count."""
    return sum(
        m.unmerge()
        for m in model.modules()
        if isinstance(m, LoRALinear)
    )


# ---------------------------------------------------------------------------
# QLoRA — simulated Int4 quantisation
# ---------------------------------------------------------------------------

def _quantize_int4(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Q4_0: per-output-channel symmetric Int4 in an Int8 container."""
    w = weight.detach().float()
    max_abs = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scale = max_abs / 7.0
    q = torch.clamp(torch.round(w / scale), -8, 7).to(torch.int8)
    return q, scale.squeeze(1)


def _dequantize_int4(
    q: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return q.float() * scale.unsqueeze(1)


class QLinearSim(nn.Module):
    """Simulates an Int4-quantised Linear layer in float forward()."""

    def __init__(self, base: nn.Linear) -> None:
        super().__init__()
        self.base = base
        self.enabled = True
        q, s = _quantize_int4(base.weight.data)
        self.register_buffer("qweight", q, persistent=False)
        self.register_buffer("scale",   s, persistent=False)

    def reinit_quant(self) -> None:
        q, s = _quantize_int4(self.base.weight.data)
        self.qweight.copy_(q)
        self.scale.copy_(s)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return self.base(x)
        w = _dequantize_int4(self.qweight, self.scale).to(x.dtype).to(x.device)
        return torch.nn.functional.linear(x, w, self.base.bias)


def apply_qlora_sim(
    model: nn.Module,
    patterns: Iterable[str] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ),
) -> int:
    """Wrap matching Linear / LoRALinear.base with QLinearSim. Returns count."""
    count = 0
    named_mods = dict(model.named_modules())
    for name, mod in list(named_mods.items()):
        if not any(p in name for p in patterns):
            continue
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        leaf_name   = name.split(".")[-1]
        parent = named_mods.get(parent_name, None) if parent_name else model
        if parent is None:
            continue
        if isinstance(mod, LoRALinear):
            mod.base = QLinearSim(mod.base)
            count += 1
        elif isinstance(mod, nn.Linear):
            setattr(parent, leaf_name, QLinearSim(mod))
            count += 1
    return count


# ---------------------------------------------------------------------------
# Freeze / LR group helpers
# ---------------------------------------------------------------------------

def freeze_by_regex(
    model: nn.Module,
    patterns: Iterable[str] = ("embed_tokens", "norm"),
) -> int:
    """Freeze parameters whose module path contains any of the patterns."""
    pats = tuple(patterns)
    count = 0
    for name, p in model.named_parameters():
        if any(s in name for s in pats):
            p.requires_grad_(False)
            count += 1
    return count


def param_groups_with_adapter_lrmult(
    model: nn.Module,
    base_lr: float,
    adapter_lr_mult: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """Build optimizer param groups with per-adapter LR multipliers.

    Args:
        model:            The model.
        base_lr:          Learning rate for non-adapter parameters.
        adapter_lr_mult:  Dict mapping adapter name → LR multiplier.
                          Defaults to 1.0 for unknown adapter names.

    Returns:
        List of param-group dicts suitable for an optimizer constructor.
    """
    adapter_lr_mult = adapter_lr_mult or {}
    visited: set = set()
    groups: List[Dict] = []

    for _, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            for ad_name, ad in mod.adapters.items():
                mult = float(adapter_lr_mult.get(ad_name, 1.0))
                params = [p for p in (ad.lora_A, ad.lora_B)
                          if p.requires_grad and id(p) not in visited]
                if params:
                    groups.append({"params": params, "lr": base_lr * mult})
                    visited.update(id(p) for p in params)

    base_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in visited
    ]
    if base_params:
        groups.append({"params": base_params, "lr": base_lr})

    return groups
