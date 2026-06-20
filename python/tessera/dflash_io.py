"""DFlash checkpoint I/O — map HF/safetensors state dicts to DFlashWeights.

The z-lab ``*-DFlash`` checkpoints ship as ``safetensors`` with the usual HF
transformer naming. This module provides:

  * a minimal dependency-free ``safetensors`` reader/writer (numpy),
  * ``dflash_weights_from_state_dict`` / ``_to_state_dict`` that map between the
    HF ``nn.Linear`` ``(out, in)`` convention and DFlashWeights' ``x @ W``
    ``(in, out)`` convention, and
  * ``load_dflash_weights`` / ``save_dflash_weights`` file wrappers.

The embedding and LM head are shared with (and frozen from) the target model, so
they are supplied separately rather than read from the draft checkpoint.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .dflash import DFlashConfig, DFlashLayerWeights, DFlashWeights

# numpy dtype <-> safetensors dtype string
_ST_DTYPE = {"F64": np.float64, "F32": np.float32, "F16": np.float16,
             "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
             "BOOL": np.bool_}
_NP_TO_ST = {np.dtype(v): k for k, v in _ST_DTYPE.items()}
try:  # bf16 is optional
    import ml_dtypes
    _ST_DTYPE["BF16"] = ml_dtypes.bfloat16
    _NP_TO_ST[np.dtype(ml_dtypes.bfloat16)] = "BF16"
except Exception:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Minimal safetensors I/O
# ---------------------------------------------------------------------------

def load_safetensors(path) -> Dict[str, np.ndarray]:
    """Read a ``.safetensors`` file into a ``{name: ndarray}`` dict."""
    data = Path(path).read_bytes()
    if len(data) < 8:
        raise ValueError("safetensors file too small for an 8-byte header length")
    (n,) = struct.unpack("<Q", data[:8])
    if 8 + n > len(data):
        raise ValueError(
            f"safetensors header length {n} exceeds file size {len(data)} — truncated/corrupt"
        )
    header = json.loads(data[8:8 + n])
    base = 8 + n
    out: Dict[str, np.ndarray] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dt = _ST_DTYPE.get(meta["dtype"])
        if dt is None:
            raise ValueError(f"unsupported safetensors dtype {meta['dtype']!r}")
        s, e = meta["data_offsets"]
        if not (0 <= s <= e and base + e <= len(data)):
            raise ValueError(
                f"safetensors tensor {name!r} data_offsets [{s}, {e}] out of bounds "
                f"for file size {len(data)}"
            )
        buf = data[base + s:base + e]
        arr: np.ndarray = np.frombuffer(buf, dtype=dt)
        expected = int(np.prod(meta["shape"])) if meta["shape"] else 1
        if arr.size != expected:
            raise ValueError(
                f"safetensors tensor {name!r}: {arr.size} elements != prod(shape)={expected}"
            )
        out[name] = arr.reshape(meta["shape"]).copy()
    return out


def save_safetensors(path, tensors: Dict[str, np.ndarray]) -> None:
    """Write a ``{name: ndarray}`` dict to a ``.safetensors`` file."""
    header: Dict[str, Any] = {}
    blobs = []
    offset = 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        st = _NP_TO_ST.get(arr.dtype)
        if st is None:
            raise ValueError(f"unsupported numpy dtype {arr.dtype} for {name!r}")
        b = arr.tobytes()
        header[name] = {"dtype": st, "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(b)]}
        blobs.append(b)
        offset += len(b)
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)


# ---------------------------------------------------------------------------
# State-dict <-> DFlashWeights
# ---------------------------------------------------------------------------

def _t(a):
    return np.ascontiguousarray(np.asarray(a).T)


def dflash_weights_from_state_dict(sd: Dict[str, np.ndarray], cfg: DFlashConfig,
                                   *, embed_tokens, lm_head=None,
                                   prefix: str = "model.") -> DFlashWeights:
    """Build DFlashWeights from an HF/safetensors state dict.

    HF ``nn.Linear`` weights are ``(out, in)`` and applied as ``x @ W.T``; the
    DFlash reference applies ``x @ W`` with ``(in, out)`` weights, so the 2-D
    projection weights are transposed. RMSNorm ``.weight`` vectors pass through.
    ``embed_tokens`` / ``lm_head`` are the (frozen, shared) target tensors.
    """
    def g(name):
        for p in (prefix, ""):
            if p + name in sd:
                return sd[p + name]
        raise KeyError(f"missing state-dict key {prefix + name!r} (or unprefixed)")

    layers = []
    for i in range(cfg.num_hidden_layers):
        b = f"layers.{i}."
        layers.append(DFlashLayerWeights(
            q_proj=_t(g(b + "self_attn.q_proj.weight")),
            k_proj=_t(g(b + "self_attn.k_proj.weight")),
            v_proj=_t(g(b + "self_attn.v_proj.weight")),
            o_proj=_t(g(b + "self_attn.o_proj.weight")),
            q_norm=np.asarray(g(b + "self_attn.q_norm.weight")),
            k_norm=np.asarray(g(b + "self_attn.k_norm.weight")),
            input_layernorm=np.asarray(g(b + "input_layernorm.weight")),
            post_attention_layernorm=np.asarray(g(b + "post_attention_layernorm.weight")),
            mlp_gate=_t(g(b + "mlp.gate_proj.weight")),
            mlp_up=_t(g(b + "mlp.up_proj.weight")),
            mlp_down=_t(g(b + "mlp.down_proj.weight")),
        ))
    return DFlashWeights(
        embed_tokens=np.asarray(embed_tokens),
        fc=_t(g("fc.weight")),
        hidden_norm=np.asarray(g("hidden_norm.weight")),
        layers=layers,
        final_norm=np.asarray(g("norm.weight")),
        lm_head=None if lm_head is None else np.asarray(lm_head),
    )


def dflash_weights_to_state_dict(w: DFlashWeights, *, prefix: str = "model.") -> Dict[str, np.ndarray]:
    """Inverse of :func:`dflash_weights_from_state_dict` (excludes the shared
    embedding/LM head). Useful for saving a trained draft in HF layout."""
    sd: Dict[str, np.ndarray] = {
        prefix + "fc.weight": _t(w.fc),
        prefix + "hidden_norm.weight": np.asarray(w.hidden_norm),
        prefix + "norm.weight": np.asarray(w.final_norm),
    }
    for i, lw in enumerate(w.layers):
        b = f"{prefix}layers.{i}."
        sd[b + "self_attn.q_proj.weight"] = _t(lw.q_proj)
        sd[b + "self_attn.k_proj.weight"] = _t(lw.k_proj)
        sd[b + "self_attn.v_proj.weight"] = _t(lw.v_proj)
        sd[b + "self_attn.o_proj.weight"] = _t(lw.o_proj)
        sd[b + "self_attn.q_norm.weight"] = np.asarray(lw.q_norm)
        sd[b + "self_attn.k_norm.weight"] = np.asarray(lw.k_norm)
        sd[b + "input_layernorm.weight"] = np.asarray(lw.input_layernorm)
        sd[b + "post_attention_layernorm.weight"] = np.asarray(lw.post_attention_layernorm)
        sd[b + "mlp.gate_proj.weight"] = _t(lw.mlp_gate)
        sd[b + "mlp.up_proj.weight"] = _t(lw.mlp_up)
        sd[b + "mlp.down_proj.weight"] = _t(lw.mlp_down)
    return sd


def load_dflash_weights(path, cfg: DFlashConfig, *, embed_tokens, lm_head=None,
                        prefix: str = "model.") -> DFlashWeights:
    """Load a ``z-lab/*-DFlash`` ``.safetensors`` draft checkpoint into
    DFlashWeights (embedding + LM head supplied from the target)."""
    return dflash_weights_from_state_dict(load_safetensors(path), cfg,
                                          embed_tokens=embed_tokens, lm_head=lm_head,
                                          prefix=prefix)


def save_dflash_weights(path, w: DFlashWeights, *, prefix: str = "model.") -> None:
    """Save a (trained) DFlash draft to ``.safetensors`` in HF layout."""
    save_safetensors(path, dflash_weights_to_state_dict(w, prefix=prefix))


__all__ = [
    "load_safetensors", "save_safetensors",
    "dflash_weights_from_state_dict", "dflash_weights_to_state_dict",
    "load_dflash_weights", "save_dflash_weights",
]
