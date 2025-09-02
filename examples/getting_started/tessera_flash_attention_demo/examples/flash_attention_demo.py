
#!/usr/bin/env python3
"""
Tessera Flash Attention Demo

This example demonstrates how to use Tessera's Flash Attention implementation
for memory-efficient attention computation. It also provides a PyTorch reference
path for validation and timing.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import numpy as np

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# Optional Tessera import (graceful fallback)
try:
    import tessera as tsr
    TESSERA_AVAILABLE = True
except Exception as exc:
    tsr = None  # type: ignore
    TESSERA_AVAILABLE = False
    _import_error = str(exc)

from utils.attention_ref import sdpa_reference

# ---- IR dump utility (best-effort across possible APIs) --------------------
DUMP_SPEC = [
    ("graph",   ["debug.graph_ir", "debug.dump_graph_ir", "core.dump_graph_ir"],   "graph_ir.mlir"),
    ("schedule",["debug.schedule_ir", "debug.dump_schedule_ir"],                   "schedule_ir.mlir"),
    ("tile",    ["debug.tile_ir", "debug.dump_tile_ir"],                           "tile_ir.mlir"),
    ("target",  ["debug.target_ir", "debug.dump_target_ir", "core.dump_target_ir"],"target_ir.mlir"),
]

def _resolve_attr_chain(root, chain):
    obj = root
    for part in chain.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj

def dump_all_irs(func_obj, sample_args, sample_kwargs):
    summary = {"tessera_available": TESSERA_AVAILABLE, "stages": []}
    # trigger compilation once if possible
    try:
        _ = func_obj(*sample_args, **sample_kwargs)
    except Exception as e:
        summary["compile_call_error"] = repr(e)

    if not TESSERA_AVAILABLE:
        for (_, _, fname) in DUMP_SPEC:
            (ARTIFACTS / fname).write_text("// Tessera not available — placeholder\n")
        summary["note"] = "Tessera not available; wrote placeholder IR files."
        (ARTIFACTS / "compilation_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    for stage, candidates, fname in DUMP_SPEC:
        out_path = ARTIFACTS / fname
        dumped = False
        last_err = None
        for cand in candidates:
            try:
                cb = _resolve_attr_chain(tsr, cand)  # type: ignore
                if cb is None:
                    continue
                try:
                    text = cb(func_obj)  # type: ignore
                except TypeError:
                    try:
                        text = cb(func_obj, *sample_args, **sample_kwargs)  # type: ignore
                    except TypeError:
                        text = cb(compiled=func_obj)  # type: ignore
                if not isinstance(text, str):
                    text = getattr(text, "text", str(text))
                out_path.write_text(text if text else f"// {stage}: empty output\n")
                dumped = True
                break
            except Exception as e:
                last_err = e
                continue
        summary["stages"].append({
            "stage": stage,
            "file": str(out_path),
            "ok": dumped,
            "error": repr(last_err) if not dumped and last_err else None,
        })
        if not dumped:
            out_path.write_text(f"// Could not dump {stage} IR (no matching debug hooks)\n")
    (ARTIFACTS / "compilation_summary.json").write_text(json.dumps(summary, indent=2))
    return summary

# ---- Tessera wrapper (best-effort) ----------------------------------------
if TESSERA_AVAILABLE:
    @tsr.function  # type: ignore
    def flash_attn_tessera(q, k, v, causal: bool = False):
        # Expect Tessera tensors or adapters; if given torch, user should wrap beforehand.
        # Using hypothetical API tsr.nn.flash_attention; adjust if your build differs.
        return tsr.nn.flash_attention(q, k, v, causal=causal)  # type: ignore
else:
    def flash_attn_tessera(*args, **kwargs):
        raise RuntimeError("Tessera not available; flash_attn_tessera cannot run.")

def pick_dtype(name: str):
    name = name.lower()
    if name in ("f32", "float32", "fp32"):
        return torch.float32
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("f16", "float16", "fp16", "half"):
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--dtype", type=str, default="bf16", help="f32|bf16|f16")
    ap.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu")
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--dump-ir", action="store_true")
    ap.add_argument("--iters", type=int, default=10, help="timed iterations")
    args = ap.parse_args()

    print("🚀 Tessera Flash Attention Demo")
    print("=" * 50)

    B, H, S, D = args.batch, args.heads, args.seq, args.dim
    dtype = pick_dtype(args.dtype)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("Configuration:")
    print(f"  Batch size: {B}")
    print(f"  Heads: {H}")
    print(f"  Sequence length: {S}")
    print(f"  Head dimension: {D}")
    print(f"  Device: {device}")
    print(f"  DType: {dtype}  (requested: {args.dtype})")
    print(f"  Causal: {args.causal}")
    print(f"  Tessera available: {TESSERA_AVAILABLE}")

    # Create QKV
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)

    print(f"\n✅ Created tensors: Q{list(q.shape)}, K{list(k.shape)}, V{list(v.shape)}")

    # Reference path (always available)
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    out_ref = sdpa_reference(q, k, v, causal=args.causal)
    torch.cuda.synchronize() if device == "cuda" else None
    ref_time = (time.time() - t0) * 1000.0
    print(f"🧪 Reference SDPA: {ref_time:.2f} ms (1 run)")

    timings = {"ref_ms_per_iter": None, "tessera_ms_per_iter": None}

    # Warmup + timing loops
    iters = max(1, args.iters)
    if device == "cuda":
        torch.cuda.synchronize()

    # Reference timing
    t_acc = 0.0
    for _ in range(iters):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        _ = sdpa_reference(q, k, v, causal=args.causal)
        if device == "cuda":
            torch.cuda.synchronize()
        t_acc += (time.time() - t0) * 1000.0
    timings["ref_ms_per_iter"] = t_acc / iters

    # Tessera timing (if available)
    if TESSERA_AVAILABLE:
        # Convert to Tessera tensors if needed (best-effort)
        try:
            tq = tsr.tensor(q, shape=["B","H","S","D"])  # type: ignore
            tk = tsr.tensor(k, shape=["B","H","S","D"])  # type: ignore
            tv = tsr.tensor(v, shape=["B","H","S","D"])  # type: ignore
        except Exception:
            # Assume q/k/v might already be compatible; pass through
            tq, tk, tv = q, k, v

        # Warmup
        try:
            _ = flash_attn_tessera(tq, tk, tv, args.causal)
        except Exception as e:
            print("⚠️ Tessera call failed on warmup:", repr(e))

        t_acc = 0.0
        for _ in range(iters):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            try:
                out_ts = flash_attn_tessera(tq, tk, tv, args.causal)
            except Exception as e:
                print("⚠️ Tessera call failed:", repr(e))
                break
            if device == "cuda":
                torch.cuda.synchronize()
            t_acc += (time.time() - t0) * 1000.0
        else:
            timings["tessera_ms_per_iter"] = t_acc / iters

        # Basic correctness check (if conversion back is straightforward)
        try:
            # If Tessera returns its own Tensor wrapper, try to unwrap to torch
            if hasattr(out_ts, "to_torch"):
                out_ts_torch = out_ts.to_torch()
            else:
                out_ts_torch = torch.as_tensor(out_ts)
            max_diff = (out_ref.to(torch.float32) - out_ts_torch.to(torch.float32)).abs().max().item()
            print(f"✅ Max abs diff (ref vs Tessera): {max_diff:.3e}")
        except Exception as e:
            print("ℹ️ Skipped correctness diff (conversion issue):", repr(e))
    else:
        print("\n📊 Tessera path is a placeholder (Tessera not installed).")
        print("   Import error:", _import_error)

    # Save timings
    (ARTIFACTS / "timings.json").write_text(json.dumps(timings, indent=2))
    print(f"\n⏱️ Timings: {json.dumps(timings, indent=2)}")

    # Optional IR dump
    if args.dump_ir:
        print("\n📦 Dumping IR (best-effort)…")
        if TESSERA_AVAILABLE:
            # Small sample to keep IR compact
            tq = tsr.randn([2,"H",64,64])  # type: ignore
            tk = tsr.randn([2,"H",64,64])  # type: ignore
            tv = tsr.randn([2,"H",64,64])  # type: ignore
            summary = dump_all_irs(flash_attn_tessera, (tq, tk, tv, False), {})
        else:
            summary = dump_all_irs(flash_attn_tessera, (), {})
        print(json.dumps(summary, indent=2))
        print(f"Artifacts written to: {ARTIFACTS}")

    print("\n🎉 Demo complete.")

if __name__ == "__main__":
    main()
