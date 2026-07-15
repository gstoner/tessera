"""Measure ROCM-9 routes on exact gfx1151 and update the shared D2 corpus.

Both candidates consume the same non-identity PLHD page table and token order.
The runtime oracle gates every row before its separate HIP-event and full-call
latencies are recorded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", nargs="+", type=int,
                        default=(128, 512, 2048, 8192))
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-save-corpus", action="store_true")
    args = parser.parse_args()

    from tessera import runtime as rt
    from tessera.cache.paged_kv import (
        PagedKVBufferABI,
        _paged_attention_rocm,
        _rocm_paged_attention_route_evidence,
    )
    from tessera.compiler.emit import autotune as at
    from tessera.compiler.emit.kernel_emitter import SpecPolicy, bucket_key

    chip = str(rt._rocm_chip())
    if chip != "gfx1151" or not rt._rocm_compiled_flash_attn_available():
        print(f"exact gfx1151 ROCm lane unavailable (detected {chip!r})")
        return 0
    if args.heads % args.kv_heads:
        raise ValueError("query heads must be divisible by KV heads")

    cache = at.MeasureCache()
    at.load_corpus(cache=cache)
    rng = np.random.default_rng(20260714)
    rows: list[dict[str, object]] = []
    for tokens in args.tokens:
        pages = (tokens + args.page_size - 1) // args.page_size
        table = np.roll(np.arange(pages, dtype=np.int32), max(1, pages // 3))
        kp = (rng.standard_normal(
            (pages, args.page_size, args.kv_heads, args.dim)) * .1).astype(np.float32)
        vp = (rng.standard_normal(kp.shape) * .1).astype(np.float32)
        abi = PagedKVBufferABI(kp, vp, table, logical_length=tokens)
        q = (rng.standard_normal((args.heads, 1, args.dim)) * .1).astype(np.float32)
        # Reverse chunks to cover arbitrary ordering and every page crossing.
        indices = np.arange(tokens, dtype=np.int64).reshape(
            -1, args.page_size)[:, ::-1].reshape(-1)
        _rocm_paged_attention_route_evidence.clear()
        _, execution = _paged_attention_rocm(
            q, abi, indices, args.dim ** -.5, True, _force_measure=True)
        if execution != "native_gpu":
            raise RuntimeError(f"native ROCm route measurement failed at T={tokens}")
        evidence = next(iter(_rocm_paged_attention_route_evidence.values()))
        bucket = bucket_key(
            (1, args.heads, args.kv_heads, tokens, args.dim, args.page_size),
            SpecPolicy.BUCKET)
        for timing, field in ((at.TIMING_DEVICE, "device_ms"),
                              (at.TIMING_END_TO_END, "end_to_end_ms")):
            candidates = dict(evidence[field])
            winner = min(candidates, key=candidates.__getitem__)
            cache.put(("rocm:gfx1151", "rocm", "paged_kv_decode", bucket,
                       "f32", timing), at.MeasureRecord(
                           winner, candidates[winner], candidates))
        rows.append({"backend": "rocm", "device": chip,
                     "op": "paged_kv_decode", "tokens": tokens,
                     "heads": args.heads, "kv_heads": args.kv_heads,
                     "dim": args.dim, "page_size": args.page_size, **evidence})
        print(f"T={tokens}: device={evidence['device_winner']} "
              f"end_to_end={evidence['end_to_end_winner']}")

    if not args.no_save_corpus:
        at.save_corpus(cache=cache)
    payload = {"schema_version": 1, "evidence_arch": chip, "rows": rows}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
