"""Serve the re-recorded sm_120 registry rows, then prove a changed NVIDIA
emitter misses (sync SM120-AUTOTUNE-FOLLOWUPS-2026-09-27).

Adapted from ``autotune_corpus_rerecord_sm120_20260927/check_emitted_identity.py``
for the follow-up re-record. Run on The-Super-Bear, in a fresh process, from
the repo root of the tree that recorded the rows (its ``build/`` and
``build-nvidia-cuda/`` built):

    source ~/programming/tessera/.venv/bin/activate && source scripts/_nvidia_env.sh
    PYTHONPATH=python python benchmarks/baselines/autotune_corpus_rerecord_sm120_followups_20260927/check_emitted_identity.py

Per ``nvidia:sm_120`` registry row (workloads rebuilt from the recorder's
shape lists):

* ``ident`` -- every live candidate the row TIMED carries an identity equal to
  its stamp;
* ``partial_field`` -- live candidates the row did NOT time. Since the scalar
  lanes (``nvidia_generic_cuda`` / ``nvidia_flash_attn`` / ``nvidia_gated``)
  gained a device timer this must be empty for every row: a non-empty list
  fails the check;
* ``served`` -- what ``corpus_winner`` returns asked the way ``run_arbitrated``
  asks (no explicit dims). ``gated_matmul`` rows are now found this way too
  (``_infer_dims`` gained the gated rule, AUTOTUNE-GATED-INFER-DIMS); the
  answer with the recorder's explicit dims (``served_dims``) must agree.

Then each NVIDIA emitter is perturbed in turn with the toolchain digest
asserted unchanged, and finally all together: every row must miss. Exit 0
only when every row matches, no row has an untimed live candidate, the
inferred-dims and explicit-dims answers agree, every row misses with all
emitters perturbed, and every row missed under some single perturbation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from tessera import runtime as rt                               # noqa: E402
from tessera.compiler import ptx_emit as pe                     # noqa: E402
from tessera.compiler import toolchain_identity as TI           # noqa: E402
from tessera.compiler.emit import autotune as at                # noqa: E402
from tessera.compiler.emit import nvidia_cuda as N              # noqa: E402
from tessera.compiler.emit.candidate import (                   # noqa: E402
    OP_ATTENTION,
    OP_FUSED_REGION,
    OP_GATED_MATMUL,
    OP_MATMUL,
    live_candidates,
)
from tessera.compiler.emit.kernel_emitter import SpecPolicy, bucket_key  # noqa: E402
from tessera.compiler.fusion import (                           # noqa: E402
    AttentionRegion,
    FusedRegion,
    GatedMatmulRegion,
    MatmulRegion,
)

DEVICE = "nvidia:sm_120"
MATMUL = ("64x64x64", "256x256x256", "512x512x512", "1024x1024x1024",
          "2048x2048x2048", "128x256x64", "127x259x63")
FUSED = ("64x64x64", "256x256x256", "128x512x256", "127x259x63", "128x256x256")
ATTN = ("128x128x64x64", "64x512x64x64", "64x256x64x64")
GATED = ("64x256x256", "128x512x512")
COMPOSED = ("f32", "fp8_e4m3", "fp8_e5m2")
TIMINGS = (at.TIMING_END_TO_END, at.TIMING_DEVICE)


def _dims(text: str) -> tuple[int, ...]:
    return tuple(int(p) for p in text.split("x"))


def workloads() -> dict[tuple[Any, ...], tuple[Any, tuple[Any, ...], tuple[int, ...]]]:
    """key -> (region, inputs, recorder dims) for every workload the recorder
    races. Operand values do not enter any identity, so a fixed seed suffices."""
    rng = np.random.default_rng(0)
    out: dict[tuple[Any, ...], tuple[Any, tuple[Any, ...], tuple[int, ...]]] = {}

    def add(op: str, dtype: str, region: Any, inputs: tuple[Any, ...],
            dims: tuple[int, ...]) -> None:
        # The recorder races f16 fused/attention end to end only.
        for timing in (TIMINGS if dtype != "f16" else (at.TIMING_END_TO_END,)):
            out[(DEVICE, "nvidia", op, bucket_key(dims, SpecPolicy.BUCKET),
                 dtype, timing)] = (region, inputs, dims)

    def mat(*shape: int) -> Any:
        return (rng.standard_normal(shape) * .1).astype(np.float32)

    for dtype in ("float16", "bfloat16"):
        for s in MATMUL:
            m, n, k = _dims(s)
            add(OP_MATMUL, dtype, MatmulRegion(dtype=dtype), (mat(m, k), mat(k, n)),
                (m, n, k))
    for storage in ("f16", *COMPOSED):
        for s in FUSED:
            m, n, k = _dims(s)
            region = (FusedRegion(epilogue=("bias", "gelu")) if storage == "f16"
                      else FusedRegion(epilogue=("bias", "gelu"), storage_dtype=storage))
            add(OP_FUSED_REGION, storage, region, (mat(m, k), mat(k, n), mat(n)),
                (m, n, k))
        for s in ATTN:
            m, nk, d, dv = _dims(s)
            region = (AttentionRegion(scale=d ** -.5, causal=True) if storage == "f16"
                      else AttentionRegion(scale=d ** -.5, causal=True,
                                           storage_dtype=storage))
            add(OP_ATTENTION, storage, region, (mat(m, d), mat(nk, d), mat(nk, dv)),
                (m, nk, d, dv))
    for storage in COMPOSED:
        for s in GATED:
            m, h, k = _dims(s)
            add(OP_GATED_MATMUL, storage,
                GatedMatmulRegion(gate_act="silu", storage_dtype=storage),
                (mat(m, k), mat(k, h), mat(k, h)), (m, h, k))
    return out


def check(cache: at.MeasureCache, work: dict, keys: list) -> dict:
    result = {}
    for key in keys:
        region, inputs, dims = work[key]
        _, _, op, _, dtype, timing = key
        rec = cache._store[key]
        live = live_candidates(region, op, "nvidia", inputs)
        timed = {n: c for n, c in live.items() if n in rec.candidates}
        ident = at._record_matches_live_delegates(rec, timed, region, inputs)
        partial = sorted(n for n in live if n not in rec.candidates)
        served = at.corpus_winner(region, op, "nvidia", *inputs, dtype=dtype,
                                  cache=cache, device=DEVICE, timing=timing)
        served_dims = at.corpus_winner(region, op, "nvidia", *inputs, dims=dims,
                                       dtype=dtype, cache=cache, device=DEVICE,
                                       timing=timing)
        result[key] = (ident, served, served_dims,
                       at.record_is_admissible(rec), rec.winner, partial)
    return result


def _plus_line(fn: Callable[..., str], line: str) -> Callable[..., str]:
    return lambda *a, **k: fn(*a, **k) + line


def perturbations() -> dict[str, tuple[Any, str, Callable[..., Any]]]:
    cu = "\n/* AUTOTUNE-EMITTED-IDENTITY: emitter changed */\n"
    ptx = '\n.pragma "nounroll";\n'
    tile = rt._nvidia_tile_matmul_ptx
    return {
        "_synthesize_fused_cuda": (N, "_synthesize_fused_cuda",
                                   _plus_line(N._synthesize_fused_cuda, cu)),
        "_synthesize_attention_cuda": (N, "_synthesize_attention_cuda",
                                       _plus_line(N._synthesize_attention_cuda, cu)),
        "_synthesize_gated_cuda": (N, "_synthesize_gated_cuda",
                                   _plus_line(N._synthesize_gated_cuda, cu)),
        "_synthesize_mma_fused_cuda": (N, "_synthesize_mma_fused_cuda",
                                       _plus_line(N._synthesize_mma_fused_cuda, cu)),
        "_synthesize_mma_attn_cuda": (N, "_synthesize_mma_attn_cuda",
                                      _plus_line(N._synthesize_mma_attn_cuda, cu)),
        "_synthesize_mma_gated_cuda": (N, "_synthesize_mma_gated_cuda",
                                       _plus_line(N._synthesize_mma_gated_cuda, cu)),
        "_synthesize_resident_ops_cuda": (
            N, "_synthesize_resident_ops_cuda",
            _plus_line(N._synthesize_resident_ops_cuda, cu)),
        "ptx_emit.emit_mma_sync_gemm_ptx": (
            pe, "emit_mma_sync_gemm_ptx", _plus_line(pe.emit_mma_sync_gemm_ptx, ptx)),
        "tessera-nvidia-opt Tile PTX": (
            rt, "_nvidia_tile_matmul_ptx",
            lambda s, d: (tile(s, d)[0], tile(s, d)[1] + ptx)),
    }


def main() -> int:
    if rt._nvidia_device_name() != "sm_120":
        print("no sm_120 device: this host cannot evaluate the check")
        return 2
    cache = at.MeasureCache()
    loaded = at.load_corpus(cache=cache)
    work = workloads()
    keys = sorted((k for k in cache._store
                   if k[0] == DEVICE and k[2] in
                   (OP_MATMUL, OP_FUSED_REGION, OP_ATTENTION, OP_GATED_MATMUL)),
                  key=str)
    missing = [k for k in keys if k not in work]
    unrecorded = [k for k in work if k not in cache._store]
    pins = TI.toolchain_identity("nvidia").digest
    print(f"loaded {loaded} rows, {len(cache.stale_records())} stale; "
          f"{len(keys)} sm_120 registry rows; nvidia toolchain {pins[:23]}...")
    if missing or unrecorded:
        print(f"rows without a rebuilt workload: {missing}")
        print(f"workloads with no row: {unrecorded}")
        return 1

    before = check(cache, work, keys)
    for key in keys:
        ident, served, served_dims, admissible, winner, partial = before[key]
        _, _, op, bucket, dtype, timing = key
        print(f"  {op:12s} {dtype:9s} {timing:10s} {list(bucket)!s:22s} "
              f"winner={winner:36s} admissible={admissible!s:5s} ident={ident!s:5s} "
              f"served={served} served_dims={served_dims}"
              + (f" partial_field(untimed)={partial}" if partial else ""))
    matched = sum(v[0] for v in before.values())
    served = sum(v[1] is not None for v in before.values())
    served_dims = sum(v[2] is not None for v in before.values())
    admissible = sum(v[3] for v in before.values())
    partial_rows = sum(bool(v[5]) for v in before.values())
    disagree = [k for k in keys if before[k][1] != before[k][2]]
    print(f"rows whose live field includes a candidate the row did not time: "
          f"{partial_rows}")
    print(f"rows where inferred dims and the recorder's dims are served "
          f"differently: {len(disagree)}")
    print(f"before: identities match {matched}/{len(keys)}; admissible "
          f"{admissible}; served with inferred dims {served}; served with the "
          f"recorder's dims {served_dims}")

    flipped: set = set()
    for label, (mod, attr, changed) in perturbations().items():
        original = getattr(mod, attr)
        setattr(mod, attr, changed)
        try:
            TI.clear_identity_cache()
            assert TI.toolchain_identity("nvidia").digest == pins, "a pin moved"
            after = check(cache, work, keys)
        finally:
            setattr(mod, attr, original)
        missed = {k for k in keys if before[k][0] and not after[k][0]}
        flipped |= missed
        still = sum(after[k][2] is not None for k in keys)
        print(f"perturb {label:34s}: {len(missed):3d} rows now miss; "
              f"served (dims) {still}")

    originals = []
    for mod, attr, changed in perturbations().values():
        originals.append((mod, attr, getattr(mod, attr)))
        setattr(mod, attr, changed)
    try:
        TI.clear_identity_cache()
        assert TI.toolchain_identity("nvidia").digest == pins, "a pin moved"
        after = check(cache, work, keys)
    finally:
        for mod, attr, original in originals:
            setattr(mod, attr, original)
    all_missed = sum(not after[k][0] for k in keys)
    all_served = sum(after[k][2] is not None or after[k][1] is not None for k in keys)
    print(f"all emitters perturbed (pins unchanged): {all_missed}/{len(keys)} miss "
          f"their identity; {all_served} served")
    print(f"every row missed under some single perturbation: "
          f"{len(flipped)}/{len(keys)}")
    ok = (matched == len(keys) and partial_rows == 0 and not disagree
          and all_missed == len(keys) and all_served == 0
          and len(flipped) == len(keys))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
