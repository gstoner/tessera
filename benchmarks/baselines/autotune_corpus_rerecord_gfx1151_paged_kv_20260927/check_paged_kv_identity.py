"""Serve the re-recorded gfx1151 paged-KV rows, then prove a changed route misses.

Sync AUTOTUNE-LAUNCH-INTEGRITY-2026-09-27 (closes
AUTOTUNE-KERNEL-IDENTITY-PAGED-KV). Run on Princess-Luna in a fresh process
from the repo root, with the corpus under test in ``TESSERA_AUTOTUNE_CORPUS``
and, to check the gate's "a build tree other than the recording one",
``TESSERA_OPT`` pointing at a ``tessera-opt`` from another build tree:

    source ~/programming/tessera/.venv/bin/activate && source scripts/_rocm_env.sh
    PYTHONPATH=python python benchmarks/baselines/autotune_corpus_rerecord_gfx1151_paged_kv_20260927/check_paged_kv_identity.py

Per ``rocm:gfx1151`` ``paged_kv_decode`` row: the live route identities
(``rocm_paged_attention_route_identities``, the recorder's 4/4 heads, head_dim
32, causal) must equal the stamped ones, and the production warm start
(``cache/paged_kv.py::_rocm_paged_attention_corpus_winner``) must return the
row's winner exactly when the row is admissible. Then each HIP emitter, and the
FA-2 image the ``gather_fa`` route launches, is perturbed with the ROCm pins
asserted unchanged: every row must miss, and nothing is served. Exit 0 only
when all of that holds.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from tessera import runtime as rt                               # noqa: E402
from tessera.cache import paged_kv                              # noqa: E402
from tessera.compiler import toolchain_identity as TI           # noqa: E402
from tessera.compiler.emit import autotune as at                # noqa: E402
from tessera.compiler.emit import rocm_hip as R                 # noqa: E402

DEVICE = "rocm:gfx1151"
HEADS, KV_HEADS, DIM, PAGE = 4, 4, 32, 16


def _rows(cache: at.MeasureCache) -> list:
    return sorted((k for k in cache._store
                   if k[0] == DEVICE and k[2] == "paged_kv_decode"), key=str)


def check(cache: at.MeasureCache, keys: list) -> dict:
    routes = R.rocm_paged_attention_route_identities(
        q_heads=HEADS, kv_heads=KV_HEADS, head_dim=DIM, causal=True)
    out = {}
    for key in keys:
        rec = cache._store[key]
        q_len, heads, kv_heads, tokens, dim, page = key[3]
        matched = at.route_record_matches(rec, routes)
        served = (paged_kv._rocm_paged_attention_corpus_winner(
            heads, kv_heads, q_len, tokens, dim, page)
            if key[5] == at.TIMING_END_TO_END else None)
        out[key] = (matched, served, at.record_is_admissible(rec), rec.winner)
    return out


def main() -> int:
    if rt._rocm_live_arch() != "gfx1151":
        print("no gfx1151 device: this host cannot evaluate the check")
        return 2
    cache = at.MeasureCache()
    loaded = at.load_corpus(cache=cache)
    keys = _rows(cache)
    pins = TI.toolchain_identity("rocm").digest
    print(f"loaded {loaded} rows; {len(keys)} gfx1151 paged-KV rows; "
          f"rocm toolchain {pins[:23]}...; tessera-opt {rt._tessera_opt_path()}")
    before = check(cache, keys)
    for key in keys:
        matched, served, admissible, winner = before[key]
        print(f"  {key[5]:10s} {list(key[3])!s:26s} winner={winner:9s} "
              f"admissible={admissible!s:5s} routes_match={matched!s:5s} "
              f"warm_start={served}")
    ok = all(v[0] for v in before.values())
    wrong = [k for k in keys if k[5] == at.TIMING_END_TO_END
             and before[k][1] != (before[k][3] if before[k][2] else None)]
    print(f"identities match {sum(v[0] for v in before.values())}/{len(keys)}; "
          f"warm start served {sum(v[1] is not None for v in before.values())} "
          f"(end-to-end rows; the reader does not consult device rows); "
          f"warm-start answers that disagree with the row's admission: {len(wrong)}")

    cu = "\n/* AUTOTUNE-LAUNCH-INTEGRITY: emitter changed */\n"
    image = rt._rocm_flash_attn_image

    def fa_changed(head_dim, *a, **k):
        # A genuinely different FA-2 kernel (the head_dim-64 image) where the
        # route launches the head_dim-32 one: what a changed generator yields.
        return image(2 * int(head_dim), *a, **k)

    perturb = {
        "_synthesize_paged_kv_read_hip": (
            R, "_synthesize_paged_kv_read_hip",
            (lambda f: lambda: f() + cu)(R._synthesize_paged_kv_read_hip)),
        "_synthesize_paged_attention_direct_hip": (
            R, "_synthesize_paged_attention_direct_hip",
            (lambda f: lambda: f() + cu)(R._synthesize_paged_attention_direct_hip)),
        "FA-2 image (gather_fa)": (rt, "_rocm_flash_attn_image", fa_changed),
    }
    flipped: set = set()
    for label, (mod, attr, changed) in perturb.items():
        original = getattr(mod, attr)
        setattr(mod, attr, changed)
        try:
            TI.clear_identity_cache()
            assert TI.toolchain_identity("rocm").digest == pins, "a pin moved"
            after = check(cache, keys)
        finally:
            setattr(mod, attr, original)
        missed = {k for k in keys if before[k][0] and not after[k][0]}
        flipped |= missed
        print(f"perturb {label:40s}: {len(missed)} rows now miss; served "
              f"{sum(after[k][1] is not None for k in keys)}")
    ok = ok and not wrong and len(flipped) == len(keys)
    print(f"every row missed under some single perturbation: {len(flipped)}/{len(keys)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
