"""Serve the re-recorded gfx1151 fused_region rows, then prove a changed
emitter misses (sync AUTOTUNE-EMITTED-IDENTITY-2026-09-27).

Run on the box that recorded the rows, in a fresh process, from the repo root:

    source ~/programming/tessera/.venv/bin/activate && source scripts/_rocm_env.sh
    PYTHONPATH=python python benchmarks/baselines/autotune_corpus_rerecord_20260927/check_emitted_identity.py

For each of the 8 rows it asks `corpus_winner` the way ordinary
`run_arbitrated` dispatch does (no explicit dims: `_infer_dims` derives
(M, N, K) from the operands), prints the served winner and every live
candidate's identity digest, then perturbs `rocm_hip._synthesize_fused_hip`
(one added comment line; no pin moves) and asks again: every row must miss.
Exit status 0 only when all 8 are served before and none after.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from tessera.compiler import fusion as F                       # noqa: E402
from tessera.compiler import toolchain_identity as TI          # noqa: E402
from tessera.compiler.emit import autotune as at               # noqa: E402
from tessera.compiler.emit import rocm_hip                     # noqa: E402
from tessera.compiler.emit.candidate import (                  # noqa: E402
    OP_FUSED_REGION,
    live_candidates,
)

SHAPES = (64, 256, 512, 1024)
TIMINGS = (at.TIMING_END_TO_END, at.TIMING_DEVICE)


def ask(cache, region, size, timing):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((size, size)).astype(np.float32)
    b = rng.standard_normal((size, size)).astype(np.float32)
    bias = rng.standard_normal((size,)).astype(np.float32)
    winner = at.corpus_winner(region, OP_FUSED_REGION, "rocm", a, b, bias,
                              dtype="f16", cache=cache, device="rocm:gfx1151",
                              timing=timing)
    live = live_candidates(region, OP_FUSED_REGION, "rocm", (a, b, bias))
    idents = {name: cand.artifact_identity(region, a, b, bias)
              for name, cand in sorted(live.items())}
    return winner, idents


def digest(ident):
    if ident is None:
        return "NONE"
    return str(ident.get("source_sha256")
               or ident.get("instruction_stream_sha256"))[:16]


def main() -> int:
    cache = at.MeasureCache()
    loaded = at.load_corpus(cache=cache)
    print(f"loaded {loaded} rows, {len(cache.stale_records())} stale; "
          f"device {at._device_id('rocm')}; rocm toolchain "
          f"{TI.toolchain_identity('rocm').digest[:23]}...")
    region = F.FusedRegion(epilogue=("bias", "gelu"))
    served = 0
    for size in SHAPES:
        for timing in TIMINGS:
            winner, idents = ask(cache, region, size, timing)
            served += winner is not None
            print(f"  {size}^3 {timing:10s} served={winner}  " +
                  " ".join(f"{n}={digest(i)}" for n, i in idents.items()))
    print(f"served before the emitter change: {served}/8")

    pins = TI.toolchain_identity("rocm").digest
    original = rocm_hip._synthesize_fused_hip
    rocm_hip._synthesize_fused_hip = (
        lambda region: original(region) + "\n/* emitter changed */\n")
    TI.clear_identity_cache()
    assert TI.toolchain_identity("rocm").digest == pins, "a pin moved"
    missed = 0
    for size in SHAPES:
        for timing in TIMINGS:
            winner, idents = ask(cache, region, size, timing)
            missed += winner is None
            print(f"  {size}^3 {timing:10s} served={winner}  "
                  f"rocm_generic_hip={digest(idents.get('rocm_generic_hip'))}")
    rocm_hip._synthesize_fused_hip = original
    print(f"missed after the emitter change (pins unchanged): {missed}/8")
    return 0 if served == 8 and missed == 8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
