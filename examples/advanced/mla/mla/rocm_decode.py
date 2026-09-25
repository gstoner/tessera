"""Exact-device ROCm MLA decode-step smoke for the example.

This exercises the existing compiled runtime lane, not the Apple-only demo or
the toy Graph IR prefill skeleton. A reference fallback is a failed smoke.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tessera import runtime as rt
from tessera.cache import LatentKVCacheHandle
from tessera.stdlib import attention


@dataclass(frozen=True)
class RocmDecodeSummary:
    live_arch: str
    execution_kind: str
    compiler_path: str
    max_abs_error: float
    cache_rows: int


def run_rocm_decode_smoke(*, seed: int = 46) -> RocmDecodeSummary:
    """Compare one native absorbed-latent step and cache mutation to stdlib."""
    live = rt._rocm_live_arch()
    if live is None:
        raise RuntimeError("ROCm MLA smoke requires a selected HIP device")
    # The runtime's compiler chip (TESSERA_ROCM_CHIP, default gfx1151) must
    # name the device the kernel launches on -- the same check Gumiho makes.
    chip = rt._rocm_chip().split(":", 1)[0]
    if chip != live:
        raise RuntimeError(
            f"set TESSERA_ROCM_CHIP={live} for the selected device; compiler chip is {chip!r}"
        )

    rng = np.random.default_rng(seed)
    hidden, heads, d_nope, d_rope, d_v, d_c = 8, 2, 3, 2, 4, 5
    weights = attention.MLAWeights(
        w_dkv=(rng.standard_normal((hidden, d_c)) * 0.12).astype(np.float32),
        w_uk=(rng.standard_normal((d_c, heads * d_nope)) * 0.11).astype(np.float32),
        w_uv=(rng.standard_normal((d_c, heads * d_v)) * 0.10).astype(np.float32),
        w_q=(rng.standard_normal((hidden, heads * (d_nope + d_rope))) * 0.09).astype(np.float32),
        w_kr=(rng.standard_normal((hidden, d_rope)) * 0.08).astype(np.float32),
        num_heads=heads, d_nope=d_nope, d_rope=d_rope, d_v=d_v,
    )
    prompt = (rng.standard_normal((4, hidden)) * 0.2).astype(np.float32)
    token = (rng.standard_normal((2, hidden)) * 0.2).astype(np.float32)

    def caches() -> tuple[LatentKVCacheHandle, LatentKVCacheHandle]:
        latent = LatentKVCacheHandle(d_c, max_seq=16, dtype="fp32")
        rope = LatentKVCacheHandle(d_rope, max_seq=16, dtype="fp32")
        c, kr = attention.compress_latent(prompt, weights)
        latent.append(c)
        rope.append(kr)
        return latent, rope

    ref_lat, ref_rope = caches()
    got_lat, got_rope = caches()
    reference = attention.mla_decode_step(token, ref_lat, ref_rope, weights)
    names = ["x_t", "latent_cache", "rope_cache", "weights"]
    artifact = rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_exotic_attn_compiled",
        "executable": True, "arg_names": names, "output_name": "o",
        "ops": [{"op_name": "tessera.mla_decode_step", "result": "o",
                 "operands": names, "kwargs": {"absorb": True}}],
    })
    result = rt.launch(artifact, (token, got_lat, got_rope, weights))
    if not result["ok"]:
        raise RuntimeError(f"ROCm MLA launch failed: {result.get('reason')}")
    kind = str(result["execution_kind"])
    if kind != "native_gpu":
        raise RuntimeError(f"ROCm MLA smoke refused {kind} fallback")
    if result.get("compiler_path") != "rocm_exotic_attn_compiled":
        raise RuntimeError(
            f"ROCm MLA smoke ran {result.get('compiler_path')!r}, not the requested compiled lane"
        )
    output = np.asarray(result["output"])
    np.testing.assert_allclose(output, reference, rtol=2e-4, atol=2e-4)
    if (got_lat.current_seq != ref_lat.current_seq
            or got_rope.current_seq != ref_rope.current_seq):
        raise AssertionError("ROCm MLA cache length differs from reference")
    np.testing.assert_array_equal(
        got_lat.read(0, got_lat.current_seq), ref_lat.read(0, ref_lat.current_seq)
    )
    np.testing.assert_array_equal(
        got_rope.read(0, got_rope.current_seq), ref_rope.read(0, ref_rope.current_seq)
    )
    return RocmDecodeSummary(
        live_arch=live,
        execution_kind=kind,
        compiler_path=str(result["compiler_path"]),
        max_abs_error=float(np.max(np.abs(output - reference))),
        cache_rows=got_lat.current_seq,
    )
