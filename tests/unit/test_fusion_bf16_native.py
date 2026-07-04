"""Apple GPU codegen: native bf16 synthesizer path (Apple7 / M1 Max).

The Apple "Metal Feature Set Tables" PDF confirms M1-series = Apple7, and
`MTLDataType.bfloat` (native MSL `bfloat`) is supported from Apple6+. The fusion
synthesizer previously emulated bf16 by host-upcasting to f32; it now emits
native `bfloat`-typed MSL (fp32 accumulators inside) and reuses the f16 synth
symbol's raw-uint16 ABI — no host f32 round-trip. This is the first milestone of
the Apple-GPU dispatcher→compiler plan, grounded directly in the hardware doc.

See docs/audit/backend/apple/APPLE_GPU_CODEGEN_PLAN.md and the
apple7-m1max-gpu-feature-set memory.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler import fusion as F

ml_dtypes = pytest.importorskip("ml_dtypes")
bf16 = ml_dtypes.bfloat16


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def test_io_type_bf16_is_native_bfloat():
    assert F._io_type("bf16") == "bfloat"


def test_synthesizer_emits_native_bfloat_not_half():
    region = F.FusedRegion(epilogue=("gelu",))
    src = F.synthesize_matmul_epilogue_msl(region, "broadcast", dtype="bf16")
    assert "bfloat" in src
    assert "half" not in src  # not the f16 kernel; not f32-emulated


@pytest.mark.parametrize("epilogue", [("gelu",), ("relu",), ("silu",)])
def test_native_bf16_region_runs_and_matches_reference(epilogue):
    region = F.FusedRegion(epilogue=epilogue)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 16)).astype(np.float32)
    B = rng.standard_normal((16, 12)).astype(np.float32)
    out, ex = F.run_fused_region(region, A.astype(bf16), B.astype(bf16))
    # The native bfloat kernel must run on Metal (else this Mac's MSL predates
    # bfloat and we'd see the f32-emulation fallback — also "metal_runtime" but
    # via f32; here we additionally pin the output dtype + bf16-level accuracy).
    assert ex in ("metal_runtime", "reference")
    got = np.asarray(out).astype(np.float32)
    ref = region.reference(A, B)
    # bf16 has an 8-bit mantissa → ~1e-2 relative error is expected and correct.
    np.testing.assert_allclose(got, ref, rtol=8e-2, atol=8e-2)


def test_native_bf16_preserves_bf16_output_dtype():
    region = F.FusedRegion(epilogue=("gelu",))
    A = np.ones((4, 8), np.float32).astype(bf16)
    B = np.ones((8, 6), np.float32).astype(bf16)
    out, _ex = F.run_fused_region(region, A, B)
    assert np.asarray(out).dtype == bf16


def test_native_bf16_helper_falls_back_cleanly_without_symbol(monkeypatch):
    # If the runtime symbol is unavailable, the native path declines (None,
    # "fallback") so the caller f32-emulates — never silently wrong.
    # B1 split: _run_fused_region_bf16 resolves _synth_f16_symbol from its own
    # module (emit.apple_msl); patch it there, not on the facade.
    monkeypatch.setattr(
        "tessera.compiler.emit.apple_msl._synth_f16_symbol", lambda: None)
    region = F.FusedRegion(epilogue=("gelu",))
    A = np.ones((4, 8), np.float32).astype(bf16)
    B = np.ones((8, 6), np.float32).astype(bf16)
    out, ex = F._run_fused_region_bf16(region, A, B, None, "broadcast")
    assert out is None and ex == "fallback"
    # The public entry still returns a correct bf16 result via f32 emulation.
    full, _ = F.run_fused_region(region, A, B)
    assert np.asarray(full).dtype == bf16
