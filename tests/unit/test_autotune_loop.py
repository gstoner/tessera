"""Phase 3 — close the optimizing loop: measured-latency variant selection.

The synthesizer's measured autotuner (`autotune_matmul_epilogue`, which times
each variant on-device and correctness-gates it) is now invoked automatically:
`select_variant` measures + caches the fastest variant on first use of a shape
when `TESSERA_AUTOTUNE` is on, so the executed Apple GPU lane runs the
measured-best kernel rather than a static default. See
docs/audit/compiler/COMPILER_AUDIT.md (Phase 3).
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler import fusion as F


def _gelu(x):
    t = np.clip(0.7978845608028654 * (x + 0.044715 * x**3), -30.0, 30.0)
    return 0.5 * x * (1.0 + np.tanh(t))


@pytest.fixture(autouse=True)
def _fresh_corpus():
    F.clear_autotune_corpus()
    yield
    F.clear_autotune_corpus()


def test_autotune_enabled_reads_env(monkeypatch):
    monkeypatch.delenv("TESSERA_AUTOTUNE", raising=False)
    assert F.autotune_enabled() is False
    monkeypatch.setenv("TESSERA_AUTOTUNE", "1")
    assert F.autotune_enabled() is True


def test_select_variant_default_is_static_no_corpus_write():
    region = F.FusedRegion(epilogue=("gelu",))
    v = F.select_variant(region, 64, 128, 64, autotune=False)
    assert v in F.SYNTH_VARIANTS
    assert F._corpus_key(region, 64, 128, 64) not in F._AUTOTUNE_CORPUS  # no measurement


def test_select_variant_autotune_measures_and_caches():
    region = F.FusedRegion(epilogue=("gelu",))
    key = F._corpus_key(region, 64, 128, 64)
    assert key not in F._AUTOTUNE_CORPUS
    v = F.select_variant(region, 64, 128, 64, autotune=True)
    assert v in F.SYNTH_VARIANTS
    # On Metal the measurement populates the corpus with the measured-best; off
    # Metal (no native run) the corpus stays empty and the default is returned.
    if key in F._AUTOTUNE_CORPUS:
        assert F._AUTOTUNE_CORPUS[key] == v          # cached == returned
        # second call is an O(1) hit (no re-measure path needed).
        assert F.select_variant(region, 64, 128, 64, autotune=True) == v


def test_autotuned_variant_is_correct():
    # Whichever variant autotuning picks, it must be numerically correct — the
    # measure step gates every candidate against the numpy reference (F4) before
    # its latency can win.
    region = F.FusedRegion(epilogue=("gelu",))
    v = F.select_variant(region, 32, 64, 32, autotune=True)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((32, 32)).astype(np.float32)
    B = rng.standard_normal((32, 64)).astype(np.float32)
    out, _ex = F.run_fused_region(region, A, B, variant=v)
    np.testing.assert_allclose(np.asarray(out), _gelu(A @ B), rtol=1e-4, atol=1e-4)


def test_autotune_record_only_counts_correct_variants():
    # The measured record's chosen variant (if any) must be one it verified
    # correct — never a fast-but-wrong kernel (the Sakana invariant).
    region = F.FusedRegion(epilogue=("gelu",))
    rec = F.autotune_matmul_epilogue(region, 32, 64, 32)
    if rec is not None and rec.chosen is not None:
        assert rec.correct.get(rec.chosen) is True
