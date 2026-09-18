"""A compiled-family test skips, not fails, where the pipeline would refuse.

`rocm_pipeline.promoted_families` is the fail-closed rule the executable
pipeline applies at launch; these guards ask it the same question first. The
rule and the guards are held together here so a promotion (or a new arch) that
changes one cannot silently strand the other.
"""
from __future__ import annotations

import pytest

from tessera.compiler.rocm_pipeline import FAMILY_PLUGINS, ROCMExecutablePipeline, promoted_families
from tests._support import rocm_build


def test_gfx1151_has_every_family_and_unknown_archs_have_none():
    assert promoted_families("gfx1151") == frozenset(FAMILY_PLUGINS)
    assert promoted_families("gfx1201") == {
        "softmax", "reduction", "matmul", "attention", "attention_backward",
        "control_state_machine", "ebm_affine_langevin",
        # GFX1201-PARITY slice 2 (2026-09-17): the scalar and row-program
        # families, measured on Tajasarus (972 tests skip -> pass, 0 kernel
        # failures).
        "scalar_unary", "scalar_binary", "scalar_compare", "scalar_logical",
        "scalar_bitwise", "scalar_predicate", "scalar_where",
        "scalar_activation", "loss_binary", "loss_pointwise", "loss_policy",
        "normalization", "rng_philox", "indexing_gather", "indexing_scatter",
        "position_alibi", "position_rope", "quant_dequant_gemm", "quant_fp",
        "quant_int4_pack", "reduction_arg", "scan", "optimizer",
        "fused_silu_mul"}
    assert promoted_families("gfx1201") < frozenset(FAMILY_PLUGINS)
    for arch in ("gfx1200", "gfx1250", "gfx1100", "gfx942", ""):
        assert promoted_families(arch) == frozenset()


@pytest.mark.parametrize("arch,family,accepted", [
    ("gfx1151", "scalar_unary", True),
    ("gfx1201", "softmax", True),
    ("gfx1201", "spectral_dft", False),
    ("gfx1200", "softmax", False),
])
def test_the_config_refuses_exactly_what_the_rule_says(arch, family, accepted):
    kwargs = dict(family=family, arch=arch)
    if accepted:
        ROCMExecutablePipeline(**kwargs)
    else:
        with pytest.raises(ValueError, match="no promoted family plugins"):
            ROCMExecutablePipeline(**kwargs)


def test_guards_skip_where_the_rule_refuses(monkeypatch):
    monkeypatch.setattr(rocm_build, "rocm_host_arch", lambda: "gfx1201")
    assert rocm_build.require_rocm_compiled_family("softmax", "matmul") == "gfx1201"
    with pytest.raises(pytest.skip.Exception, match="spectral_dft.*gfx1201"):
        rocm_build.require_rocm_compiled_family("softmax", "spectral_dft")
    with pytest.raises(pytest.skip.Exception, match="gfx1151 only"):
        rocm_build.require_rocm_compiled_lane_host()
    monkeypatch.setattr(rocm_build, "rocm_host_arch", lambda: "gfx1151")
    assert rocm_build.require_rocm_compiled_lane_host() == "gfx1151"
    assert rocm_build.require_rocm_compiled_family("scalar_unary") == "gfx1151"
    monkeypatch.setattr(rocm_build, "rocm_host_arch", lambda: None)
    with pytest.raises(pytest.skip.Exception, match="no ROCm device arch"):
        rocm_build.require_rocm_compiled_lane_host()


class _FakeRuntime:
    RuntimeArtifact = object

    def __init__(self, result=None, exc=None):
        self.result, self.exc = result, exc

    def launch(self, *a, **k):
        if self.exc:
            raise self.exc
        return self.result


def test_runtime_for_host_skips_only_this_hosts_unpromoted_refusal(monkeypatch):
    monkeypatch.setattr(rocm_build, "rocm_host_arch", lambda: "gfx1201")
    refusal = ("ROCm executable pipeline has no promoted family plugins for gfx1201; "
               "gfx1200/gfx1250 remain fail-closed pending exact-device evidence")
    wrapped = rocm_build.runtime_for_host(_FakeRuntime({"ok": False, "reason": refusal}))
    with pytest.raises(pytest.skip.Exception, match="gfx1201"):
        wrapped.launch()
    with pytest.raises(pytest.skip.Exception):
        rocm_build.runtime_for_host(_FakeRuntime(exc=ValueError(refusal))).launch()
    # A refusal about an arch a test pinned on purpose is not this host's, and
    # any other failure is a real failure.
    other = refusal.replace("gfx1201", "gfx1200")
    assert rocm_build.runtime_for_host(_FakeRuntime({"ok": False, "reason": other})).launch()["ok"] is False
    assert rocm_build.runtime_for_host(_FakeRuntime({"ok": False, "reason": "device_error"})).launch()["ok"] is False
    with pytest.raises(ValueError, match="something else"):
        rocm_build.runtime_for_host(_FakeRuntime(exc=ValueError("something else"))).launch()
    assert rocm_build.runtime_for_host(_FakeRuntime({"ok": True})).launch() == {"ok": True}
    # Attribute access forwards to the real module.
    assert rocm_build.runtime_for_host(_FakeRuntime()).RuntimeArtifact is object
    # No resolvable arch: nothing is rewritten.
    monkeypatch.setattr(rocm_build, "rocm_host_arch", lambda: None)
    assert rocm_build.runtime_for_host(_FakeRuntime({"ok": False, "reason": refusal})).launch()["ok"] is False


def test_runtime_for_host_skips_an_isa_contract_refusal_naming_this_host(monkeypatch):
    wmma = ("compiled WMMA GEMM: the 16x16x16 f16/bf16 WMMA fragment layout is a gfx11 "
            "(RDNA3/RDNA3.5) contract, hardware-verified on gfx1151; target 'gfx1201' needs "
            "its own layout (RDNA4 gfx12xx: 16x16x32 WMMA; CDNA: MFMA 32x32x8) which is "
            "arch-gated on that fragment ISA + silicon")
    monkeypatch.setattr(rocm_build, "rocm_host_arch", lambda: "gfx1201")
    with pytest.raises(pytest.skip.Exception, match="16x16x16"):
        rocm_build.runtime_for_host(_FakeRuntime({"ok": False, "reason": wmma})).launch()

    class Unavailable(Exception):
        pass

    rt = _FakeRuntime(exc=Unavailable(wmma))
    rt._RocmCompiledUnavailable = Unavailable
    with pytest.raises(pytest.skip.Exception):
        rocm_build.runtime_for_host(rt).launch()
    # On the host the contract IS verified for, the same text would be a real failure.
    monkeypatch.setattr(rocm_build, "rocm_host_arch", lambda: "gfx1151")
    assert rocm_build.runtime_for_host(_FakeRuntime({"ok": False, "reason": wmma})).launch()["ok"] is False


@pytest.mark.parametrize("text", [
    "exact gfx1151 spectral reverse package is unavailable",
    "gfx1151 native HSACO module load failed",
    "gfx1151 streaming STFT physical package is unavailable",
    "solver IFT package is verified for gfx1151, not gfx1201",
    "attention backward requires its exact owning ROCm device",
])
def test_gfx1151_owned_refusals_are_skips_only_off_gfx1151(text):
    assert rocm_build.refused_for_host_arch(text, "gfx1201")
    assert not rocm_build.refused_for_host_arch(text, "gfx1151")


def test_compiled_unavailable_by_type_is_a_skip_only_off_gfx11():
    class _RocmCompiledUnavailable(Exception):
        pass

    exc = _RocmCompiledUnavailable("rocm f32 GEMM lane unavailable — no chunked SSD")
    assert rocm_build.refused_by_type_for_host_arch(exc, "gfx1201")
    assert not rocm_build.refused_by_type_for_host_arch(exc, "gfx1151")
    assert not rocm_build.refused_by_type_for_host_arch(ValueError("x"), "gfx1201")
