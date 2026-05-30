"""Regression coverage for the reworked examples/advanced/Diffusion_LLM MDLM
GPU-denoise demo.

The torch-dependent `tessera_diffusion_llm` package stays a research sketch; this
covers the new standalone `gpu_denoise.py` demo that drives the Apple GPU
backbone (bmm / rowop) + the Gumbel sampler through an MDLM unmasking loop,
validated against numpy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_EX = Path(__file__).resolve().parents[2] / "examples" / "advanced" / "Diffusion_LLM"


@pytest.fixture(scope="module")
def demo():
    if not (_EX / "gpu_denoise.py").exists():
        pytest.skip("Diffusion_LLM gpu_denoise demo not present")
    if str(_EX) not in sys.path:
        sys.path.insert(0, str(_EX))
    import gpu_denoise  # noqa: E402
    return gpu_denoise


def test_mdlm_demo_validates(demo):
    s = demo.run_mdlm_demo(demo.tiny_diffusion_config())
    assert s.all_unmasked
    assert s.tokens_in_range
    assert s.gpu_backbone_matches_numpy
    assert s.gpu_sampler_matches_numpy
    assert s.deterministic
    assert s.backend in ("metal", "numpy")
    assert s.steps >= 1


def test_mdlm_deterministic(demo):
    cfg = demo.tiny_diffusion_config()
    a = demo.run_mdlm_demo(cfg)
    b = demo.run_mdlm_demo(cfg)
    assert a.deterministic and b.deterministic
