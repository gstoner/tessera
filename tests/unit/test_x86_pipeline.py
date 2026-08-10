"""Typed x86 family-plugin configuration and compatibility-retirement gates."""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera.compiler.pipeline_registry import pipeline_lookup
from tessera.compiler.x86_pipeline import (
    FAMILY_PLUGINS,
    X86ExecutablePipeline,
)


REPO = Path(__file__).resolve().parents[2]


def test_x86_family_plugins_are_cross_registry_total() -> None:
    spec = pipeline_lookup("tessera-x86-executable")
    assert spec is not None
    assert set(FAMILY_PLUGINS) == set(dict(spec.family_plugins))


def test_x86_configuration_is_typed_and_fail_closed() -> None:
    config = X86ExecutablePipeline(family="spectral_backward")
    pipeline = config.pass_pipeline()
    assert pipeline.startswith("builtin.module(tessera-x86-executable{")
    assert "family=spectral_backward" in pipeline
    assert "arch=x86_64_avx512" in pipeline
    assert "tessera-tile-to-x86" not in pipeline

    with pytest.raises(ValueError, match="unknown x86 family plugin"):
        X86ExecutablePipeline(family="generic")
    with pytest.raises(ValueError, match="no family-plugin profile"):
        X86ExecutablePipeline(family="matmul", architecture="x86_64_v3")
    with pytest.raises(ValueError, match="softmax/reduction"):
        X86ExecutablePipeline(family="matmul", architecture="x86_64_base")


def test_x86_native_packager_has_no_generic_pass_option_escape_hatch() -> None:
    source = (REPO / "python/tessera/compiler/x86_native.py").read_text()
    assert '"--tessera-tile-to-x86=' not in source
    assert "X86ExecutablePipeline(" in source
