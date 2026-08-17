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
    assert all(
        plugin.async_role_policy == "no_async_noop"
        for plugin in FAMILY_PLUGINS.values()
    )


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


def test_tile_to_x86_declares_optional_target_dialect_dependency() -> None:
    """The pass manager, never ``runOnOperation``, must load Target IR."""
    source = (REPO / "src/transforms/lib/TileToX86Pass.cpp").read_text()
    transforms_cmake = (REPO / "src/transforms/lib/CMakeLists.txt").read_text()
    source_cmake = (REPO / "src/CMakeLists.txt").read_text()

    assert 'getOrLoadDialect("tessera_x86")' not in source
    assert "void getDependentDialects(DialectRegistry &registry)" in source
    assert "registry.insert<mlir::tessera_x86::TesseraX86Dialect>()" in source
    assert "#ifdef TESSERA_HAS_X86_TARGET_IR" in source
    assert "TESSERA_BUILD_X86_BACKEND=ON" in source

    assert "TESSERA_HAS_X86_TARGET_IR=1" in transforms_cmake
    assert (
        "target_link_libraries(TesseraPasses PRIVATE TesseraX86IR)"
        in transforms_cmake
    )
    assert source_cmake.index(
        "add_subdirectory(compiler/codegen/tessera_x86_backend/lib/IR)"
    ) < source_cmake.index("add_subdirectory(transforms)")


def test_composite_pipeline_has_one_schedule_dialect_authority() -> None:
    """Distribution lowering must not shadow the production ODS dialect."""
    source = (
        REPO / "src/transforms/lib/DistributionLoweringPass.cpp"
    ).read_text()
    transforms_cmake = (REPO / "src/transforms/lib/CMakeLists.txt").read_text()
    pm_cmake = (
        REPO / "src/compiler/programming_model/CMakeLists.txt"
    ).read_text()

    assert "class ScheduleDialect" not in source
    assert '#include "tessera/ProgrammingModel/ScheduleDialect.h"' in source
    assert "registry.insert<tessera::schedule::ScheduleDialect>()" in source
    assert "TesseraScheduleIR" in transforms_cmake
    assert "add_library(TesseraScheduleIR" in pm_cmake
