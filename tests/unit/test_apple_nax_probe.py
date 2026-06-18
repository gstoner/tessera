"""Apple NAX availability + feature-probe vocabulary (MLX-mined)."""

from __future__ import annotations

import pytest

from tessera.compiler.apple_target import (
    APPLE_FEATURE_PROBES,
    AppleProbeKind,
    apple_probes_by_kind,
    nax_available,
)


@pytest.mark.parametrize("macos,arch,gen,want", [
    ((26, 2), "g", 17, True),    # macOS 26.2, non-p arch, gen 17 → NAX
    ((26, 2), "p", 18, True),    # p arch needs gen 18
    ((26, 2), "p", 17, False),   # p arch, gen 17 → no NAX
    ((26, 2), "g", 16, False),   # gen too low
    ((26, 1), "g", 18, False),   # macOS too old (< 26.2)
    ((27, 0), "d", 20, True),    # newer everything
])
def test_nax_available_grounded_gate(macos, arch, gen, want):
    assert nax_available(macos, arch, gen) is want


def test_this_dev_mac_has_no_nax():
    # M1 Max (Apple7) on macOS 26.5.1 — arch gen well below the NAX threshold.
    assert nax_available((26, 5), "g", 7) is False


def test_probe_vocabulary_splits_compile_required_vs_runtime():
    names = {p.name for p in APPLE_FEATURE_PROBES}
    assert {"metal_language_version", "apple_gpu_family",
            "os_availability", "arch_generation", "nax_available"} <= names
    compile_required = {p.name for p in apple_probes_by_kind(AppleProbeKind.COMPILE_REQUIRED)}
    runtime_observed = {p.name for p in apple_probes_by_kind(AppleProbeKind.RUNTIME_OBSERVED)}
    # NAX + OS + arch-gen are observed at runtime; lang version + family are compile-required.
    assert "nax_available" in runtime_observed
    assert "metal_language_version" in compile_required
    assert compile_required.isdisjoint(runtime_observed)
