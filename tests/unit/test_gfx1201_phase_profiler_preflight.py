"""Fail-closed admission tests for the non-instrumenting phase-probe preflight."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.rocm import probe_gfx1201_phase_profiler as probe


def test_missing_kfd_refuses_phase_attribution(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(probe.rt, "_rocm_live_arch", lambda: "gfx1201")
    monkeypatch.setattr(
        probe.rt, "_load_hip_for_launch", lambda: SimpleNamespace(hipInit=lambda _: 0),
    )
    monkeypatch.setattr(probe.shutil, "which", lambda name: f"/bin/{name}")
    from benchmarks.rocm import benchmark_gfx1201_mxfp4_production as base

    monkeypatch.setattr(base, "_selected_device_name", lambda _: "AMD Radeon RX 9070 XT")
    packet = probe.probe(kfd=tmp_path / "missing-kfd")
    assert packet["architecture"] == "gfx1201"
    assert packet["device"] == "AMD Radeon RX 9070 XT"
    assert packet["refusal_reasons"] == ["kfd_device_missing"]
    assert packet["isa_preserving_phase_probe_available"] is False
    assert packet["cross_cu_clock_validated"] is False
    assert packet["phase_attribution_admissible"] is False
    assert packet["promotion_eligible"] is False


def test_available_profiler_still_refuses_without_clock_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(probe.rt, "_rocm_live_arch", lambda: "gfx1201")
    monkeypatch.setattr(
        probe.rt, "_load_hip_for_launch", lambda: SimpleNamespace(hipInit=lambda _: 0),
    )
    monkeypatch.setattr(probe.shutil, "which", lambda name: f"/bin/{name}")
    from benchmarks.rocm import benchmark_gfx1201_mxfp4_production as base

    monkeypatch.setattr(base, "_selected_device_name", lambda _: "AMD Radeon RX 9070 XT")
    kfd = tmp_path / "kfd"
    kfd.touch()
    monkeypatch.setattr(Path, "is_char_device", lambda _: True)
    monkeypatch.setattr(
        probe.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="AMD Radeon RX 9070 XT", stderr="",
        ),
    )
    packet = probe.probe(kfd=kfd)
    assert packet["isa_preserving_phase_probe_available"] is True
    assert packet["refusal_reasons"] == ["pc_samples_and_clock_validation_not_recorded"]
    assert packet["phase_attribution_admissible"] is False
