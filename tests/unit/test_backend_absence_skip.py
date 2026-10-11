"""The backend-absent skip converter must never hide a defect on a host that has the backend."""

from pathlib import Path

import pytest

from tests._support import backend_absence, compiler_tool

NVIDIA_DIALECT = (
    "scheduled compiler boundary --tessera-schedule-to-tile failed: <stdin>:4:10: error: "
    "SM120 scheduled matmul requires the registered NVIDIA Target IR dialect"
)
ROCM_PASS = "tessera-opt: Unknown command line argument '--lower-tile-to-rocm=arch=gfx1201'."
ROCM_EXEC = "error: 'tessera-rocm-executable' does not refer to a registered pass"
NVIDIA_TOOLS = "NVIDIA native packaging tools unavailable: ['tessera-nvidia-opt', 'ptxas']"
TOOL = Path("/fake/tessera-opt")


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.setattr(backend_absence, "_driver", lambda: TOOL)
    return monkeypatch


def probes(monkeypatch, *, dialects=(), passes=()):
    monkeypatch.setattr(compiler_tool, "registered_dialects", lambda tool: frozenset(dialects))
    monkeypatch.setattr(compiler_tool, "registered_passes", lambda tool: frozenset(passes))


def test_nvidia_dialect_refusal_skips_only_when_dialect_is_unregistered(driver):
    probes(driver, dialects={"tessera"})
    assert "tessera_nvidia dialect not registered" in backend_absence.backend_absence_reason(NVIDIA_DIALECT)
    probes(driver, dialects={"tessera", "tessera_nvidia"})
    assert backend_absence.backend_absence_reason(NVIDIA_DIALECT) is None


@pytest.mark.parametrize("text,name", [(ROCM_PASS, "lower-tile-to-rocm"), (ROCM_EXEC, "tessera-rocm-executable")])
def test_rocm_pass_refusals_skip_only_when_pass_is_unregistered(driver, text, name):
    probes(driver, passes={"tessera-autodiff"})
    assert name in backend_absence.backend_absence_reason(text)
    probes(driver, passes={name})
    assert backend_absence.backend_absence_reason(text) is None


def test_nvidia_tool_message_skips_only_when_every_named_tool_is_absent(driver):
    driver.setattr(backend_absence, "_tool_is_absent", lambda name: True)
    assert "not installed" in backend_absence.backend_absence_reason(NVIDIA_TOOLS)
    driver.setattr(backend_absence, "_tool_is_absent", lambda name: name == "ptxas")
    assert backend_absence.backend_absence_reason(NVIDIA_TOOLS) is None


def test_unrelated_failures_are_never_converted(driver):
    probes(driver)
    for text in ("assert 1 == 2", "tessera-opt crashed: Stack dump", "KeyError: 'TESSERA_OPT'"):
        assert backend_absence.backend_absence_reason(text) is None
