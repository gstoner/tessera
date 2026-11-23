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


NVIDIA_OPTION = ("scheduled compiler boundary --lower-tile-to-nvidia=sm=120 failed: "
                 "tessera-opt: Unknown command line argument '--lower-tile-to-nvidia=sm=120'.")
NVIDIA_PIPELINE = ("MLIR Textual PassPipeline Parser:1:1: error: "
                   "'tessera-lower-to-nvidia-sm120' does not refer to a registered pass or pass pipeline")
ROCDL_OPTION = "tessera-opt: Unknown command line argument '--lower-tessera-target-to-rocdl'."


@pytest.mark.parametrize("text,name,backend", [
    (NVIDIA_OPTION, "lower-tile-to-nvidia", "NVIDIA"),
    (NVIDIA_PIPELINE, "tessera-lower-to-nvidia-sm120", "NVIDIA"),
    (ROCDL_OPTION, "lower-tessera-target-to-rocdl", "ROCm"),
])
def test_backend_pass_refusals_skip_only_when_unregistered(driver, text, name, backend):
    probes(driver, passes={"tessera-autodiff"})
    reason = backend_absence.backend_absence_reason(text)
    assert reason and name in reason and f"no {backend} backend" in reason
    probes(driver, passes={name})  # a host that registers the pass: the failure is a defect
    assert backend_absence.backend_absence_reason(text) is None


@pytest.mark.parametrize("name", ["tessera-graph-to-schedule", "canonicalize", "tessera-autodiff"])
def test_missing_generic_pass_is_never_converted(driver, name):
    """A generic pass missing from the driver is a regression, not an absent backend."""
    probes(driver)
    for text in (f"Unknown command line argument '--{name}'.",
                 f"'{name}' does not refer to a registered pass"):
        assert backend_absence.backend_absence_reason(text) is None


def test_amd_clang_refusal_skips_only_without_a_rocm_toolkit(driver):
    text = ("RuntimeError: ROCm native packaging requires AMD clang to identify "
            "OCML/OCKL/OCLC; set TESSERA_ROCM_CLANG or ROCM_PATH")
    probes(driver)
    driver.setattr(backend_absence, "_amd_clang_is_absent", lambda: True)
    assert "no ROCm toolkit" in backend_absence.backend_absence_reason(text)
    driver.setattr(backend_absence, "_amd_clang_is_absent", lambda: False)
    assert backend_absence.backend_absence_reason(text) is None


def test_hipcc_gate_skips_only_without_hipcc(driver):
    text = "first failing gate `toolchain` — hipcc not on PATH (ROCm 10.0 not installed)."
    probes(driver)
    driver.setattr(backend_absence, "_hipcc_is_absent", lambda: True)
    assert "hipcc not found" in backend_absence.backend_absence_reason(text)
    driver.setattr(backend_absence, "_hipcc_is_absent", lambda: False)
    assert backend_absence.backend_absence_reason(text) is None


def test_unloaded_backend_dialect_skips_only_when_unregistered(driver):
    text = ("<stdin>:6:2: error: Dialect `tessera_nvidia' not found for custom op "
            "'tessera_nvidia.program_invoke'")
    probes(driver, dialects={"tessera"})
    assert "tessera_nvidia dialect not registered" in backend_absence.backend_absence_reason(text)
    probes(driver, dialects={"tessera", "tessera_nvidia"})
    assert backend_absence.backend_absence_reason(text) is None


FAILED_ROCM_COMMAND = ("Command '['/x/tessera-opt', '--tessera-graph-to-schedule', '--tessera-schedule-to-tile', "
                       "'--lower-tile-to-rocm=arch=gfx1201']' returned non-zero exit status 1.")


def test_failed_command_skips_only_for_an_unregistered_backend_pass(driver):
    probes(driver, passes={"tessera-graph-to-schedule", "tessera-schedule-to-tile"})
    assert "lower-tile-to-rocm" in backend_absence.backend_absence_reason(FAILED_ROCM_COMMAND)
    probes(driver, passes={"tessera-graph-to-schedule", "tessera-schedule-to-tile", "lower-tile-to-rocm"})
    assert backend_absence.backend_absence_reason(FAILED_ROCM_COMMAND) is None


def test_failed_command_with_only_generic_passes_is_a_defect(driver):
    text = ("Command '['/x/tessera-opt', '--tessera-graph-to-schedule']' "
            "returned non-zero exit status 1.")
    probes(driver)
    assert backend_absence.backend_absence_reason(text) is None


def test_apple_gap_map_names_real_files_and_tracked_items():
    """Decision #29: the tag is only worth reading if both ends exist."""
    root = Path(__file__).resolve().parents[2]
    todo = (root / "docs/audit/backend/apple/todo.md").read_text(encoding="utf-8")
    for name, item in backend_absence.APPLE_NEEDS_IMPLEMENTATION.items():
        assert (root / "tests/unit" / name).is_file(), name
        assert f"### {item} " in todo, f"{item} has no entry in the Apple todo"


def test_apple_gap_note_only_tags_mapped_files_on_a_mac(monkeypatch):
    import platform
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    assert "APPLE-NTBI-2" in backend_absence.apple_gap_note("/r/tests/unit/test_native_tensor_shape_plan.py")
    assert backend_absence.apple_gap_note("/r/tests/unit/test_rocm_fp8_cast_lowering.py") == ""
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    assert backend_absence.apple_gap_note("/r/tests/unit/test_native_tensor_shape_plan.py") == ""
