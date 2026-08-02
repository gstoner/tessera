"""X86-3 local AMX proof-gate contract."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "scripts/run_x86_amx_release_gate.sh"
DEVICE_TEST = ROOT / "tests/device/x86/test_amx_int8_gemm.py"
OLD_UNIT_TEST = ROOT / "tests/unit/test_codereview_red_regressions.py"


def test_amx_release_gate_is_local_exact_device_proof() -> None:
    text = GATE.read_text(encoding="utf-8")
    assert "local exact-device correctness gate" in text
    assert "Intel AMX exact-device proof bundle" in text
    assert "not GitHub-hosted or" in text
    assert "self-hosted-runner evidence" in text
    assert "machine-identity.txt" in text
    assert "status=success" in text
    assert "status=failed" in text
    assert "flock -n 9" in text


def test_amx_release_gate_owns_repeated_serial_correctness() -> None:
    text = GATE.read_text(encoding="utf-8")
    assert "tests/device/x86" in text
    assert "hardware_amx and not performance" in text
    assert "--collect-only" in text
    assert "device-correctness-1.xml" in text
    assert "device-correctness-2.xml" in text
    assert text.count("-n 0") == 2
    assert "amx_tile" in text
    assert "amx_int8" in text


def test_native_amx_regression_has_device_ownership_and_crash_isolation() -> None:
    device_text = DEVICE_TEST.read_text(encoding="utf-8")
    unit_text = OLD_UNIT_TEST.read_text(encoding="utf-8")
    hardware_marker = "pytest" + ".mark.hardware_amx"
    integration_marker = "pytest" + ".mark.integration"
    assert hardware_marker in device_text
    assert integration_marker in device_text
    assert "test_amx_int8_gemm_does_not_truncate_large_k" in device_text
    assert '[sys.executable, "-c", program' in device_text
    assert "terminated by signal" in device_text
    assert "test_amx_int8_gemm_does_not_truncate_large_k" not in unit_text
