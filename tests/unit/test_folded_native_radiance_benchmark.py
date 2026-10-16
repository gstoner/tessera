"""Independent reference metadata must be checked without probing hardware."""
import hashlib
import sys

import pytest

from benchmarks.rocm import record_gfx1201_folded_native_package as recorder


@pytest.mark.parametrize("case", ["module_only", "revision_only", "missing", "layout"])
def test_reference_cli_contract_precedes_device_probe(tmp_path, monkeypatch, case):
    module = tmp_path / "reference.so"
    module.write_bytes(b"reference metadata only")
    monkeypatch.setenv("RADIANCE_MXFP4_WPERM", "1")
    monkeypatch.setattr(recorder.rt, "_rocm_live_arch",
                        lambda: pytest.fail("invalid reference contract probed a GPU"))
    args = ["record", "--tessera-opt", "/unused/compiler", "--llvm-bin", "/unused/llvm",
            "--output", str(tmp_path / "packet.json")]
    expected = ValueError
    if case == "module_only":
        args += ["--radiance-module", str(module)]
    elif case == "revision_only":
        args += ["--radiance-revision", "declared"]
    else:
        if case == "missing":
            module.unlink()
            expected = FileNotFoundError
        else:
            monkeypatch.setenv("RADIANCE_MXFP4_WPERM", "0")
        args += ["--radiance-module", str(module), "--radiance-revision", "declared"]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(expected, match="Radiance"):
        recorder.main()


def test_reference_identity_uses_actual_file_bytes(tmp_path, monkeypatch):
    module = tmp_path / "reference.so"
    data = b"independent reference first image"
    module.write_bytes(data)
    monkeypatch.setenv("RADIANCE_MXFP4_WPERM", "1")
    monkeypatch.setattr(recorder.rt, "_rocm_live_arch",
                        lambda: pytest.fail("metadata collection probed a GPU"))
    first = recorder.radiance_metadata(module, "declared")
    assert first["binary_sha256"] == hashlib.sha256(data).hexdigest()
    assert first["declared_revision"] == "declared"
    module.write_bytes(b"independent reference changed image")
    second = recorder.radiance_metadata(module, "declared")
    assert second["binary_sha256"] != first["binary_sha256"]
    assert recorder.radiance_metadata(None, None) is None
