"""``tessera-translate`` CLI scaffold tests.

Locks the Python-side surface for ``tools/tessera-translate/``
(closes the M5 follow-up "tessera-translate directory is empty").
The C++ MLIR-side ``tessera-translate`` binary is still gated on
``tessera-opt`` building against MLIR 23; the Python CLI surfaces
the subset of inter-IR translation that doesn't need it.
"""

from __future__ import annotations

import argparse

import pytest

from tessera.cli import translate


def test_main_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit) as excinfo:
        translate.main([])
    # argparse exits with 2 on usage error.
    assert excinfo.value.code == 2


def test_subcommands_are_canonical_set(capsys) -> None:
    """The subcommand surface is a deliberate decision — changing
    it requires updating both this test and the README in
    ``tools/tessera-translate/``."""
    with pytest.raises(SystemExit):
        translate.main(["--help"])
    out = capsys.readouterr().out
    for sub in ("stablehlo", "gguf", "safetensors", "info", "mlir"):
        assert sub in out


def test_each_subcommand_has_required_in_flag(capsys) -> None:
    """Every subcommand reads from ``--in``; missing it must fail
    with argparse's standard error message."""
    for sub in ("stablehlo", "gguf", "safetensors", "info"):
        with pytest.raises(SystemExit):
            translate.main([sub])


def test_mlir_subcommand_dispatches_to_cxx_binary(monkeypatch, tmp_path) -> None:
    """`tessera-translate mlir` is a pass-through to the C++
    binary.  Test with a fake stand-in to keep this test fast."""
    fake_bin = tmp_path / "fake-translate"
    fake_bin.write_text(
        "#!/bin/sh\necho 'forwarded:' \"$@\"\nexit 0\n",
        encoding="utf-8",
    )
    fake_bin.chmod(0o755)
    monkeypatch.setattr(
        translate, "_find_tessera_translate_mlir", lambda: str(fake_bin),
    )
    # Also patch within the namespace used by the passthrough closure
    # (it captures the symbol at import time on first call).
    rc = translate.main(["mlir", "--mlir-to-llvmir", "x.mlir"])
    assert rc == 0


def test_mlir_subcommand_reports_missing_binary(monkeypatch, capsys) -> None:
    """When the C++ binary isn't built, the Python CLI returns a
    clean exit code and a diagnostic — *not* a Python traceback."""
    monkeypatch.setattr(
        translate, "_find_tessera_translate_mlir", lambda: None,
    )
    rc = translate.main(["mlir", "--mlir-to-llvmir", "x.mlir"])
    assert rc == 127
    err = capsys.readouterr().err
    assert "tessera-translate-mlir" in err
    assert "cmake --build" in err


def test_translate_module_documents_cxx_companion() -> None:
    """The Python CLI documents the C++ companion binary."""
    doc = translate.__doc__ or ""
    assert "tessera-translate-mlir" in doc
