"""``tessera-translate`` CLI scaffold tests.

Locks the Python-side surface for ``tools/tessera-translate/``
(closes the M5 follow-up "tessera-translate directory is empty").
The C++ MLIR-side ``tessera-translate`` binary is still gated on
``tessera-opt`` building against MLIR 21; the Python CLI surfaces
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
    for sub in ("stablehlo", "gguf", "safetensors", "info"):
        assert sub in out


def test_each_subcommand_has_required_in_flag(capsys) -> None:
    """Every subcommand reads from ``--in``; missing it must fail
    with argparse's standard error message."""
    for sub in ("stablehlo", "gguf", "safetensors", "info"):
        with pytest.raises(SystemExit):
            translate.main([sub])


def test_translate_module_documents_cxx_gating() -> None:
    """The Python CLI is honest that the C++ side isn't here yet."""
    assert "MLIR 21" in (translate.__doc__ or "")
