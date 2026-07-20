"""Offline source-discovery guards for the structured RDNA ISA archive."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "docs/reference/isa/rdna/tools/build_archive.py"
_SPEC = importlib.util.spec_from_file_location("rdna_build_archive", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
archive = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = archive
_SPEC.loader.exec_module(archive)


def test_source_discovery_prefers_the_first_directory(tmp_path) -> None:
    preferred = tmp_path / "preferred"
    fallback = tmp_path / "fallback"
    preferred.mkdir()
    fallback.mkdir()
    (preferred / "rdna.pdf").write_bytes(b"preferred")
    (fallback / "rdna.pdf").write_bytes(b"fallback")
    assert archive.find_source_pdf("rdna.pdf", (preferred, fallback)) == (
        preferred / "rdna.pdf"
    )


def test_source_discovery_falls_back_per_document(tmp_path) -> None:
    preferred = tmp_path / "preferred"
    fallback = tmp_path / "fallback"
    preferred.mkdir()
    fallback.mkdir()
    (fallback / "mes.pdf").write_bytes(b"mes")
    assert archive.find_source_pdf("mes.pdf", (preferred, fallback)) == (
        fallback / "mes.pdf"
    )


def test_source_discovery_names_every_searched_directory(tmp_path) -> None:
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    with pytest.raises(FileNotFoundError) as error:
        archive.find_source_pdf("missing.pdf", (one, two))
    assert str(one) in str(error.value)
    assert str(two) in str(error.value)
