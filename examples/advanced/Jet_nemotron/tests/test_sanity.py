"""Sanity coverage for the Jet_nemotron example surface.

The previous version was a literal ``assert True`` placeholder, so
the e2e_infer.py demo's import-time crash on
``tessera_jetnemotron.transformer_block`` (a package that doesn't
exist — the supporting modules live as plain files alongside the
example) went uncaught.

This file is intentionally minimal: it validates that the example
script's import block resolves cleanly *or* skips with an
honest-reporting diagnostic naming the missing dependency
(``tessera.stdlib``, which is part of the broader Jet_nemotron
research stack that's not always co-located with this scaffold).
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest


JET_NEMOTRON_ROOT = Path(__file__).resolve().parents[1]
E2E_INFER = JET_NEMOTRON_ROOT / "examples" / "e2e_infer.py"


def test_e2e_infer_script_exists() -> None:
    """The canonical entry-point file must exist where the
    README points."""
    assert E2E_INFER.is_file(), f"missing example: {E2E_INFER}"


def test_e2e_infer_import_block_resolves_or_skips_cleanly() -> None:
    """The first few imports of ``e2e_infer.py`` must either fully
    resolve (when the Jet_nemotron support stack is on PATH) or
    fail with a specific module-not-found pointing at the missing
    upstream dep — never with a confusing
    ``tessera_jetnemotron`` ghost-package error.

    Locks the post-2026-05-19 fix that replaced the bogus
    ``from tessera_jetnemotron.transformer_block import ...`` with
    a sys.path bootstrap pointing at the actual sibling modules.
    """
    src = E2E_INFER.read_text(encoding="utf-8")
    assert "from tessera_jetnemotron" not in src, (
        "e2e_infer.py still references the non-existent "
        "`tessera_jetnemotron` package; replace with a sys.path "
        "bootstrap pointing at the sibling `transformer_block.py`."
    )
    # Try the real import chain; if downstream modules need
    # extras we don't ship (``tessera.stdlib``), report that
    # explicitly instead of pretending the example is fine.
    if str(JET_NEMOTRON_ROOT) not in sys.path:
        sys.path.insert(0, str(JET_NEMOTRON_ROOT))
    try:
        importlib.import_module("transformer_block")
    except ModuleNotFoundError as exc:
        pytest.skip(
            f"Jet_nemotron support stack incomplete on PATH: "
            f"{exc.name!r} missing.  The example is a research "
            f"scaffold; install the matching upstream stack to "
            f"exercise it."
        )
