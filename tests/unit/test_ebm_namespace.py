"""EBM0 acceptance: `from tessera import ebm` resolves and exposes __version__.

Sprint: EBM0 (scope lock + archived EBT revival).
Roadmap: docs/audit/domain/DOMAIN_AUDIT.md
Scope lock: docs/audit/domain/DOMAIN_AUDIT.md
Spec: docs/spec/EBM_SPEC.md
"""

from __future__ import annotations

import pathlib


def test_ebm_namespace_imports() -> None:
    from tessera import ebm

    assert ebm is not None


def test_ebm_namespace_exposes_version() -> None:
    from tessera import ebm

    assert isinstance(ebm.__version__, str)
    # Version stamp advances with each EBM sprint (ebm0, ebm1, ...).
    assert ebm.__version__.startswith("0.0.0-ebm")


def test_ebm_namespace_module_path() -> None:
    """EBM namespace must live under python/tessera/ebm/, not as a stray attribute."""
    from tessera import ebm

    assert ebm.__name__ == "tessera.ebm"
    assert ebm.__package__ == "tessera.ebm"


def test_ebm_spec_doc_exists() -> None:
    """EBM_SPEC.md is the normative spec; EBM0 ships it alongside the namespace."""
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    spec = repo_root / "docs" / "spec" / "EBM_SPEC.md"
    assert spec.is_file(), f"missing normative spec at {spec}"
    body = spec.read_text(encoding="utf-8")
    # Quick provenance + primitive-surface contract checks; not a full lint.
    assert "Adapted from" in body
    for primitive in (
        "ebm.energy",
        "ebm.inner_step",
        "ebm.langevin_step",
        "ebm.self_verify",
        "ebm.decode_init",
    ):
        assert primitive in body, f"spec missing primitive {primitive}"
