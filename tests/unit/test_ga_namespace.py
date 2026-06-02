"""GA0 acceptance: `from tessera import ga` resolves and exposes __version__.

Sprint: GA0 (scope lock + namespace).
Roadmap: docs/audit/domain/DOMAIN_AUDIT.md
Scope lock: docs/audit/domain/DOMAIN_AUDIT.md
"""

from __future__ import annotations


def test_ga_namespace_imports() -> None:
    from tessera import ga

    assert ga is not None


def test_ga_namespace_exposes_version() -> None:
    from tessera import ga

    assert isinstance(ga.__version__, str)
    # Version stamp advances with each GA sprint (ga0, ga1, ...).
    assert ga.__version__.startswith("0.0.0-ga")


def test_ga_namespace_module_path() -> None:
    """GA namespace must live under python/tessera/ga/, not as a stray attribute."""
    from tessera import ga

    assert ga.__name__ == "tessera.ga"
    assert ga.__package__ == "tessera.ga"
