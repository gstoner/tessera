"""Fleet-wide drift gate for the generated-doc registry.

`tessera.compiler.generated_docs` is the single source of truth for every
generated audit dashboard. This test:

  * renders every doc (catches a generator that raises),
  * asserts the canonical artifact (CSV when present, else Markdown) is
    in sync for every gated doc — one fleet-wide drift gate replacing the
    former scatter of per-doc byte-exact checks,
  * asserts no generated Markdown under `docs/audit/generated/` is
    *un*registered, so a new dashboard can't land outside the registry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera.compiler import generated_docs as gd


REPO_ROOT = Path(__file__).resolve().parents[2]
GEN_DIR = REPO_ROOT / "docs" / "audit" / "generated"


def test_registry_names_unique() -> None:
    names = [d.name for d in gd.REGISTRY]
    assert len(names) == len(set(names)), "duplicate doc name in registry"


@pytest.mark.parametrize("doc", gd.REGISTRY, ids=lambda d: d.name)
@pytest.mark.slow
def test_doc_renders(doc: gd.GeneratedDoc) -> None:
    """Every render callable must produce non-empty text without error."""
    md = doc.render_md()
    assert isinstance(md, str) and md.strip(), f"{doc.name} render_md is empty"
    if doc.render_csv is not None:
        csv = doc.render_csv()
        assert isinstance(csv, str) and csv.strip(), f"{doc.name} render_csv is empty"


@pytest.mark.parametrize(
    "doc", [d for d in gd.REGISTRY if d.gated], ids=lambda d: d.name
)
@pytest.mark.slow
def test_doc_in_sync(doc: gd.GeneratedDoc) -> None:
    """The canonical on-disk artifact must match the live render."""
    msg = gd.check(doc)
    assert msg is None, msg


def test_csv_docs_have_csv_path() -> None:
    """A doc declaring render_csv must also declare csv_path (and vice
    versa) so the canonical-artifact rule is unambiguous."""
    for d in gd.REGISTRY:
        assert (d.render_csv is None) == (d.csv_path is None), (
            f"{d.name}: render_csv and csv_path must both be set or both unset"
        )


def test_no_unregistered_generated_markdown() -> None:
    """Every fully-generated Markdown under docs/audit/generated/ must be
    in the registry.  Exceptions: CSV companions, and docs gated by their
    own bespoke mechanism (none today besides the partial-snapshot
    standalone_primitive_coverage, which lives in docs/audit/ root)."""
    registered = {d.md_path.resolve() for d in gd.REGISTRY}
    on_disk = {p.resolve() for p in GEN_DIR.glob("*.md")}
    orphans = sorted(p.name for p in (on_disk - registered))
    assert not orphans, (
        f"generated Markdown not in the registry: {orphans}. "
        f"Register them in tessera.compiler.generated_docs.REGISTRY."
    )
