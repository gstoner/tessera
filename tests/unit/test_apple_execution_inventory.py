"""Drift and taxonomy guards for the Apple execution inventory."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "audit" / "generated" / "apple_execution_inventory.md"
CSV = ROOT / "docs" / "audit" / "generated" / "apple_execution_inventory.csv"


def test_inventory_is_generated_and_current() -> None:
    from tessera.compiler.apple_execution_inventory import render_csv, render_markdown

    assert DOC.read_text(encoding="utf-8") == render_markdown()
    assert CSV.read_text(encoding="utf-8") == render_csv()


def test_inventory_keeps_execution_units_distinct() -> None:
    from tessera.compiler.apple_execution_inventory import rows

    inventory = rows()
    assert any(r.target == "apple_cpu" and r.compiler_form == "generic JIT dispatch"
               for r in inventory)
    assert any(r.target == "apple_cpu" and r.compiler_form == "Apple Value Target-IR cpu.call"
               for r in inventory)
    package = [r for r in inventory if r.unit == "subgraph"]
    assert package
    assert any(r.scope.startswith("AOT package") for r in package)
    assert any(
        r.compiler_form == "package recognition + evidence-gated JIT auto selection"
        and "Value Target-IR package_call remains gated" in r.scope
        for r in package
    )
