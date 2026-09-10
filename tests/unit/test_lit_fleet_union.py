import json
from pathlib import Path

from scripts.check_lit_fleet_union import active_fixtures, reconcile


def _report(path: Path, rows: list[tuple[str, str]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "tests": [
                    {"name": f"Tessera-IR :: {name}", "code": code}
                    for name, code in rows
                ]
            }
        )
    )
    return path


def test_union_accepts_a_fixture_that_passes_only_in_its_owning_lane(
    tmp_path: Path,
) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "core.mlir").write_text("// RUN: true\n")
    (fixtures / "apple.mlir").write_text("// RUN: true\n")
    core = _report(
        tmp_path / "core.json",
        [("core.mlir", "PASS"), ("apple.mlir", "UNSUPPORTED")],
    )
    apple = _report(
        tmp_path / "apple.json",
        [("core.mlir", "PASS"), ("apple.mlir", "PASS")],
    )

    result = reconcile(fixtures, [core, apple])

    assert result["covered"] == 2
    assert result["uncovered"] == []
    assert result["unexpected"] == []


def test_union_reports_uncovered_and_unexpected_results(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "missing.mlir").write_text("// RUN: true\n")
    (fixtures / "broken.mlir").write_text("// RUN: true\n")
    (fixtures / "data.mlir").write_text("// UNSUPPORTED: true\n")
    report = _report(tmp_path / "core.json", [("broken.mlir", "FAIL")])

    assert active_fixtures(fixtures) == {"missing.mlir", "broken.mlir"}
    result = reconcile(fixtures, [report])
    assert result["uncovered"] == ["broken.mlir", "missing.mlir"]
    assert result["unexpected"] == [
        {"lane": "core.json", "fixture": "broken.mlir", "code": "FAIL"}
    ]
