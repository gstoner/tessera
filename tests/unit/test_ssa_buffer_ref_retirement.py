from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_active_compiler_passes_have_no_buffer_ref_compatibility_reader() -> None:
    roots = (
        ROOT / "src" / "transforms",
        ROOT / "src" / "compiler" / "codegen" / "Tessera_ROCM_Backend" / "lib",
    )
    readers: list[str] = []
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix not in {".cpp", ".h", ".td"}:
                continue
            if "TileBufferRefAttr" in path.read_text():
                readers.append(str(path.relative_to(ROOT)))
    assert readers == []


def test_rocm_fixtures_do_not_supply_name_based_buffer_identity() -> None:
    fixture_root = (
        ROOT / "src" / "compiler" / "codegen"
        / "Tessera_ROCM_Backend" / "test"
    )
    producers: list[str] = []
    for path in fixture_root.rglob("*.mlir"):
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if "#tile.buffer_ref" in stripped and not stripped.startswith("//"):
                producers.append(f"{path.relative_to(ROOT)}:{line_number}")
    assert producers == []


def test_deprecated_buffer_ref_is_parser_only() -> None:
    implementation_files = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.suffix in {".cpp", ".h", ".td"}
        and "TileBufferRefAttr" in path.read_text()
    }
    assert implementation_files == {
        "src/compiler/ir/TileDialect.cpp",
    }
