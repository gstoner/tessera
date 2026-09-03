from __future__ import annotations

from pathlib import Path

from tests._support.repo_scan import iter_repo_files


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
    # `iter_repo_files` rather than `ROOT.rglob`: a nested git worktree is a
    # complete second copy of the repo, so a plain walk finds this file twice
    # under two paths. This assertion compares a set of PATHS, so a duplicate
    # is a new element and the test failed on whichever machine happened to
    # have a worktree open -- and nowhere else.
    implementation_files = {
        path.relative_to(ROOT).as_posix()
        for path in iter_repo_files(ROOT, suffixes={".cpp", ".h", ".td"})
        if "TileBufferRefAttr" in path.read_text()
    }
    assert implementation_files == {
        "src/compiler/ir/TileDialect.cpp",
    }
