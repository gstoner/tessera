"""Arch-4 (2026-05-22) — dialect registration manifest drift gate.

Asserts the 5 touchpoints stay consistent:

  1. Public header exists on disk.
  2. ``register_fn`` symbol appears in a sibling .cpp file.
  3. ``cmake_flag`` is referenced in tools/tessera-opt/CMakeLists.txt
     AND tools/tessera-opt/tessera-opt.cpp.
  4. ``eager_load_parent`` references a registered dialect.
  5. ``standalone_lit_parseable`` claim matches reality (False entries
     come with a documented blocker note).

When a future sprint adds a dialect: append one DialectSpec entry and
let this drift gate guide you to the 5 missing files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera.compiler.dialects_manifest import (
    REGISTERED_DIALECTS,
    DialectSpec,
    all_dialect_names,
    cpp_files_for,
    dialect_lookup,
    header_path,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TESSERA_OPT_CPP = REPO_ROOT / "tools" / "tessera-opt" / "tessera-opt.cpp"
TESSERA_OPT_CMAKE = REPO_ROOT / "tools" / "tessera-opt" / "CMakeLists.txt"


# ─────────────────────────────────────────────────────────────────────────
# Manifest structural tests
# ─────────────────────────────────────────────────────────────────────────


def test_registered_dialects_have_required_fields() -> None:
    for spec in REGISTERED_DIALECTS:
        assert spec.name, "empty dialect name"
        assert spec.target, f"empty target for {spec.name}"
        assert spec.header, f"empty header for {spec.name}"
        assert spec.register_fn, f"empty register_fn for {spec.name}"
        assert spec.sprint, f"empty sprint label for {spec.name}"


def test_registered_dialects_have_no_duplicates() -> None:
    seen: set[str] = set()
    for spec in REGISTERED_DIALECTS:
        assert spec.name not in seen, f"duplicate dialect: {spec.name}"
        seen.add(spec.name)


def test_registered_dialects_are_alphabetized() -> None:
    names = [d.name for d in REGISTERED_DIALECTS]
    assert names == sorted(names), (
        f"REGISTERED_DIALECTS must be alphabetised; out-of-order: "
        f"{[n for n, s in zip(names, sorted(names)) if n != s]}"
    )


def test_lookup_helpers_work() -> None:
    names = all_dialect_names()
    assert len(names) == len(REGISTERED_DIALECTS)
    for spec in REGISTERED_DIALECTS:
        assert dialect_lookup(spec.name) is spec
    assert dialect_lookup("never_registered") is None


# ─────────────────────────────────────────────────────────────────────────
# Touchpoint 1: public header exists on disk
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("spec", REGISTERED_DIALECTS, ids=lambda s: s.name)
def test_public_header_exists(spec: DialectSpec) -> None:
    path = header_path(spec)
    assert path.exists(), (
        f"dialect {spec.name!r} public header missing: "
        f"{path.relative_to(REPO_ROOT)}"
    )


# ─────────────────────────────────────────────────────────────────────────
# Touchpoint 2: register_fn symbol appears in matching .cpp
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("spec", REGISTERED_DIALECTS, ids=lambda s: s.name)
def test_register_fn_implementation_present(spec: DialectSpec) -> None:
    """The unqualified function name (last :: segment) must appear in
    at least one of the dialect's .cpp files."""
    short_name = spec.register_fn.rsplit("::", 1)[-1]
    cpp_files = cpp_files_for(spec)
    assert cpp_files, (
        f"no .cpp files found near {spec.header} — has the layout changed?"
    )
    found_in: list[Path] = []
    for cpp_path in cpp_files:
        try:
            text = cpp_path.read_text(errors="replace")
        except OSError:
            continue
        if short_name in text:
            found_in.append(cpp_path)
    assert found_in, (
        f"register_fn {spec.register_fn!r} (short name {short_name!r}) "
        f"not found in any .cpp under {cpp_files[0].parent.relative_to(REPO_ROOT)}"
    )


# ─────────────────────────────────────────────────────────────────────────
# Touchpoint 3: cmake_flag is referenced in tessera-opt's CMake AND .cpp
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "spec",
    [d for d in REGISTERED_DIALECTS if d.cmake_flag is not None],
    ids=lambda s: s.name,
)
def test_cmake_flag_referenced_in_tessera_opt_cmake(spec: DialectSpec) -> None:
    """The TESSERA_HAVE_* macro must appear in tools/tessera-opt/CMakeLists.txt
    so the `target_compile_definitions(... PRIVATE <FLAG>)` is wired."""
    assert TESSERA_OPT_CMAKE.exists()
    text = TESSERA_OPT_CMAKE.read_text()
    assert spec.cmake_flag in text, (
        f"cmake_flag {spec.cmake_flag!r} (for dialect {spec.name!r}) "
        f"not referenced in {TESSERA_OPT_CMAKE.relative_to(REPO_ROOT)}; "
        f"add the `if(TARGET {spec.target}) ... endif()` block."
    )


@pytest.mark.parametrize(
    "spec",
    [d for d in REGISTERED_DIALECTS if d.cmake_flag is not None],
    ids=lambda s: s.name,
)
def test_cmake_flag_referenced_in_tessera_opt_cpp(spec: DialectSpec) -> None:
    """The TESSERA_HAVE_* macro must gate the registration call in
    tools/tessera-opt/tessera-opt.cpp."""
    assert TESSERA_OPT_CPP.exists()
    text = TESSERA_OPT_CPP.read_text()
    assert f"#ifdef {spec.cmake_flag}" in text, (
        f"cmake_flag {spec.cmake_flag!r} (for dialect {spec.name!r}) "
        f"not used in #ifdef in tessera-opt.cpp — add the "
        f"`#ifdef {spec.cmake_flag} ... #endif` block that calls "
        f"`{spec.register_fn}(registry)`."
    )


@pytest.mark.parametrize(
    "spec",
    [d for d in REGISTERED_DIALECTS if d.cmake_flag is not None],
    ids=lambda s: s.name,
)
def test_register_fn_called_in_tessera_opt_cpp(spec: DialectSpec) -> None:
    """The fully-qualified register_fn must be called in tessera-opt.cpp
    so the dialect actually lands in the registry at startup."""
    assert TESSERA_OPT_CPP.exists()
    text = TESSERA_OPT_CPP.read_text()
    assert spec.register_fn in text, (
        f"register_fn {spec.register_fn!r} not called in tessera-opt.cpp"
    )


# ─────────────────────────────────────────────────────────────────────────
# Touchpoint 4: eager_load_parent must be a registered dialect
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "spec",
    [d for d in REGISTERED_DIALECTS if d.eager_load_parent is not None],
    ids=lambda s: s.name,
)
def test_eager_load_parent_is_registered(spec: DialectSpec) -> None:
    parent = dialect_lookup(spec.eager_load_parent)
    assert parent is not None, (
        f"dialect {spec.name!r} declares eager_load_parent="
        f"{spec.eager_load_parent!r} but that's not a registered "
        f"dialect — fix the parent name or register the parent first"
    )


# ─────────────────────────────────────────────────────────────────────────
# Touchpoint 5: standalone_lit_parseable=False entries carry documented
# blocker notes
# ─────────────────────────────────────────────────────────────────────────


def test_unparseable_dialects_have_doc_note() -> None:
    """When standalone_lit_parseable=False, the dialect's .cpp must
    contain a comment explaining WHY (so the next reader doesn't
    re-discover the blocker)."""
    for spec in REGISTERED_DIALECTS:
        if spec.standalone_lit_parseable:
            continue
        # Look for a 'standalone' / 'parse' / 'dotted' / 'lit' comment
        # in at least one of the dialect's .cpp files.
        found_explanation = False
        for cpp_path in cpp_files_for(spec):
            try:
                text = cpp_path.read_text(errors="replace")
            except OSError:
                continue
            # Look for any of these substrings as a sign that the
            # limitation is documented.
            if (
                "standalone lit" in text.lower()
                or "dotted-dialect" in text.lower()
                or "dotted dialect" in text.lower()
                or "dotted-name dialect" in text.lower()
            ):
                found_explanation = True
                break
        assert found_explanation, (
            f"dialect {spec.name!r} has standalone_lit_parseable=False but "
            f"no .cpp file documents the blocker.  Add a comment "
            f"explaining why standalone lit IR doesn't work today."
        )
