"""Arch-6 (2026-05-22) — pass metadata drift gate.

Pins the cross-registry consistency:

  * Every diagnostic_codes entry exists in Arch-1's REGISTERED_CODES.
  * Every must_run_after / can_run_after target is itself a Layer-B
    pass.
  * Every input_dialect / output_dialect is registered or a standard
    MLIR dialect.
  * The cpp_class identifier appears in the actual C++ source tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera.compiler.diagnostic_codes import all_codes as all_diagnostic_codes
from tessera.compiler.dialects_manifest import REGISTERED_DIALECTS
from tessera.compiler.pass_metadata import (
    REGISTERED_PASSES,
    PassMetadata,
    all_pass_names,
    pass_lookup,
    passes_emitting_code,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"


# ─────────────────────────────────────────────────────────────────────────
# Structural tests
# ─────────────────────────────────────────────────────────────────────────


def test_registered_passes_have_required_fields() -> None:
    for spec in REGISTERED_PASSES:
        assert spec.name, "empty pass name"
        assert spec.cpp_class, f"empty cpp_class for {spec.name}"
        assert spec.summary, f"empty summary for {spec.name}"
        assert spec.input_dialects, (
            f"empty input_dialects for {spec.name}"
        )
        assert spec.output_dialects, (
            f"empty output_dialects for {spec.name}"
        )
        assert spec.pass_kind in ("verifier", "transform", "lowering"), (
            f"invalid pass_kind {spec.pass_kind!r} for {spec.name}"
        )
        assert spec.sprint, f"empty sprint label for {spec.name}"


def test_pass_names_are_unique() -> None:
    seen: set[str] = set()
    for spec in REGISTERED_PASSES:
        assert spec.name not in seen, f"duplicate pass: {spec.name}"
        seen.add(spec.name)


def test_pass_metadata_is_alphabetized() -> None:
    names = [p.name for p in REGISTERED_PASSES]
    assert names == sorted(names), (
        f"REGISTERED_PASSES must be alphabetised; out-of-order: "
        f"{[n for n, s in zip(names, sorted(names)) if n != s]}"
    )


# ─────────────────────────────────────────────────────────────────────────
# Cross-registry: diagnostic_codes ↔ Arch-1
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("spec", REGISTERED_PASSES, ids=lambda s: s.name)
def test_diagnostic_codes_are_registered(spec: PassMetadata) -> None:
    """Every code a pass claims to emit must be in Arch-1's registry."""
    known = set(all_diagnostic_codes())
    unknown = [c for c in spec.diagnostic_codes if c not in known]
    assert not unknown, (
        f"pass {spec.name!r} declares diagnostic codes "
        f"{unknown!r} that are not in REGISTERED_CODES.  Either add "
        f"them to `python/tessera/compiler/diagnostic_codes.py` or "
        f"remove from this pass's diagnostic_codes tuple."
    )


def test_passes_emitting_code_helper() -> None:
    """For each pass's declared codes, the inverse lookup must
    return at least that pass."""
    for spec in REGISTERED_PASSES:
        for code in spec.diagnostic_codes:
            emitters = passes_emitting_code(code)
            assert spec in emitters, (
                f"passes_emitting_code({code!r}) didn't return {spec.name!r}"
            )


# ─────────────────────────────────────────────────────────────────────────
# Cross-registry: dialects ↔ Arch-4 / standard MLIR
# ─────────────────────────────────────────────────────────────────────────


_STANDARD_MLIR_DIALECTS = frozenset({
    "func", "scf", "arith", "memref", "tensor", "linalg",
    "llvm", "math", "builtin", "vector", "affine", "bufferization",
    "nvvm", "rocdl",
    # Tessera-side dialects not yet in Arch-4's manifest — see Arch-4
    # rollout (today only the 3 already-registered dialects are
    # captured; this set absorbs the rest).
    "schedule.mesh", "tessera.neighbors", "tessera.solver",
    "tessera_apple", "tpp",
})


@pytest.mark.parametrize("spec", REGISTERED_PASSES, ids=lambda s: s.name)
def test_dialects_are_known(spec: PassMetadata) -> None:
    tessera_dialects = {d.name for d in REGISTERED_DIALECTS}
    known = tessera_dialects | _STANDARD_MLIR_DIALECTS
    for d in spec.input_dialects + spec.output_dialects:
        assert d in known, (
            f"pass {spec.name!r} references unknown dialect {d!r}; "
            f"either add to dialects_manifest or the standard MLIR set."
        )


# ─────────────────────────────────────────────────────────────────────────
# Cross-registry: ordering constraints
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("spec", REGISTERED_PASSES, ids=lambda s: s.name)
def test_must_run_after_references_known_pass(spec: PassMetadata) -> None:
    """`must_run_after` targets must be passes we track (Layer-B
    coverage) or one of the standard MLIR passes."""
    layer_b = set(all_pass_names())
    # Standard MLIR passes that Tessera pipelines often chain through —
    # not in our registry but still legitimate ordering anchors.
    standard_mlir = {"canonicalize", "cse", "loop-invariant-code-motion"}
    known = layer_b | standard_mlir
    for ref in spec.must_run_after + spec.can_run_after:
        assert ref in known, (
            f"pass {spec.name!r} declares must_run_after / can_run_after "
            f"{ref!r} which is neither a Layer-B pass nor a standard "
            f"MLIR pass.  Either add metadata for {ref!r} or fix the "
            f"name."
        )


# ─────────────────────────────────────────────────────────────────────────
# cpp_class lives in the source tree
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("spec", REGISTERED_PASSES, ids=lambda s: s.name)
def test_cpp_class_exists_in_source(spec: PassMetadata) -> None:
    """The C++ class name must appear somewhere under src/.  Catches
    a rename without registry update."""
    found = False
    for ext in ("*.cpp", "*.h"):
        for path in SRC_ROOT.rglob(ext):
            try:
                if spec.cpp_class in path.read_text(errors="replace"):
                    found = True
                    break
            except OSError:
                continue
        if found:
            break
    assert found, (
        f"cpp_class {spec.cpp_class!r} for pass {spec.name!r} not found "
        f"in any .cpp / .h under src/.  Was the class renamed?"
    )


# ─────────────────────────────────────────────────────────────────────────
# Helper sanity
# ─────────────────────────────────────────────────────────────────────────


def test_pass_lookup_round_trips() -> None:
    for spec in REGISTERED_PASSES:
        assert pass_lookup(spec.name) is spec
    assert pass_lookup("never-registered-pass") is None


def test_known_sentinels_present() -> None:
    """Lock the canonical passes that we know are real and locked."""
    expected = {
        "tessera-distribution-lower",
        "tessera-effect-annotate",
        "tessera-layout-legality",
        "tessera-symdim-equality",
    }
    actual = set(all_pass_names())
    missing = expected - actual
    assert not missing, f"locked Layer-B passes missing: {sorted(missing)}"
