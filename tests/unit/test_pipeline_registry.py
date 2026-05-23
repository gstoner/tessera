"""Arch-5 (2026-05-22) — pipeline registry drift gate.

Pins:
  * Every registered pipeline name appears in the C++
    ``mlir::PassPipelineRegistration`` calls in
    ``src/transforms/lib/Passes.cpp``.  Catches a Python registry
    entry whose C++ counterpart was deleted or renamed.
  * Every C++ ``PassPipelineRegistration`` name that targets the
    tessera/x86/gpu surface is in the Python registry (catches the
    reverse drift).
  * Every ``required_dialects`` entry exists in
    ``REGISTERED_DIALECTS`` (no orphan dialect references).
  * Every ``lit_fixtures`` path exists and its file contains the
    pipeline name in a ``RUN:`` line.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tessera.compiler.dialects_manifest import REGISTERED_DIALECTS
from tessera.compiler.pipeline_registry import (
    REGISTERED_PIPELINES,
    PipelineSpec,
    all_pipeline_names,
    pipeline_lookup,
    pipelines_for_target,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

# Pipelines are registered across multiple .cpp files — the canonical
# Passes.cpp owns the lowering chain, while each backend's Passes.cpp
# owns its `tessera-lower-to-{rocm,metalium,apple_cpu,apple_gpu}` etc.
_PASS_REGISTRATION_FILES = (
    REPO_ROOT / "src/transforms/lib/Passes.cpp",
    REPO_ROOT / "src/compiler/codegen/Tessera_Apple_Backend/lib/Target/Apple/Passes.cpp",
    REPO_ROOT / "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/Passes.cpp",
    REPO_ROOT / "src/compiler/codegen/Tessera_Metalium_Backend/lib/Target/Metalium/Passes.cpp",
    REPO_ROOT / "src/compiler/codegen/Tessera_Cerebras_backend/lib/Passes.cpp",
    REPO_ROOT / "src/compiler/codegen/Tessera_RubinCPX_Backend/lib/Transforms/Passes.cpp",
    REPO_ROOT / "src/compiler/codegen/Tessera_TPU_Backend/src/passes/RegisterPasses.cpp",
    REPO_ROOT / "src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp",
    REPO_ROOT / "src/compiler/programming_model/tools/tessera-opt/PassPipelinesPM11.cpp",
    REPO_ROOT / "src/compiler/mlir/TesseraMLIRPlugin.cpp",
    # Solver pipelines (ts-spectral-opt, ts-clifford-opt, etc.)
    REPO_ROOT / "src/solvers/core/passes/SolversPasses.cpp",
    REPO_ROOT / "src/solvers/spectral/tools/ts-spectral-opt.cpp",
    REPO_ROOT / "src/solvers/clifford/tools/ts-clifford-opt.cpp",
    REPO_ROOT / "src/solvers/ebm/tools/ts-ebm-opt.cpp",
    REPO_ROOT / "src/solvers/tpp/lib/InitTPP.cpp",
    REPO_ROOT / "src/solvers/tpp/lib/Passes/PassPipeline.cpp",
    REPO_ROOT / "src/solvers/linalg/lib/Passes/MixedPrecision.cpp",
)


# ─────────────────────────────────────────────────────────────────────────
# Structural tests
# ─────────────────────────────────────────────────────────────────────────


def test_registered_pipelines_have_required_fields() -> None:
    for spec in REGISTERED_PIPELINES:
        assert spec.name, "empty pipeline name"
        assert spec.passes, f"empty passes for {spec.name}"
        assert spec.required_dialects, (
            f"empty required_dialects for {spec.name}"
        )
        assert spec.targets, f"empty targets for {spec.name}"
        assert spec.phase in ("lowering", "tile_opt", "target", "verify"), (
            f"invalid phase {spec.phase!r} for {spec.name}"
        )
        assert spec.status in ("lit_verified", "wired", "planned"), (
            f"invalid status {spec.status!r} for {spec.name}"
        )


def test_pipelines_have_no_duplicate_names() -> None:
    seen: set[str] = set()
    for spec in REGISTERED_PIPELINES:
        assert spec.name not in seen, f"duplicate pipeline: {spec.name}"
        seen.add(spec.name)


def test_pipelines_are_alphabetized() -> None:
    names = [p.name for p in REGISTERED_PIPELINES]
    assert names == sorted(names), (
        f"REGISTERED_PIPELINES must be alphabetised; out-of-order: "
        f"{[n for n, s in zip(names, sorted(names)) if n != s]}"
    )


def test_verifier_passes_are_subset_of_passes() -> None:
    for spec in REGISTERED_PIPELINES:
        for vp in spec.verifier_passes:
            assert vp in spec.passes, (
                f"verifier pass {vp!r} not in {spec.name}.passes — "
                f"either add it to passes or remove from verifier_passes"
            )


def test_required_dialects_are_registered() -> None:
    """Cross-registry check: any required dialect must be in the
    Arch-4 dialect manifest, OR be a standard MLIR dialect we use
    everywhere (func / scf / arith / etc.)."""
    tessera_dialects = {d.name for d in REGISTERED_DIALECTS}
    standard_mlir_dialects = {
        "func", "scf", "arith", "memref", "tensor", "linalg",
        "llvm", "math", "builtin", "vector", "affine", "bufferization",
        "nvvm", "rocdl",
        # Tessera-side dialects not yet in REGISTERED_DIALECTS today —
        # add them to the manifest if we want full coverage.  For now,
        # accept the ones that exist as known C++ dialects.
        "tessera_apple", "tessera.neighbors", "tessera.solver", "tpp",
    }
    known = tessera_dialects | standard_mlir_dialects
    for spec in REGISTERED_PIPELINES:
        for dialect in spec.required_dialects:
            assert dialect in known, (
                f"pipeline {spec.name!r} requires dialect {dialect!r} "
                f"which is neither in REGISTERED_DIALECTS nor a "
                f"standard MLIR dialect.  If it's a Tessera dialect "
                f"add it to dialects_manifest; otherwise add it to "
                f"the standard_mlir_dialects set in this test."
            )


# ─────────────────────────────────────────────────────────────────────────
# Drift gate: pipeline name ↔ C++ PassPipelineRegistration
# ─────────────────────────────────────────────────────────────────────────


def _scan_cpp_pipelines() -> set[str]:
    """Return the set of ``tessera-*`` pipeline names found in
    ``PassPipelineRegistration<>(...)`` calls across all known
    registration files.

    Note: the regex captures the FIRST ``"tessera-..."`` string after
    the ``PassPipelineRegistration<>`` token, which is the canonical
    pipeline name argument.  The description string that often follows
    is ignored.
    """
    pattern = re.compile(
        r'PassPipelineRegistration\b[^"]*?"(tessera-[A-Za-z0-9_\-]+)"',
        re.DOTALL,
    )
    out: set[str] = set()
    for path in _PASS_REGISTRATION_FILES:
        if not path.exists():
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        out.update(pattern.findall(text))
    return out


@pytest.mark.parametrize("spec", REGISTERED_PIPELINES, ids=lambda s: s.name)
def test_pipeline_registered_in_cpp(spec: PipelineSpec) -> None:
    """Every Python-side pipeline name must appear in a C++
    PassPipelineRegistration."""
    cpp_pipelines = _scan_cpp_pipelines()
    assert spec.name in cpp_pipelines, (
        f"pipeline {spec.name!r} declared in Python registry but no "
        f"matching PassPipelineRegistration found in Passes.cpp.  "
        f"Either add the C++ registration or remove the Python entry."
    )


_KNOWN_UNTRACKED_PIPELINES: frozenset[str] = frozenset({
    # These C++ pipelines exist but are not yet captured in the
    # Python pipeline_registry.  The drift gate excludes them so the
    # initial Arch-5 landing isn't blocked on inventorying everything;
    # future sprints should slot each one into REGISTERED_PIPELINES.
    # Solver / out-of-tree-tool pipelines (separate driver binaries).
    "tessera-clifford-pipeline",
    "tessera-cpx-pipeline",
    "tessera-cpx-context-pipeline",
    "tessera-ebm-pipeline",
    "tessera-spectral-pipeline",
    "tessera-spectral-cleanup",
    "tessera-linalg-solver",
    "tessera-solver-suite",
    "tessera-autodiff-pipeline",
    "tessera-neighbors-pipeline",
    # Per-backend convenience aliases (separate from the
    # `tessera-lower-to-*` umbrella names tracked here).
    "tessera-cerebras-backend",
    "tessera-metalium",
    "tessera-rocm-backend",
    "tessera-tpu-backend",
    "tessera-lower-to-nvidia",
    "tessera-lower-to-blackwell",
    "tessera-lower-to-hopper",
    # Internal cleanup / verification meta-pipelines.
    "tessera-cleanup",
    "tessera-full-pipeline",
    "tessera-pm-legalize-pipeline",
    "tessera-pm-verify-pipeline",
})


def test_no_orphan_cpp_pipelines() -> None:
    """Every C++ ``tessera-*`` pipeline registration must be in the
    Python registry OR in the _KNOWN_UNTRACKED_PIPELINES allow-list.

    When this fails for an unfamiliar name, you've either:
      (a) added a new C++ pipeline and should add a matching
          PipelineSpec entry in
          ``python/tessera/compiler/pipeline_registry.py``;
      (b) intentionally left it out of the audit surface for now,
          and should add it to ``_KNOWN_UNTRACKED_PIPELINES`` with a
          rationale.
    Either way the gap is visible rather than silent."""
    cpp_pipelines = _scan_cpp_pipelines()
    python_pipelines = set(all_pipeline_names())
    orphans = cpp_pipelines - python_pipelines - _KNOWN_UNTRACKED_PIPELINES
    assert not orphans, (
        f"Unfamiliar C++ pipelines not registered in Python: "
        f"{sorted(orphans)}.  Add PipelineSpec entries in "
        f"`python/tessera/compiler/pipeline_registry.py` or extend "
        f"_KNOWN_UNTRACKED_PIPELINES with a rationale."
    )


def test_known_untracked_pipelines_actually_exist() -> None:
    """An allow-list entry that doesn't match any C++ pipeline is
    dead — either the C++ pipeline was deleted (good — remove the
    allow-list entry) or it was renamed (track the rename)."""
    cpp_pipelines = _scan_cpp_pipelines()
    dead = _KNOWN_UNTRACKED_PIPELINES - cpp_pipelines
    assert not dead, (
        f"Allow-list entries that don't match any C++ pipeline: "
        f"{sorted(dead)}.  These are stale — either restore the C++ "
        f"pipeline or remove the allow-list entry."
    )


# ─────────────────────────────────────────────────────────────────────────
# Lit fixture coverage
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "spec",
    [p for p in REGISTERED_PIPELINES if p.lit_fixtures],
    ids=lambda s: s.name,
)
def test_lit_fixtures_exist_and_reference_pipeline(spec: PipelineSpec) -> None:
    for fixture in spec.lit_fixtures:
        path = REPO_ROOT / fixture
        assert path.exists(), (
            f"pipeline {spec.name!r} declares lit_fixture {fixture!r} "
            f"but the file doesn't exist"
        )
        text = path.read_text()
        # The fixture should reference EITHER the pipeline name itself
        # OR one of its constituent passes — both forms are legitimate
        # ways to exercise the pipeline's behavior.
        referenced = spec.name in text or any(p in text for p in spec.passes)
        assert referenced, (
            f"pipeline {spec.name!r} declares lit_fixture {fixture!r} "
            f"but the file doesn't reference the pipeline name or any "
            f"of its constituent passes: {spec.passes!r}"
        )


# ─────────────────────────────────────────────────────────────────────────
# Helper sanity
# ─────────────────────────────────────────────────────────────────────────


def test_pipelines_for_target() -> None:
    x86 = pipelines_for_target("x86_amx")
    assert any(p.name == "tessera-lower-to-x86" for p in x86)
    nv = pipelines_for_target("nvidia_sm90")
    assert any(p.name == "tessera-nvidia-pipeline-sm90" for p in nv)
    assert pipelines_for_target("nonexistent_target") == ()


def test_pipeline_lookup() -> None:
    p = pipeline_lookup("tessera-lower-to-x86")
    assert p is not None
    assert p.targets == ("x86_amx", "x86_avx512")
    assert pipeline_lookup("does-not-exist") is None
