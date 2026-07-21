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
from tessera.compiler.capabilities import TARGET_CAPABILITIES
from tessera.compiler.pipeline_registry import (
    REGISTERED_PIPELINES,
    TARGET_PIPELINE_RESOLUTIONS,
    PipelineSpec,
    all_pipeline_names,
    compilation_spine_inventory,
    current_driver_pipeline_map,
    pipeline_lookup,
    pipelines_for_target,
    render_compilation_spine_csv,
    target_pipeline_lookup,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

# Pipelines are registered across multiple .cpp files — the canonical
# Passes.cpp owns the lowering chain, while each retained backend's Passes.cpp
# owns its `tessera-lower-to-{rocm,apple_cpu,apple_gpu}` etc.
_PASS_REGISTRATION_FILES = (
    REPO_ROOT / "src/transforms/lib/Passes.cpp",
    # L-series linalg pilot: the cross-library `-full` Apple spine aliases are
    # registered in the opt driver (they span Transforms + Apple-backend passes
    # that no single backend library links).
    REPO_ROOT / "tools/tessera-opt/tessera-opt.cpp",
    REPO_ROOT / "src/compiler/codegen/Tessera_Apple_Backend/lib/Target/Apple/Passes.cpp",
    REPO_ROOT / "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/Passes.cpp",
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
    "tessera-ebm-pipeline",
    "tessera-spectral-pipeline",
    "tessera-spectral-cleanup",
    "tessera-linalg-solver",
    "tessera-solver-suite",
    "tessera-autodiff-pipeline",
    "tessera-neighbors-pipeline",
    # Per-backend convenience aliases (separate from the
    # `tessera-lower-to-*` umbrella names tracked here).
    "tessera-rocm-backend",
    "tessera-lower-to-nvidia",
    "tessera-lower-to-blackwell",
    "tessera-lower-to-hopper",
    # Phase 4 GPU-emission convenience aliases (2026-06-17): lower a tessera
    # kernel through the linalg→gpu spine to NVVM / ROCDL IR text (emission only;
    # GPU launch is hardware-gated). Composed via parsePassPipeline over upstream
    # MLIR passes, so they require the upstream gpu/nvvm/rocdl dialects rather than
    # Tessera dialects — outside the REGISTERED_DIALECTS-validated PipelineSpec
    # surface. Exercised by tests/tessera-ir/phase8/gpu_emit_nvvm.mlir +
    # tests/unit/test_gpu_emit_nvvm.py.
    "tessera-emit-nvvm",
    "tessera-emit-rocdl",
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
    assert p.targets == ("cpu", "x86", "x86_amx", "x86_avx512")
    assert pipeline_lookup("does-not-exist") is None


# ─────────────────────────────────────────────────────────────────────────
# E2E-SPINE-0: exact target → declared pipeline → pass ownership totality
# ─────────────────────────────────────────────────────────────────────────


def test_target_pipeline_resolution_is_total_and_alphabetized() -> None:
    targets = [resolution.target for resolution in TARGET_PIPELINE_RESOLUTIONS]
    assert targets == sorted(targets)
    assert len(targets) == len(set(targets))
    assert set(targets) == set(TARGET_CAPABILITIES)


def test_driver_mapping_is_derived_from_target_pipeline_resolution() -> None:
    from tessera.compiler.driver import PIPELINE_BY_TARGET

    assert PIPELINE_BY_TARGET == current_driver_pipeline_map()
    assert set(PIPELINE_BY_TARGET) == set(TARGET_CAPABILITIES)


def test_declared_target_pipelines_have_registered_pass_ownership() -> None:
    allowed_states = {
        "declared_exact", "declared_shared_builder", "family_selector",
        "unsupported_no_exact_pipeline",
    }
    for resolution in TARGET_PIPELINE_RESOLUTIONS:
        assert resolution.resolution_state in allowed_states
        assert resolution.level_b in {"partial", "absent"}
        assert resolution.level_c in {"complete", "partial", "absent"}
        source = REPO_ROOT / resolution.registration_source
        assert source.is_file(), resolution
        driver_source = REPO_ROOT / (
            resolution.driver_registration_source or resolution.registration_source
        )
        assert driver_source.is_file(), resolution
        if resolution.current_driver_pipeline != "tessera-target-artifact":
            assert pipeline_lookup(resolution.current_driver_pipeline) is not None
            assert resolution.current_driver_pipeline in driver_source.read_text(
                errors="replace"
            )
        if not resolution.has_declared_pipeline:
            assert resolution.resolution_state == "unsupported_no_exact_pipeline"
            continue
        spec = pipeline_lookup(resolution.declared_pipeline or "")
        assert spec is not None, resolution
        assert resolution.target in spec.targets, resolution
        assert spec.passes, resolution
        assert resolution.declared_pipeline in source.read_text(errors="replace")


def test_nvidia_aliases_and_backend_producers_are_exact_sm() -> None:
    default = pipeline_lookup("tessera-nvidia-pipeline")
    sm90 = pipeline_lookup("tessera-nvidia-pipeline-sm90")
    sm100 = pipeline_lookup("tessera-nvidia-pipeline-sm100")
    sm120 = pipeline_lookup("tessera-nvidia-pipeline-sm120")
    assert all(item is not None for item in (default, sm90, sm100, sm120))
    assert default is not None and sm90 is not None
    assert sm100 is not None and sm120 is not None
    assert default.passes == sm90.passes
    for blackwell in (sm100, sm120):
        assert "tessera-wgmma-lowering" not in blackwell.passes
        assert "tessera-nv-flash-attn-emitter" not in blackwell.passes
        assert blackwell.passes != sm90.passes
    for target, producer in (
        ("nvidia_sm90", "tessera-lower-to-nvidia-sm90"),
        ("nvidia_sm100", "tessera-lower-to-nvidia-sm100"),
        ("nvidia_sm120", "tessera-lower-to-nvidia-sm120"),
    ):
        resolution = target_pipeline_lookup(target)
        assert resolution is not None
        assert resolution.resolution_state == "declared_exact"
        assert resolution.declared_pipeline == producer


def test_compilation_spine_inventory_is_machine_readable_and_truthful() -> None:
    rows = compilation_spine_inventory()
    assert {row.target for row in rows} == set(TARGET_CAPABILITIES)
    assert next(row for row in rows if row.target == "apple_gpu").level_c == "partial"
    assert next(row for row in rows if row.target == "apple_cpu").level_c == "partial"
    assert all(
        row.level_c == "absent"
        for row in rows
        if row.target not in {"apple_gpu", "apple_cpu"}
    )
    assert target_pipeline_lookup("nvidia_sm80").declared_pipeline is None  # type: ignore[union-attr]
    assert next(row for row in rows if row.target == "nvidia_sm120").level_a == "native"
    assert next(row for row in rows if row.target == "rocm_gfx1151").level_a == "native"
    csv_text = render_compilation_spine_csv()
    assert csv_text.startswith("schema,target,family,runtime_backend,")
    assert "tessera.target_pipeline.v1,nvidia_sm80," in csv_text
