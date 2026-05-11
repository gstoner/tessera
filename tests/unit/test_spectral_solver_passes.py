"""Guard tests for the Tessera Spectral solver C++ pass bodies.

The spectral solver lives under ``src/solvers/spectral/`` and is a separate
MLIR/LLVM subproject (a ``ts-spectral-opt`` tool plus six passes) that the
Python test suite cannot build or invoke end-to-end here.  What we *can*
guard from Python is structural:

  1. Every pass body must have left the ``// TODO: implement`` placeholder
     stage — these are real implementations, not scaffolds.
  2. Every pass file must define its ``runOnOperation`` entry point and the
     ``createXxxPass`` factory the driver wires into the registration list.
  3. ``ts-spectral-opt.cpp`` must register all six passes and expose the
     canonical ``tessera-spectral-pipeline`` pipeline alias.
  4. Every differentiable spectral primitive (fft, ifft, rfft, irfft, stft,
     istft, dct, spectral_filter, spectral_conv) must show vjp=complete +
     jvp=complete + lowering_rule=complete in the standalone primitive
     coverage registry.

If anything regresses (a pass body gets reverted to a stub, the driver
forgets to register a pass, or a spectral primitive's contract axis falls
back to ``planned``), this test catches it without needing the MLIR build.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SPECTRAL = REPO / "src" / "solvers" / "spectral"

PASS_FILES = {
    "LegalizeSpectral": "LegalizeSpectralPass",
    "SpectralMXP": "SpectralMXPPass",
    "TransposePlan": "TransposePlanPass",
    "Autotune": "AutotunePass",
    "LowerToTargetIR": "LowerToTargetIRPass",
    "DistributedFFT": "DistributedFFTPass",
}

FACTORY_NAMES = {
    "LegalizeSpectral": "createLegalizeSpectralPass",
    "SpectralMXP": "createSpectralMXPPass",
    "TransposePlan": "createSpectralTransposePlanPass",
    "Autotune": "createSpectralAutotunePass",
    "LowerToTargetIR": "createLowerSpectralToTargetIRPass",
    "DistributedFFT": "createSpectralDistributedPass",
}

DIFFERENTIABLE_SPECTRAL_PRIMS = (
    "fft",
    "ifft",
    "rfft",
    "irfft",
    "stft",
    "istft",
    "dct",
    "spectral_filter",
    "spectral_conv",
)


# ──────────────────────────────────────────────────────────────────────────
#                     C++ pass bodies — not stubs
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("stem,cls", list(PASS_FILES.items()))
def test_spectral_pass_body_is_not_stub(stem, cls):
    path = SPECTRAL / "lib" / "Passes" / f"{stem}.cpp"
    assert path.exists(), f"{path} missing"
    body = path.read_text()
    # Old stub marker that meant "no implementation yet".
    assert "// TODO: implement" not in body, (
        f"{stem}.cpp still has the legacy `// TODO: implement` stub marker"
    )
    # Pass struct + runOnOperation override + factory must all be present.
    assert f"struct {cls}" in body, f"{stem}.cpp missing struct {cls}"
    assert "runOnOperation()" in body, (
        f"{stem}.cpp missing runOnOperation override"
    )
    assert FACTORY_NAMES[stem] in body, (
        f"{stem}.cpp missing factory {FACTORY_NAMES[stem]}"
    )
    # Real implementations walk the spectral exec ops.  Bodies that walk
    # the module are how we distinguish "real" from "no-op stub".
    assert "tessera_spectral.fft" in body or "tessera_spectral." in body, (
        f"{stem}.cpp body should mention tessera_spectral.* ops"
    )


# ──────────────────────────────────────────────────────────────────────────
#                  ts-spectral-opt registers all six passes
# ──────────────────────────────────────────────────────────────────────────

def test_driver_registers_all_spectral_passes():
    driver = SPECTRAL / "tools" / "ts-spectral-opt.cpp"
    body = driver.read_text()
    for factory in FACTORY_NAMES.values():
        assert factory in body, (
            f"ts-spectral-opt.cpp must reference {factory} (got: {body})"
        )
    # Canonical end-to-end pipeline alias is part of the contract.
    assert "tessera-spectral-pipeline" in body, (
        "ts-spectral-opt.cpp must register the `tessera-spectral-pipeline` alias"
    )


# ──────────────────────────────────────────────────────────────────────────
#                Differentiable spectral primitives — fully wired
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", DIFFERENTIABLE_SPECTRAL_PRIMS)
def test_spectral_primitive_has_vjp_and_jvp(name):
    import tessera.autodiff.vjp as vjp_mod
    import importlib

    jvp_mod = importlib.import_module("tessera.autodiff.jvp")
    assert name in vjp_mod._VJPS, f"VJP for {name} not registered"
    assert name in jvp_mod._JVPS, f"JVP for {name} not registered"


@pytest.mark.parametrize("name", DIFFERENTIABLE_SPECTRAL_PRIMS)
def test_spectral_primitive_contract_axes_complete(name):
    """The coverage registry must reflect vjp/jvp/lowering complete now that
    the spectral solver passes are real and the autodiff (V/J)VPs are
    registered.  Backend kernel + sharding remain ``partial`` until a real
    distributed GPU runtime lights up — those are universally gated."""
    from tessera.compiler import primitive_coverage as pc

    entry = pc.coverage_for(name)
    assert entry is not None, f"{name} missing from coverage registry"
    cs = entry.contract_status
    assert cs["vjp"] == "complete", f"{name}.vjp = {cs['vjp']!r}, expected complete"
    assert cs["jvp"] == "complete", f"{name}.jvp = {cs['jvp']!r}, expected complete"
    assert cs["lowering_rule"] == "complete", (
        f"{name}.lowering_rule = {cs['lowering_rule']!r}, expected complete"
    )


# ──────────────────────────────────────────────────────────────────────────
#                       Lit fixtures upgraded
# ──────────────────────────────────────────────────────────────────────────

def test_lit_fixtures_no_placeholder_check_lines():
    """The original lit fixtures all had ``// TODO: expect ...`` placeholder
    CHECK lines.  Real CHECK lines that exercise the structured attributes
    each pass attaches must have replaced them."""
    fixtures = list((SPECTRAL / "test" / "ir").glob("*.mlir"))
    assert len(fixtures) >= 6, f"expected ≥6 lit fixtures, got {len(fixtures)}"
    for f in fixtures:
        body = f.read_text()
        assert "// TODO: expect" not in body, (
            f"{f.name} still uses placeholder `// TODO: expect` CHECK line"
        )
        # Every fixture must use FileCheck — either the default `CHECK:`
        # prefix or a custom prefix configured via `--check-prefix=`.
        assert "FileCheck" in body, f"{f.name} has no FileCheck RUN line"
        import re

        match = re.search(r"--check-prefix=(\w+)", body)
        prefix = match.group(1) if match else "CHECK"
        assert f"// {prefix}:" in body or f"// {prefix}-SAME:" in body, (
            f"{f.name} has no // {prefix}: directives matching its "
            f"--check-prefix={prefix}"
        )
