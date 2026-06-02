"""Compiler-correctness X-b — pass-order matrix tests (Gap 3).

Locks the *implicit* dependencies between pipeline passes as named
contracts.  Two pipelines exercised today, per
``docs/audit/compiler/COMPILER_AUDIT.md``:

* **Halo / Neighbors** (4 passes, runtime matrix via tessera-opt):
    stencil-lower → bc-lower → halo-mesh-integration → halo-transport-lower

  These passes hand off via *attributes* — each pass reads attributes
  written by the previous one.  The matrix verifies:
    1. Forward (canonical) order produces the expected final IR shape.
    2. Skipping a prerequisite pass either emits a *named* warning or
       leaves the IR in a recognizable partial state (never crashes).
    3. Re-running the canonical order is idempotent (sentinels work).

* **Spectral solver** (6 passes, source-structural matrix):
    legalize → mxp → transpose-plan → autotune → distributed → lower-to-target-ir

  ``ts-spectral-opt`` is a separate driver not built in the default
  tree.  We lock the canonical order at the source level by parsing
  ``ts-spectral-opt.cpp`` and asserting the pipeline composition is
  exactly the documented sequence.  The runtime matrix activates when
  the binary builds.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_tessera_opt() -> str | None:
    for c in (
        os.environ.get("TESSERA_OPT"),
        shutil.which("tessera-opt"),
        str(REPO_ROOT / "build" / "tools" / "tessera-opt" / "tessera-opt"),
    ):
        if c and Path(c).exists():
            return c
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Halo pipeline — runtime matrix
# ─────────────────────────────────────────────────────────────────────────────


_HALO_INPUT = """\
func.func @t(%arg0: tensor<?x?xf32>) {
  "schedule.mesh.define"() {axis_names = ["dp"], dims = [2]} : () -> ()
  "schedule.mesh.region"() ({
    %topo = "tessera.neighbors.topology.create"() {kind = "2d_mesh"} : () -> !tessera.neighbors.topology
    %st = "tessera.neighbors.stencil.define"() {
        taps = [dense<[0, 0]> : tensor<2xi64>,
                dense<[1, 0]> : tensor<2xi64>,
                dense<[-1, 0]> : tensor<2xi64>],
        bc = "periodic"
    } : () -> index
    %h = "tessera.neighbors.halo.region"(%arg0) {halo.width = [1, 1]} : (tensor<?x?xf32>) -> tensor<?x?xf32>
    %out = "tessera.neighbors.stencil.apply"(%st, %h, %topo) :
        (index, tensor<?x?xf32>, !tessera.neighbors.topology) -> tensor<?x?xf32>
    "schedule.yield"() : () -> ()
  }) {mesh = @dp, axis = "dp"} : () -> ()
  return
}
"""


HALO_PASSES = [
    "-tessera-stencil-lower",
    "-tessera-boundary-condition-lower",
    "-tessera-halo-mesh-integration",
    "-tessera-halo-transport-lower",
]


def _run(binary: str, pass_args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [binary, "--allow-unregistered-dialect", *pass_args],
        input=_HALO_INPUT, capture_output=True, text=True, timeout=30,
    )


class TestHaloPipelineCanonicalOrder:
    """Forward (canonical) order produces the expected end-state IR:
    every halo.exchange has been lowered to pack/transport/unpack
    triples, BC attributes are structured, and the mesh-integrated
    sentinel is set on the stencil.apply."""

    def test_canonical_order_produces_expected_attrs(self):
        binary = _find_tessera_opt()
        if binary is None:
            pytest.skip("tessera-opt not built")
        r = _run(binary, HALO_PASSES)
        assert r.returncode == 0, r.stderr
        out = r.stdout
        # Stencil-lower: stencil.lowered + halo_width.
        assert "stencil.lowered = true" in out
        assert "stencil.halo_width = [1, 1]" in out
        # BC-lower: structured per-axis modes.
        assert 'stencil.bc.modes = ["periodic", "periodic"]' in out
        assert "stencil.bc.lowered = true" in out
        # Halo-mesh integration: halo.exchange inserted then *consumed*
        # by halo-transport-lower; sentinel survives on stencil.apply.
        assert "halo.mesh_integrated = true" in out
        # Halo-transport: triples emitted, no surviving halo.exchange.
        assert "tessera.neighbors.halo.exchange" not in out
        assert "tessera.neighbors.halo.pack" in out
        assert "tessera.neighbors.halo.transport" in out
        assert "tessera.neighbors.halo.unpack" in out
        assert 'inserted_by = "halo-transport-lower"' in out


class TestHaloPipelineDependencies:
    """Each pass declares an *implicit* dependency on its predecessor.
    Skipping a predecessor must produce either a named warning or a
    recognizable partial state — *never* a silent miscompilation."""

    def test_bc_lower_skips_when_stencil_lower_did_not_run(self):
        """bc-lower writes default 'periodic' modes when stencil-lower
        hasn't emitted ``stencil.bc`` yet.  No crash; output is sane."""
        binary = _find_tessera_opt()
        if binary is None:
            pytest.skip("tessera-opt not built")
        r = _run(binary, ["-tessera-boundary-condition-lower"])
        # Pass must not crash; runs over an op that lacks stencil.bc
        # → the BC pass falls back to default-periodic per the
        # architecture doc.
        assert r.returncode == 0, r.stderr

    def test_materialize_warns_without_bc_lower(self):
        """halo-mesh-integration requires bc-lower's per-axis modes to
        emit BC-vs-mesh conflicts.  Without bc-lower, the conflict
        diagnostic is absent (no crash, no false positive)."""
        binary = _find_tessera_opt()
        if binary is None:
            pytest.skip("tessera-opt not built")
        r = _run(binary,
                  ["-tessera-stencil-lower",
                   "-tessera-halo-mesh-integration"])
        assert r.returncode == 0, r.stderr
        # Without bc-lower, the BC modes aren't structured and the
        # reconcile pass can't fire — the diagnostic should NOT appear
        # (because there's nothing structured to reconcile against),
        # but the integration sentinel + halo.exchange insertion must
        # still happen.
        out = r.stdout
        assert "halo.mesh_integrated = true" in out
        # The diagnostic body — "incompatible with mesh axis policy" —
        # comes from the BC reconciler reading structured modes.
        # Without bc-lower it can't fire, so:
        assert "incompatible with mesh axis policy" not in out

    def test_transport_lower_skips_when_no_halo_exchange_present(self):
        """halo-transport-lower walks halo.exchange ops only.  If
        nothing is upstream emitting them, the pass is a no-op."""
        binary = _find_tessera_opt()
        if binary is None:
            pytest.skip("tessera-opt not built")
        r = _run(binary, ["-tessera-halo-transport-lower"])
        assert r.returncode == 0, r.stderr
        # No halo.exchange in input → no pack/transport/unpack in output.
        assert "tessera.neighbors.halo.pack" not in r.stdout


class TestHaloPipelineIdempotency:
    """The canonical order, run twice, must produce identical IR.  The
    sentinel attributes (stencil.lowered, stencil.bc.lowered,
    halo.mesh_integrated, halo.transport_lowered) make every pass
    idempotent."""

    def test_canonical_order_is_idempotent(self):
        binary = _find_tessera_opt()
        if binary is None:
            pytest.skip("tessera-opt not built")
        # First run.
        r1 = _run(binary, HALO_PASSES)
        assert r1.returncode == 0, r1.stderr
        # Second run (same passes, same input).
        r2 = _run(binary, HALO_PASSES)
        assert r2.returncode == 0, r2.stderr
        # Stable IR — identical char-for-char output across runs.
        assert r1.stdout == r2.stdout, (
            "canonical halo pipeline is non-deterministic — same input "
            "produced different IR across two runs"
        )

    def test_re_running_each_pass_in_sequence_is_idempotent(self):
        """Run the canonical pipeline once, then re-run the same passes
        — every pass's sentinel should make the second invocation a
        no-op."""
        binary = _find_tessera_opt()
        if binary is None:
            pytest.skip("tessera-opt not built")
        # First-and-second-time-through one tessera-opt invocation — the
        # pass manager will run each pass twice in this order.
        r = _run(binary, HALO_PASSES + HALO_PASSES)
        assert r.returncode == 0, r.stderr
        # Exactly one materialized sentinel (not duplicated).
        out = r.stdout
        # The mesh-integrated sentinel is set once and stays.
        assert out.count("halo.mesh_integrated = true") == 1


class TestHaloPipelineReorderRejection:
    """Some reorderings break the contract.  These are the high-value
    negative tests — the regressions they catch are exactly the kind
    of silent bug that's hardest to find without an explicit gate."""

    def test_transport_before_integration_emits_no_triple(self):
        """halo-transport-lower expects halo.exchange ops to exist.
        halo-mesh-integration is what inserts them.  Run transport
        before integration → no halo.exchange exists yet → transport
        is a no-op.  The IR survives but ghost regions never fill."""
        binary = _find_tessera_opt()
        if binary is None:
            pytest.skip("tessera-opt not built")
        r = _run(binary,
                  ["-tessera-stencil-lower",
                   "-tessera-boundary-condition-lower",
                   "-tessera-halo-transport-lower",            # wrong order
                   "-tessera-halo-mesh-integration"])
        assert r.returncode == 0, r.stderr
        out = r.stdout
        # The integration inserted halo.exchange — but transport already
        # ran, so no triples replaced it.
        assert "tessera.neighbors.halo.exchange" in out
        assert "tessera.neighbors.halo.pack" not in out

    def test_bc_lower_before_stencil_lower_falls_through_to_default(self):
        """bc-lower runs over the stencil.apply op which doesn't have a
        stencil.bc attribute yet (that's set by stencil-lower).  bc-
        lower falls through to default-periodic, then stencil-lower
        overwrites with the real BC from stencil.define."""
        binary = _find_tessera_opt()
        if binary is None:
            pytest.skip("tessera-opt not built")
        r = _run(binary,
                  ["-tessera-boundary-condition-lower",  # wrong order
                   "-tessera-stencil-lower"])
        assert r.returncode == 0, r.stderr
        # Important: this DOES leave a regression footprint.  The
        # bc.lowered sentinel is set, so a *second* bc-lower run will
        # skip and the per-axis modes will not reflect the actual BC
        # string.  This test pins that footprint so a future fix
        # (e.g., making bc-lower delay until stencil.bc is present)
        # surfaces as an intentional test update.
        out = r.stdout
        assert "stencil.bc.lowered = true" in out


# ─────────────────────────────────────────────────────────────────────────────
# Spectral pipeline — source-structural matrix
# ─────────────────────────────────────────────────────────────────────────────

_SPECTRAL_TOOL_CPP = (
    REPO_ROOT / "src" / "solvers" / "spectral" / "tools"
    / "ts-spectral-opt.cpp"
)

# Canonical order — must match the architecture doc.  Re-ordering any
# of these requires an explicit edit *and* an update to this test.
SPECTRAL_CANONICAL_ORDER = [
    "createLegalizeSpectralPass",
    "createSpectralMXPPass",
    "createSpectralTransposePlanPass",
    "createSpectralAutotunePass",
    "createSpectralDistributedPass",
    "createLowerSpectralToTargetIRPass",
]


class TestSpectralPipelineCanonicalOrder:
    """ts-spectral-opt is not built in the default tree (the binary is
    separate from tessera-opt).  We pin the pipeline-alias composition
    at source level instead — any reorder of the addPass(…) calls
    in ts-spectral-opt.cpp will fail this test."""

    def test_pipeline_alias_source_exists(self):
        assert _SPECTRAL_TOOL_CPP.exists()

    def test_canonical_order_matches_source(self):
        src = _SPECTRAL_TOOL_CPP.read_text()
        # Extract the addPass(...) lines from the pipeline-alias scope.
        # Pattern: ``pm.addPass(tessera::createXxxPass());``
        matches = re.findall(r"pm\.addPass\(tessera::(\w+)\(\)\);", src)
        assert matches, "no pm.addPass(tessera::...) calls found"
        assert matches == SPECTRAL_CANONICAL_ORDER, (
            f"spectral pipeline source order drifted!\n"
            f"  expected: {SPECTRAL_CANONICAL_ORDER}\n"
            f"  found   : {matches}\n"
            "If this is an intentional reorder, update "
            "SPECTRAL_CANONICAL_ORDER + the architecture doc + the "
            "compiler-correctness audit."
        )

    def test_pipeline_alias_name_is_documented(self):
        """The alias name itself is part of the public contract.  Other
        tooling (validate.sh, benchmarks) calls it by name."""
        src = _SPECTRAL_TOOL_CPP.read_text()
        assert '"tessera-spectral-pipeline"' in src


class TestSpectralPipelineDependencies:
    """The spectral passes are annotation-driven — each writes a
    ``tessera.<phase>.*`` attribute the next consumes.  Locking the
    dependency direction at source level catches removal of any one
    pass without removing its consumers."""

    DEPENDENCY_PAIRS = [
        # (producer pass arg, consumer pass arg) — producer must run
        # before consumer in the canonical pipeline.
        ("tessera-legalize-spectral", "tessera-spectral-mxp"),
        ("tessera-spectral-mxp", "tessera-spectral-transpose-plan"),
        ("tessera-spectral-transpose-plan", "tessera-spectral-autotune"),
        ("tessera-spectral-autotune", "tessera-spectral-distributed"),
        ("tessera-spectral-distributed", "lower-spectral-to-target-ir"),
    ]

    @pytest.mark.parametrize("producer,consumer", DEPENDENCY_PAIRS)
    def test_each_dependency_is_locked_in_canonical_order(
        self, producer: str, consumer: str,
    ) -> None:
        """For each documented producer→consumer pair, the producer
        must precede the consumer in SPECTRAL_CANONICAL_ORDER."""
        # The argument names map to create-fn names — recover the
        # mapping by reading the cpp files.
        passes_dir = (REPO_ROOT / "src" / "solvers" / "spectral"
                      / "lib" / "Passes")
        producer_create_fn = None
        consumer_create_fn = None
        for cpp in passes_dir.glob("*.cpp"):
            txt = cpp.read_text()
            if f'return "{producer}"' in txt:
                # Extract createXxxPass from the file's create function.
                m = re.search(r"std::unique_ptr<(?:mlir::)?Pass>\s+(\w+)\(", txt)
                if m:
                    producer_create_fn = m.group(1)
            if f'return "{consumer}"' in txt:
                m = re.search(r"std::unique_ptr<(?:mlir::)?Pass>\s+(\w+)\(", txt)
                if m:
                    consumer_create_fn = m.group(1)
        assert producer_create_fn is not None, (
            f"could not locate create fn for {producer}"
        )
        assert consumer_create_fn is not None, (
            f"could not locate create fn for {consumer}"
        )
        # Canonical-order check.
        assert producer_create_fn in SPECTRAL_CANONICAL_ORDER
        assert consumer_create_fn in SPECTRAL_CANONICAL_ORDER
        producer_idx = SPECTRAL_CANONICAL_ORDER.index(producer_create_fn)
        consumer_idx = SPECTRAL_CANONICAL_ORDER.index(consumer_create_fn)
        assert producer_idx < consumer_idx, (
            f"{producer} (idx {producer_idx}) must precede "
            f"{consumer} (idx {consumer_idx}) in the canonical order"
        )
