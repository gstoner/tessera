"""The join between primitive coverage and the MLIR/LLVM route.

`standalone_primitive_coverage.md` and `bootstrap_prune_gap.md` were both
generated, both drift-gated, and had **no join** -- so you could not read the
primitive dashboard and learn whether a primitive is MLIR-native or
bootstrap-only. `lowering_rule` is route-blind (463/515 complete says nothing
about *which* compiler), and `backend_kernel` had 0 completes out of 515, which
makes it uninformative rather than wrong.

These tests are mostly about the GUARD, because the membership half is declared
and a declared map is only as good as what stops it rotting.
"""
from __future__ import annotations

import pytest

from tessera.compiler import primitive_route_map as prm


def test_the_committed_membership_is_consistent_with_both_sources():
    """The guard is the contract; this is it running against the real repo."""
    prm.verify_family_membership()


def test_every_declared_member_is_a_real_coverage_primitive():
    """Caught three of the author's own declarations on its first run.

    `streaming_attention` and `flash_attn_bwd` are names the backend sources
    use that are NOT coverage primitives -- there is no separate backward
    primitive, because a backward is the vjp axis of its forward. A map that
    accepted them would have reported routes for ops the coverage registry has
    never heard of.
    """
    from tessera.compiler.primitive_coverage import all_primitive_coverages

    coverage = all_primitive_coverages()
    for family, members in prm.FAMILY_MEMBERS.items():
        for member in members:
            assert member in coverage, f"{family}: {member!r}"
            assert coverage[member].graph_name, f"{family}: {member!r} has no graph_name"


def test_a_member_that_is_not_a_primitive_is_refused(monkeypatch):
    monkeypatch.setattr(prm, "FAMILY_MEMBERS",
                        {**prm.FAMILY_MEMBERS, "matmul": ("not_a_primitive",)})
    with pytest.raises(ValueError, match="not a coverage primitive"):
        prm.verify_family_membership()


def test_a_family_a_backend_classifies_but_nobody_declared_is_refused(monkeypatch):
    """The half that keeps the map honest as backends grow.

    Without it, a family added to a backend silently has zero members and reads
    as "no primitive is served by it" -- indistinguishable from a family nobody
    uses, and it would quietly under-report the bootstrap surface.
    """
    monkeypatch.setattr(prm, "discovered_families",
                        lambda: {"brand_new_family": ("nvidia_sm120",)})
    with pytest.raises(ValueError, match="no membership declaration"):
        prm.verify_family_membership()


def test_a_compiled_route_family_with_no_declaration_is_refused(monkeypatch):
    """A missing declaration here loses coverage rather than gaining it.

    That direction matters: an undeclared compiled family silently DOWNGRADES
    its primitives to bootstrap-only, so the dashboard would understate the
    mainline compiler. A guard that only caught over-claims would miss it.
    """
    monkeypatch.setattr(prm, "compiled_route_families",
                        lambda: frozenset({"a_compiled_family"}))
    with pytest.raises(ValueError, match="compiled route but no membership"):
        prm.verify_family_membership()


def test_a_stale_declaration_for_a_dead_family_is_refused(monkeypatch):
    monkeypatch.setattr(prm, "FAMILY_MEMBERS",
                        {**prm.FAMILY_MEMBERS, "retired_family": ("matmul",)})
    with pytest.raises(ValueError, match="stale declaration"):
        prm.verify_family_membership()


def test_an_empty_declaration_is_refused(monkeypatch):
    """An empty member tuple would make a family vacuously satisfied."""
    monkeypatch.setattr(prm, "FAMILY_MEMBERS", {**prm.FAMILY_MEMBERS, "matmul": ()})
    with pytest.raises(ValueError, match="no members"):
        prm.verify_family_membership()


def test_routes_prefer_compiled_over_bootstrap_over_none():
    """"Best route wins" is the right reduction for the question being asked.

    The prune plan asks whether deleting the Python bootstrap would strand a
    primitive. One compiled path is enough to answer no, so a primitive served
    by both must read `compiled`.
    """
    routes = prm.primitive_routes()
    assert routes, "no primitive has a route at all"
    kinds = {kind for per_target in routes.values() for kind in per_target.values()}
    assert kinds <= {prm.ROUTE_COMPILED, prm.ROUTE_BOOTSTRAP, prm.ROUTE_NONE}

    # `softmax` is the standing example of a family with a bootstrap packager on
    # every backend and no compiled route -- if this ever flips, the mainline
    # compiler absorbed it and the prune plan moved.
    assert prm.derived_backend_kernel("softmax") == "partial"
    # `flash_attn` has a compiled route (scheduled_attention).
    assert prm.derived_backend_kernel("flash_attn") == "complete"
    # A primitive no family claims has no route, and says so rather than
    # guessing.
    assert prm.derived_backend_kernel("digamma") == "planned"


def test_the_derived_axis_says_nothing_about_device_evidence():
    """Two independent facts must not collapse into one enum.

    Whether a route exists and whether it was proven on silicon are different
    claims with different surfaces (Decision #26). Folding them together is how
    the old `backend_kernel` became a constant; keeping them apart is why this
    one can stay meaningful.
    """
    import inspect

    source = inspect.getsource(prm.derived_backend_kernel)
    for forbidden in ("runtime_execution_matrix", "device", "hardware_verified"):
        assert f"{forbidden}(" not in source, (
            f"derived_backend_kernel consults {forbidden} -- route and evidence "
            f"are separate claims")


def test_non_pattern_families_are_named_rather_than_skipped():
    """`breadth` and `cohort2` are x86 benchmark cohorts, not op patterns.

    They are listed explicitly so a reader sees that the x86 classifier mixes
    two vocabularies. Skipping unknown families silently would have hidden it,
    and would also have let a real family slip through as 'probably a cohort'.
    """
    assert prm.NON_PATTERN_FAMILIES == frozenset({"breadth", "cohort2"})
    for family in prm.NON_PATTERN_FAMILIES:
        assert family not in prm.FAMILY_MEMBERS
        assert family in prm.declared_families()
