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
    for family, per_target in prm.FAMILY_MEMBERS.items():
        for members in per_target.values():
            for member in members:
                assert member in coverage, f"{family}: {member!r}"
                assert coverage[member].graph_name, (
                    f"{family}: {member!r} has no graph_name")


def test_a_member_that_is_not_a_primitive_is_refused(monkeypatch):
    monkeypatch.setattr(prm, "FAMILY_MEMBERS",
                        {**prm.FAMILY_MEMBERS, "matmul": {"*": ("not_a_primitive",)}})
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
                        {**prm.FAMILY_MEMBERS, "retired_family": {"*": ("matmul",)}})
    with pytest.raises(ValueError, match="stale declaration"):
        prm.verify_family_membership()


def test_an_empty_declaration_is_refused(monkeypatch):
    """An empty member tuple would make a family vacuously satisfied."""
    monkeypatch.setattr(prm, "FAMILY_MEMBERS", {**prm.FAMILY_MEMBERS, "matmul": {}})
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
    assert kinds <= {prm.ROUTE_COMPILED, prm.ROUTE_BOOTSTRAP, prm.ROUTE_NONE,
                     prm.ROUTE_UNCLASSIFIED}

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


def test_a_member_the_target_never_names_is_refused(monkeypatch):
    """The check that would have caught `min`/`amin` on ROCm and x86.

    A single global member tuple was applied to every backend, publishing
    `min` and `amin` as ROCm/x86 reduction routes although those contracts
    accept only sum/mean/max/amax. The guard now cross-checks every declared
    member against the `tessera.*` literals of that target's own native module.
    """
    monkeypatch.setattr(prm, "FAMILY_MEMBERS", {
        **prm.FAMILY_MEMBERS,
        "reduction": {**prm.FAMILY_MEMBERS["reduction"],
                      "rocm_gfx1151": ("sum", "mean", "max", "amax", "amin")},
    })
    with pytest.raises(ValueError, match="contract does not name it"):
        prm.verify_family_membership()


def test_a_backend_alias_spelling_is_accepted():
    """A backend may name the canonical op OR the public alias.

    Coverage records `sum -> tessera.reduce` while `x86_native` names
    `tessera.sum`. Accepting one spelling only would reject a real route --
    the check must catch over-claims without manufacturing them.
    """
    from tessera.compiler.primitive_coverage import primitive_graph_names

    assert primitive_graph_names()["sum"] == "tessera.reduce"
    literals = prm.target_op_literals("x86")
    assert "tessera.sum" in literals and "tessera.reduce" not in literals
    prm.verify_family_membership()          # must not raise


def test_a_compiled_route_is_not_fanned_across_targets():
    """`depth_attention` is dispatched only for rocm_gfx1151 in driver.py.

    An earlier version fanned any compiled family with no classifier across
    EVERY discovered target, publishing `depth_attn` as compiled on NVIDIA and
    x86. The comment justifying it claimed the fan-out "keeps the claim no
    stronger than the source"; the source is target-aware, so it made the claim
    strictly stronger.
    """
    routes = prm.primitive_routes()["depth_attn"]
    assert routes["rocm_gfx1151"] == prm.ROUTE_COMPILED
    assert routes["nvidia_sm120"] == prm.ROUTE_NONE
    assert routes["x86"] == prm.ROUTE_NONE


def test_a_target_that_cannot_be_classified_says_so_rather_than_vanishing():
    """`unclassified` and `none` are different claims.

    Apple dropped out of an earlier map entirely: `apple_cpu`'s
    `native_package_kind` returns a computed expression so the AST walker finds
    no families, and `apple_gpu` is not in the audit at all -- yet `driver.py`
    has live scheduled dispatch for it. Absent columns read as "unserved",
    which is the `unmeasured` failure in a new place.
    """
    assert set(prm.UNCLASSIFIABLE_TARGETS) == {"apple_cpu", "apple_gpu"}
    assert all(reason for reason in prm.UNCLASSIFIABLE_TARGETS.values())
    targets = prm.all_targets()
    assert "apple_cpu" in targets and "apple_gpu" in targets
    for per_target in prm.primitive_routes().values():
        for target in prm.UNCLASSIFIABLE_TARGETS:
            assert per_target[target] == prm.ROUTE_UNCLASSIFIED


def test_a_silently_unclassifiable_target_is_refused(monkeypatch):
    """Dropping a target from UNCLASSIFIABLE_TARGETS must fail, not hide it."""
    monkeypatch.setattr(prm, "UNCLASSIFIABLE_TARGETS", {})
    with pytest.raises(ValueError, match="vanish from the map silently"):
        prm.verify_family_membership()


def test_a_stale_compiled_route_restriction_is_refused(monkeypatch):
    monkeypatch.setattr(prm, "COMPILED_ROUTE_TARGETS",
                        {**prm.COMPILED_ROUTE_TARGETS, "softmax": ("x86",)})
    with pytest.raises(ValueError, match="stale restriction"):
        prm.verify_family_membership()
