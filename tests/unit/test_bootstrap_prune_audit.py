"""The prune gap analysis must fail closed, not report a false 'covered'.

This dashboard exists to decide what may safely be deleted. Its dangerous
failure is not being wrong loudly — it is reporting a family as already
served by the mainline compiler when it is not, because acting on that
deletes the family's only lowering. Every test here guards that direction.
"""

from __future__ import annotations

import pytest

from tessera.compiler import bootstrap_prune_audit as audit


def test_declared_compiled_routes_all_exist():
    """A renamed predicate must raise, never silently mark a family covered."""
    audit.verify_declared_mapping()  # raises on drift


def test_a_stale_declared_route_fails_closed(monkeypatch):
    """The guard above is only worth having if it actually fires."""
    monkeypatch.setitem(
        audit._FAMILY_TO_COMPILED, "matmul", ("scheduled_matmul", "no_such_predicate")
    )
    with pytest.raises(RuntimeError, match="no longer exist"):
        audit.verify_declared_mapping()


def test_bootstrap_is_classified_by_graph_ir_re_entry_not_by_name():
    """`package_scheduled_*` consumes a lowered artifact and is NOT a target.

    Classifying by name suffix counted six compiled-route packagers as
    bootstrap surface — an overstatement that would have made the prune look
    larger than it is. The real discriminator is whether the function takes a
    `GraphIRModule` (re-enters Graph IR, bypassing Schedule/Tile) or an
    already-lowered `Scheduled*Artifact`.
    """
    assert audit._is_bootstrap("GraphIRModule") is True
    assert audit._is_bootstrap("ScheduledAttentionArtifact") is False
    assert audit._is_bootstrap("") is False

    by_target = {inv.target: inv for inv in audit.collect_inventories()}
    rocm = by_target["rocm_gfx1151"]
    assert "package_scheduled_attention" in rocm.compiled_packagers
    assert "package_scheduled_attention" not in rocm.bootstrap
    assert "package_softmax" in rocm.bootstrap


def test_every_packager_lands_in_exactly_one_population():
    """No packager may be silently dropped from the accounting."""
    for inv in audit.collect_inventories():
        assert len(inv.bootstrap) + len(inv.compiled_packagers) + len(inv.unclassified_packagers) == len(inv.packagers)
        assert not set(inv.bootstrap) & set(inv.compiled_packagers)


def test_gap_rows_are_only_ever_families_with_no_declared_route():
    """`gap` must mean exactly one thing, or the table cannot be acted on."""
    for _target, family, route, status in audit.family_rows():
        if status == "gap":
            assert audit._target_family_route(_target, family) is None
            assert route == "—"
        else:
            assert status == "compiled"
            assert family in audit._FAMILY_TO_COMPILED


def test_the_analysis_is_not_vacuous():
    """A gap table that found nothing would pass every check above.

    The prune has not started, so both populations must be non-empty; if this
    ever fails because `gap` reached zero, the guard should be replaced by the
    evidence that closed it rather than deleted.
    """
    s = audit.summary()
    assert s["backends"] >= 3
    assert s["bootstrap"] > 0, "no bootstrap packagers found — parsing likely broke"
    assert s["families"] > 0
    assert s["compiled"] > 0, "no family resolves to a compiled route"


def test_markdown_states_the_two_legitimate_exits():
    """The doc must not read as a delete-list."""
    text = audit.render_markdown()
    assert "Absorbed" in text and "Re-expressed" in text
    assert "abi_call" in text
    assert "Decision #31" in text


def test_packager_kind_detects_every_backends_spelling():
    """A detector keyed to one backend's helper name reports mostly nothing.

    NVIDIA spells it `_compile_tile_ir`, ROCm `_compile_attention_tile_ir`,
    x86 `emit_matmul_tile_ir` plus a direct `tessera-opt` invocation. Keying
    on the first literal classified the other two as `other`, which made the
    taxonomy useless rather than wrong-looking.
    """
    assert audit._packager_kind("x = _compile_tile_ir(t)") == "constructs_tile_ir"
    assert audit._packager_kind("_compile_attention_tile_ir(m)") == "constructs_tile_ir"
    assert audit._packager_kind("emit_matmul_tile_ir(e)") == "constructs_tile_ir"
    assert audit._packager_kind('run(["tessera-opt"])') == "constructs_tile_ir"
    assert audit._packager_kind("load('libtessera_x.so')") == "delegates"
    assert audit._packager_kind("emit_matmul_tile_ir(e); nvrtc(x)") == "both"
    assert audit._packager_kind("return _dispatch(module)") == "other"


def test_every_bootstrap_packager_is_classified():
    """No packager may fall out of the kind accounting unnoticed."""
    for inv in audit.collect_inventories():
        assert set(inv.kinds) == set(inv.bootstrap)
        assert all(
            k in {"constructs_tile_ir", "delegates", "both", "other"}
            for k in inv.kinds.values()
        )


def test_the_surface_is_mostly_ir_constructing_not_delegating():
    """The finding that redirected step 2, asserted so a regression is visible.

    If this ever flips, the prune plan changes shape: absorption work becomes
    delegation-migration work. Replace this guard with the evidence that
    flipped it rather than deleting it.
    """
    s = audit.summary()
    constructing = s["constructs_tile_ir"] + s["both"]
    assert constructing > s["delegates"] * 5, (
        f"{constructing} IR-constructing vs {s['delegates']} delegating — the "
        "bootstrap surface's character changed; re-scope the prune"
    )


def test_unknown_inputs_do_not_claim_compiled_artifact_consumption():
    for annotation in ('', 'Any', 'str', 'UnrelatedArtifact'):
        assert not audit._is_artifact(annotation)
    assert audit._is_artifact('ScheduledMatmulArtifact')
    inventory = audit.BackendInventory('test', 'fixture.py',
        (('package_a', ''), ('package_b', 'Any'), ('package_c', 'GraphIRModule'),
         ('package_d', 'ScheduledKernelArtifact')), (), 1)
    assert inventory.unclassified_packagers == ('package_a', 'package_b')
    assert inventory.compiled_packagers == ('package_d',)


def test_apple_inventory_exposes_computed_classifier_limits():
    inventories = {i.target: i for i in audit.collect_inventories()}
    gpu = inventories['apple_gpu']
    assert 'package_scheduled_kernel' in gpu.compiled_packagers
    assert 'package_native' in gpu.bootstrap
    assert 'softmax' in gpu.families
    assert not gpu.unresolved_returns
    assert not inventories['apple_cpu'].unresolved_returns
    assert 'cholesky' in inventories['apple_cpu'].families
    assert 'value_ebm_energy_quadratic' in gpu.families
    text = audit.render_markdown()
    assert 'Unresolved classifier return' in text
    assert 'unclassified/raw inputs' in text


def test_missing_backend_source_cannot_silently_shrink_census(monkeypatch):
    monkeypatch.setattr(audit, '_BACKEND_MODULES', (('missing', 'missing_module.py'),))
    with pytest.raises(ValueError, match='cannot inventory'):
        audit.collect_inventories()


def test_computed_classifier_domain_comes_from_live_producer_tables():
    import ast
    tree = ast.parse("""
_VALUE_SYMBOLS = {'tessera.new_solver': ('entry', 'symbol')}
_LOW_PRECISION_MATMUL_SYMBOLS = {}
def native_package_kind(module):
    if _entry_for(op.op_name, dtype) is None:
        return None
    return op.op_name.removeprefix('tessera.')
""")
    assert audit._classified_families(tree) == ('new_solver',)
    tree.body[-1].body[-1].value = ast.parse('unknown_classifier(module)', mode='eval').body
    assert audit._classified_families(tree) == ()
    assert audit._unresolved_classifier_returns(tree) == ('unknown_classifier(module)',)


def test_family_route_requires_consumer_on_that_target():
    assert audit._target_family_route('apple_cpu', 'matmul') is None
    assert audit._target_family_route('apple_gpu', 'matmul') is not None
    assert audit._target_family_route('nvidia_sm120', 'depth_attention') is None
    assert audit._target_family_route('rocm_gfx1151', 'depth_attention') is not None
