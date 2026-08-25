"""W4-EFFECTS-1 slice E1 — the recorded-product carrier and its verifier.

The plan (`docs/audit/compiler/W4_ADMISSIBLE_EFFECTS_PLAN.md`) admits an
effectful op into a differentiated region only when a recorded product makes
replay a function of that product. These tests hold the contract to its two
halves:

* **(R) Reproducibility** — the fields that make replay deterministic are
  REQUIRED, per class. The two that were measured to be insufficient as
  names get their own tests: a mutation without a `content_digest` and a
  collective without a `reduction_algorithm` are rejected, because metadata
  identity and issue order respectively do not determine the value.
* **(C) Confinement** — a write outside the declared set is rejected, and a
  class that writes nothing may not declare a write-set.

Plus the property that keeps "admissible" from meaning "unchecked": every
effectful op on the path needs exactly one product, and an op with none stays
fail-closed.
"""

from __future__ import annotations

import pytest

from tessera.compiler.recorded_product import (
    ADMITTED_EFFECT_CLASSES,
    RECORDED_PRODUCT_SCHEMA,
    RecordedProduct,
    RecordedProductError,
    region_digest,
    verify_confinement,
    verify_region_products,
)


def _rng(**over):
    payload = {"key": {"seed": 7, "path": ("dropout", 0)}, "shape": (4, 8),
               "dtype": "f32"}
    payload.update(over)
    return RecordedProduct(op="tessera.dropout", effect_class="keyed_rng",
                           product=payload)


def _mutation(**over):
    payload = {"lineage_id": "a" * 64, "version": 3, "content_digest": "b" * 64}
    payload.update(over)
    return RecordedProduct(op="tessera.optimizer_step",
                           effect_class="recorded_mutation", product=payload,
                           write_set=("moment",))


def _collective(**over):
    payload = {"communicator": "dp:0-7", "sequence_digest": "c" * 64,
               "reduction_algorithm": "ring_f32_pairwise_v1",
               "topology": {"ranks": 8, "chunks": 4}}
    payload.update(over)
    return RecordedProduct(op="tessera.all_reduce",
                           effect_class="ordered_collective", product=payload,
                           write_set=("grad",))


# ── (R): the classes and their required fields ──────────────────────────────

def test_admitted_classes_exclude_io():
    """I/O is closed by ARGUMENT — an external read is not a function of any
    recorded value — so it must not be reachable as a class at all."""
    assert "io" not in ADMITTED_EFFECT_CLASSES
    assert set(ADMITTED_EFFECT_CLASSES) == {
        "keyed_rng", "recorded_mutation", "ordered_collective", "observational"}
    with pytest.raises(RecordedProductError, match="not admissible"):
        RecordedProduct(op="tessera.read_file", effect_class="io",
                        product={"path": "/tmp/x"})


def test_every_class_constructs_with_its_required_fields():
    for build in (_rng, _mutation, _collective):
        item = build()
        assert item.schema == RECORDED_PRODUCT_SCHEMA
        assert len(item.digest) == 64


@pytest.mark.parametrize("missing", ["key", "shape", "dtype"])
def test_keyed_rng_requires_its_key_material(missing):
    with pytest.raises(RecordedProductError, match="reproducibility"):
        _rng(**{missing: None})


def test_unkeyed_rng_is_rejected():
    """The split the plan names: keyed draws are admissible because the S4
    generator is counter-based; an unkeyed draw has no product at all."""
    with pytest.raises(RecordedProductError, match="reproducibility"):
        _rng(key=None)


def test_mutation_requires_a_content_digest_not_only_identity():
    """Plan section 3.2, corrected after review. `state_buffer_lineage`
    hashes name/role/shape/dtype/version/access/parents and NOT contents, so
    identical metadata with different bytes shares an id: binding
    "version N" alone lets a replay bind different bytes and produce a
    different gradient silently."""
    _mutation()  # with the digest: fine
    with pytest.raises(RecordedProductError, match="content_digest"):
        _mutation(content_digest=None)


def test_collective_requires_its_reduction_algorithm_not_only_order():
    """Plan section 3.3, corrected after review. Floating-point addition is
    not associative, so the reduction tree is part of the value: identical
    inputs under an identical issue order give different bits for
    sequential, pairwise, and ring reductions."""
    _collective()  # with the tree: fine
    with pytest.raises(RecordedProductError, match="reduction_algorithm"):
        _collective(reduction_algorithm=None)
    with pytest.raises(RecordedProductError, match="topology"):
        _collective(topology=None)


def test_empty_product_is_rejected():
    with pytest.raises(RecordedProductError, match="non-empty"):
        RecordedProduct(op="tessera.dropout", effect_class="keyed_rng",
                        product={})


# ── (C): confinement ────────────────────────────────────────────────────────

def test_write_outside_the_declared_set_is_rejected():
    item = _mutation()
    verify_confinement(item, ["moment"])          # declared
    verify_confinement(item, [])                  # wrote less: fine
    with pytest.raises(RecordedProductError, match="confinement"):
        verify_confinement(item, ["moment", "neighbour_state"])


def test_over_declaration_is_allowed_but_stray_writes_are_not():
    """Declaring more than written is conservative and cannot hide a stray
    write; the converse is exactly what (C) forbids."""
    item = RecordedProduct(op="tessera.optimizer_step",
                           effect_class="recorded_mutation",
                           product={"lineage_id": "a"*64, "version": 1,
                                    "content_digest": "b"*64},
                           write_set=("moment", "variance"))
    verify_confinement(item, ["moment"])
    with pytest.raises(RecordedProductError, match="confinement"):
        verify_confinement(item, ["moment", "params"])


@pytest.mark.parametrize("cls,payload", [
    ("observational", {"origin": "compiler_extent_assertion"}),
    ("keyed_rng", {"key": {"seed": 1}, "shape": (2,), "dtype": "f32"}),
])
def test_non_writing_classes_may_not_declare_a_write_set(cls, payload):
    RecordedProduct(op="op", effect_class=cls, product=payload)  # no write-set
    with pytest.raises(RecordedProductError, match="writes nothing"):
        RecordedProduct(op="op", effect_class=cls, product=payload,
                        write_set=("something",))


def test_duplicate_write_set_names_are_rejected():
    with pytest.raises(RecordedProductError, match="duplicate"):
        RecordedProduct(op="tessera.optimizer_step",
                        effect_class="recorded_mutation",
                        product={"lineage_id": "a"*64, "version": 1,
                                 "content_digest": "b"*64},
                        write_set=("moment", "moment"))


# ── region-level totality ───────────────────────────────────────────────────

def test_every_effectful_op_needs_exactly_one_product():
    rng, mutation = _rng(), _mutation()
    by_op = verify_region_products(
        [rng, mutation],
        effectful_ops=["tessera.dropout", "tessera.optimizer_step"])
    assert set(by_op) == {"tessera.dropout", "tessera.optimizer_step"}

    # an effectful op with no product stays fail-closed
    with pytest.raises(RecordedProductError, match="without a recorded product"):
        verify_region_products(
            [rng],
            effectful_ops=["tessera.dropout", "tessera.optimizer_step"])
    # a product for an op that is not on the path is equally wrong
    with pytest.raises(RecordedProductError, match="not on the region"):
        verify_region_products([rng, mutation], effectful_ops=["tessera.dropout"])
    # two products for one op
    with pytest.raises(RecordedProductError, match="more than one"):
        verify_region_products([rng, _rng()], effectful_ops=["tessera.dropout"])


# ── identity: the digest must separate everything that matters ──────────────

def test_digest_separates_every_field_that_changes_the_value():
    """The failure mode this guards against is the one measured twice
    already (the MegaMoE schedule digest, the f32-only lineage): two
    materially different products sharing a content address."""
    base = _mutation().digest
    assert _mutation(version=4).digest != base
    assert _mutation(content_digest="c" * 64).digest != base
    assert _mutation(lineage_id="d" * 64).digest != base
    assert RecordedProduct(
        op="tessera.optimizer_step", effect_class="recorded_mutation",
        product={"lineage_id": "a"*64, "version": 3, "content_digest": "b"*64},
        write_set=("variance",)).digest != base          # write-set counts
    assert _rng().digest != _rng(shape=(4, 9)).digest
    assert _rng().digest != _rng(dtype="bf16").digest
    assert _collective().digest != _collective(
        reduction_algorithm="tree_f32_v2").digest
    assert _collective().digest != _collective(
        topology={"ranks": 4, "chunks": 4}).digest


def test_digest_is_stable_and_order_independent_at_region_level():
    rng, mutation, coll = _rng(), _mutation(), _collective()
    assert region_digest([rng, mutation, coll]) == region_digest(
        [coll, rng, mutation])
    assert region_digest([rng, mutation]) != region_digest([rng, mutation, coll])


def test_mlir_attr_carries_the_digest_and_escapes_its_payload():
    attr = _collective().to_mlir_attr()
    assert f'tessera.recorded_product.digest = "{_collective().digest}"' in attr
    assert 'tessera.recorded_product.effect_class = "ordered_collective"' in attr
    # the JSON payload is embedded as a string attribute, so its quotes escape
    assert '\\"reduction_algorithm\\"' in attr


# ── (R) demonstrated, not just declared ─────────────────────────────────────

def test_keyed_rng_product_actually_reproduces_the_draw_bit_for_bit():
    """The carrier is only worth having if the product it records really does
    determine replay. This closes the loop for the keyed-RNG class against
    the live S4 generator: record the key, throw the values away, rebuild the
    draw from the product alone, and require BIT identity — the acceptance
    bar the plan sets, which a distributional check cannot establish.
    """
    import numpy as np

    from tessera.rng import RNGKey, normal

    seed, path, shape = 20260825, ("dropout", 3), (256,)

    # Record: run the effect and capture only its product.
    key = RNGKey(seed).fold_in(path[0]).fold_in(path[1])
    recorded_values = normal(key, shape)
    product = RecordedProduct(
        op="tessera.dropout", effect_class="keyed_rng",
        product={"key": {"seed": seed, "path": list(path)},
                 "shape": list(shape), "dtype": "f32"})

    # Replay: rebuild the draw from the product alone.
    payload = product.product
    replay_key = RNGKey(payload["key"]["seed"])
    for element in payload["key"]["path"]:
        replay_key = replay_key.fold_in(element)
    replayed = normal(replay_key, tuple(payload["shape"]))

    assert np.array_equal(recorded_values, replayed), "replay diverged"
    assert recorded_values.dtype == replayed.dtype
    # A different product must NOT reproduce it — otherwise the test is vacuous.
    other = RNGKey(seed).fold_in(path[0]).fold_in(path[1] + 1)
    assert not np.array_equal(recorded_values, normal(other, shape))
