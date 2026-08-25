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
    occurrence = over.pop("occurrence_id", "bb0.op0")
    payload = {"key": {"seed": 7, "path": ("dropout", 0)}, "shape": (4, 8),
               "dtype": "f32"}
    payload.update(over)
    over["occurrence_id"] = occurrence
    return RecordedProduct(op="tessera.dropout", occurrence_id=over.pop("occurrence_id", "bb0.op0"),
                           effect_class="keyed_rng", product=payload)


def _mutation(**over):
    payload = {"lineage_id": "a" * 64, "version": 3, "content_digest": "b" * 64}
    payload.update(over)
    return RecordedProduct(op="tessera.optimizer_step", occurrence_id="bb0.op1",
                           effect_class="recorded_mutation", product=payload,
                           write_set=("moment",))


def _collective(**over):
    payload = {"communicator": "dp:0-7", "sequence_digest": "c" * 64,
               "reduction_algorithm": "ring_f32_pairwise_v1",
               "topology": {"ranks": 8, "chunks": 4}}
    payload.update(over)
    return RecordedProduct(op="tessera.all_reduce", occurrence_id="bb0.op2",
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
        RecordedProduct(op="tessera.read_file", occurrence_id="occ", effect_class="io",
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
        RecordedProduct(op="tessera.dropout", occurrence_id="occ",
                        effect_class="keyed_rng", product={})


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
    item = RecordedProduct(op="tessera.optimizer_step", occurrence_id="occ",
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
    RecordedProduct(op="op", occurrence_id="occ", effect_class=cls, product=payload)  # no write-set
    with pytest.raises(RecordedProductError, match="writes nothing"):
        RecordedProduct(op="op", occurrence_id="occ", effect_class=cls, product=payload,
                        write_set=("something",))


def test_duplicate_write_set_names_are_rejected():
    with pytest.raises(RecordedProductError, match="duplicate"):
        RecordedProduct(op="tessera.optimizer_step", occurrence_id="occ",
                        effect_class="recorded_mutation",
                        product={"lineage_id": "a"*64, "version": 1,
                                 "content_digest": "b"*64},
                        write_set=("moment", "moment"))


# ── region-level totality ───────────────────────────────────────────────────

def test_every_effectful_occurrence_needs_exactly_one_product():
    rng, mutation = _rng(), _mutation()          # bb0.op0, bb0.op1
    by_occ = verify_region_products(
        [rng, mutation], effectful_occurrences=["bb0.op0", "bb0.op1"])
    assert set(by_occ) == {"bb0.op0", "bb0.op1"}

    # an effectful occurrence with no product stays fail-closed
    with pytest.raises(RecordedProductError, match="without a recorded product"):
        verify_region_products([rng], effectful_occurrences=["bb0.op0", "bb0.op1"])
    # a product for an occurrence that is not on the path is equally wrong
    with pytest.raises(RecordedProductError, match="not on the"):
        verify_region_products([rng, mutation], effectful_occurrences=["bb0.op0"])
    # two products for one occurrence
    with pytest.raises(RecordedProductError, match="more than one"):
        verify_region_products([rng, _rng()], effectful_occurrences=["bb0.op0"])


def test_two_calls_of_the_same_op_are_two_occurrences():
    """PR #629 review, P1. Keying by op NAME breaks both ways for a region
    containing two `tessera.dropout` calls: the second product is rejected as
    a duplicate, and — worse — a set of names collapses the repetition, so a
    SINGLE product satisfies both occurrences and one effect goes unchecked.
    """
    first = _rng(occurrence_id="bb0.op0")
    second = _rng(occurrence_id="bb0.op7", key={"seed": 7, "path": ("dropout", 1)})
    assert first.op == second.op == "tessera.dropout"
    assert first.digest != second.digest

    by_occ = verify_region_products(
        [first, second], effectful_occurrences=["bb0.op0", "bb0.op7"])
    assert len(by_occ) == 2

    # one product can no longer satisfy two occurrences of the same op
    with pytest.raises(RecordedProductError, match="without a recorded product"):
        verify_region_products(
            [first], effectful_occurrences=["bb0.op0", "bb0.op7"])
    # and a repeated occurrence id is itself rejected: totality would be
    # uncheckable
    with pytest.raises(RecordedProductError, match="repeated"):
        verify_region_products(
            [first], effectful_occurrences=["bb0.op0", "bb0.op0"])


def test_occurrence_id_is_required():
    with pytest.raises(RecordedProductError, match="occurrence id"):
        RecordedProduct(op="tessera.dropout", occurrence_id="",
                        effect_class="keyed_rng",
                        product={"key": {"seed": 1}, "shape": (2,),
                                 "dtype": "f32"})


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
        op="tessera.optimizer_step", occurrence_id="bb0.op1",
        effect_class="recorded_mutation",
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
        op="tessera.dropout", occurrence_id="bb0.op0", effect_class="keyed_rng",
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


# ── review fixes: immutability and the scalar draw ──────────────────────────

def test_product_is_deeply_immutable_and_the_digest_cannot_drift():
    """PR #629 review, P2. A frozen dataclass only stops rebinding the FIELD.
    Without a deep freeze a caller could mutate `product["shape"][0]` after
    construction and move the content address of a carrier that had already
    been indexed or serialized — defeating the point of content addressing.
    """
    item = _rng()
    before = item.digest

    with pytest.raises(TypeError):
        item.product["dtype"] = "bf16"          # mapping is read-only
    with pytest.raises(TypeError):
        item.product["key"]["seed"] = 99        # nested mapping too
    assert isinstance(item.product["shape"], tuple)  # sequences are tuples
    with pytest.raises(TypeError):
        item.product["shape"][0] = 99

    assert item.digest == before
    # the canonical payload is a plain, freshly-built structure: mutating the
    # copy a caller receives must not touch the carrier either
    payload = item.canonical_payload()
    payload["product"]["dtype"] = "bf16"
    assert item.product["dtype"] == "f32"
    assert item.digest == before


def test_scalar_draw_is_a_valid_shape_not_a_missing_field():
    """PR #629 review, P2. `()` is the canonical scalar draw — `rng.normal`
    accepts and defaults to it — so it must not be read as an absent field.
    The equivalent JSON form `[]` must behave the same way."""
    from tessera.rng import RNGKey, normal

    scalar = RecordedProduct(
        op="tessera.dropout", occurrence_id="bb0.op0", effect_class="keyed_rng",
        product={"key": {"seed": 3}, "shape": (), "dtype": "f32"})
    assert scalar.product["shape"] == ()
    json_form = RecordedProduct(
        op="tessera.dropout", occurrence_id="bb0.op0", effect_class="keyed_rng",
        product={"key": {"seed": 3}, "shape": [], "dtype": "f32"})
    assert json_form.digest == scalar.digest      # () and [] are one value

    # and the scalar draw it describes is real
    assert normal(RNGKey(3), ()).shape == ()

    # emptiness that IS meaningless still fails closed
    with pytest.raises(RecordedProductError, match="empty"):
        RecordedProduct(op="tessera.dropout", occurrence_id="bb0.op0",
                        effect_class="keyed_rng",
                        product={"key": {}, "shape": (), "dtype": "f32"})
    with pytest.raises(RecordedProductError, match="empty"):
        _collective(topology={})
