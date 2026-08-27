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


def test_region_digest_is_order_independent_for_repeated_op_names():
    first = _rng(occurrence_id="bb0.op0")
    second = _rng(
        occurrence_id="bb0.op7", key={"seed": 7, "path": ("dropout", 1)}
    )
    assert region_digest([first, second]) == region_digest([second, first])


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


# ── E2: admissibility is a property of the CALL FORM, not the op ────────────

def test_only_reproducible_call_forms_get_a_product():
    """Measured on the real op, then encoded here. `tessera.dropout` has
    three call forms and only one of them replays from recorded data."""
    from tessera.compiler.recorded_product import stochastic_product_for_call

    ok = stochastic_product_for_call(
        op="tessera.dropout", occurrence_id="bb0.op0", shape=(4,),
        dtype="f32", seed=7)
    assert ok.effect_class == "keyed_rng"
    assert ok.product["key"]["seed"] == 7

    with pytest.raises(RecordedProductError, match="ambient entropy"):
        stochastic_product_for_call(
            op="tessera.dropout", occurrence_id="bb0.op0", shape=(4,),
            dtype="f32")
    with pytest.raises(RecordedProductError, match="advances per call"):
        stochastic_product_for_call(
            op="tessera.dropout", occurrence_id="bb0.op0", shape=(4,),
            dtype="f32", generator=object())
    with pytest.raises(RecordedProductError, match="not both"):
        stochastic_product_for_call(
            op="tessera.dropout", occurrence_id="bb0.op0", shape=(4,),
            dtype="f32", seed=1, key={"seed": 2})


def test_the_call_form_verdicts_match_the_measured_behaviour():
    """The classifier's verdicts are not a convention: each is what the op
    actually does. This is the check that keeps the table honest if the
    reference implementation ever changes."""
    import numpy as np

    from tessera import ops

    x = np.ones((256,), dtype=np.float32)

    # admitted form: replay is bit-identical
    assert np.array_equal(ops.dropout(x, 0.3, seed=11),
                          ops.dropout(x, 0.3, seed=11))

    # refused form 1: ambient entropy does not replay
    assert not np.array_equal(ops.dropout(x, 0.3), ops.dropout(x, 0.3))

    # refused form 2: a caller-owned generator advances between calls
    class _Wrap:
        def __init__(self, g):
            self.g = g

        def _generator(self):
            return self.g

    shared = _Wrap(np.random.default_rng(5))
    assert not np.array_equal(ops.dropout(x, 0.3, rng=shared),
                              ops.dropout(x, 0.3, rng=shared))


# ── E2b: the adjoint the recorded product licenses ──────────────────────────

def test_keyed_dropout_jacobian_is_diagonal_so_its_adjoint_is_itself():
    """W4-EFFECTS-1 E2b. The compiler emits `dx = dropout(dout, same key)`.
    That is the adjoint only because the Jacobian is DIAGONAL — a diagonal
    operator equals its own transpose — and because the mask REPLAYS from the
    key. Both are checked exactly; the pairing identity is then checked in
    float64, since two dot products over different vectors accumulate in
    different orders and bit equality there would be a statement about
    summation, not about the operator.
    """
    import numpy as np

    from tessera import ops

    n, p, seed = 4096, 0.25, 7
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n).astype(np.float32)
    ones = np.ones(n, dtype=np.float32)

    # the mask is a function of the key
    mask = ops.dropout(ones, p, seed=seed)
    assert np.array_equal(mask, ops.dropout(ones, p, seed=seed))

    # the forward is exactly elementwise scaling by that mask
    assert np.array_equal(ops.dropout(x, p, seed=seed), x * mask)

    # J v == diag(mask) v, EXACTLY -> J is diagonal -> J^T = J
    v = rng.standard_normal(n).astype(np.float32)
    u = rng.standard_normal(n).astype(np.float32)
    assert np.array_equal(ops.dropout(v, p, seed=seed), mask * v)
    assert np.array_equal(ops.dropout(u, p, seed=seed), mask * u)

    # therefore <Jv, u> == <v, Ju> up to summation order
    lhs = float(np.dot(ops.dropout(v, p, seed=seed).astype(np.float64),
                       u.astype(np.float64)))
    rhs = float(np.dot(v.astype(np.float64),
                       ops.dropout(u, p, seed=seed).astype(np.float64)))
    assert abs(lhs - rhs) / max(abs(lhs), 1e-12) < 1e-7

    # and the emitted adjoint IS the analytic pathwise gradient
    dout = rng.standard_normal(n).astype(np.float32)
    assert np.array_equal(ops.dropout(dout, p, seed=seed), dout * mask)


def test_an_unkeyed_dropout_has_no_reproducible_mask_so_no_adjoint():
    """The guard that makes the adjoint sound. Without a key the backward
    would apply a DIFFERENT mask than the forward — a plausible-looking but
    wrong gradient — so the op must not be differentiated at all. This test
    measures the premise: two ambient draws disagree."""
    import numpy as np

    from tessera import ops

    x = np.ones((512,), dtype=np.float32)
    assert not np.array_equal(ops.dropout(x, 0.3), ops.dropout(x, 0.3))


# ── E3: recorded state binds the VALUE, not only the identity ───────────────

def test_lineage_dtype_is_real_so_mixed_precision_cannot_alias():
    """W4-EFFECTS-1 E3 precondition. `_buffer` hardcoded `dtype="f32"`, so the
    identity could not express a bf16 or fp8 buffer: two materially different
    buffers would have shared a lineage id, and a mutation product binds that
    id, so the aliasing would have reached replay. The default keeps every
    lineage id built today byte-stable."""
    from tessera.compiler.stateful_training import _buffer

    base = dict(name="moment", role="moment", shape=(4, 8), version=1,
                access="read_write")
    f32 = _buffer(**base)["lineage_id"]
    assert f32 == _buffer(**base, dtype="f32")["lineage_id"]   # stable default
    ids = {f32,
           _buffer(**base, dtype="bf16")["lineage_id"],
           _buffer(**base, dtype="fp8_e4m3")["lineage_id"],
           _buffer(**base, dtype="f64")["lineage_id"]}
    assert len(ids) == 4, "dtypes must not alias in the lineage identity"


def test_content_digest_separates_what_metadata_cannot():
    """The digest must see differences the lineage identity is blind to."""
    import numpy as np

    from tessera.compiler.recorded_product import content_digest

    a = np.arange(12, dtype=np.float32)
    assert content_digest(a) == content_digest(a.copy())
    # same dtype and shape, one element changed — invisible to metadata
    changed = a.copy()
    changed[5] = 99.0
    assert content_digest(a) != content_digest(changed)
    # identical BYTES, different shape or interpretation
    assert content_digest(a) != content_digest(a.reshape(3, 4))
    assert content_digest(a) != content_digest(a.view(np.int32))
    assert content_digest(a) != content_digest(a.astype(np.float64))
    # numerically equal, different bits: (R) is bit-identity, so these differ
    signed_zero = a.copy()
    signed_zero[0] = -0.0
    assert np.array_equal(a, signed_zero)          # equal as numbers...
    assert content_digest(a) != content_digest(signed_zero)   # ...not as bits


def test_replay_rejects_changed_bytes_under_an_unchanged_identity():
    """The core E3 claim. Identity matches, version matches, VALUE moved —
    the case that would otherwise reach a gradient computation."""
    import numpy as np

    from tessera.compiler.recorded_product import (
        mutation_product_for_buffer,
        verify_recorded_state,
    )

    moment = np.zeros((4, 8), dtype=np.float32)
    recorded = mutation_product_for_buffer(
        op="tessera.optimizer_step", occurrence_id="bb0.op1",
        lineage_id="a" * 64, version=3, buffer=moment, write_set=("moment",))

    at = dict(lineage_id="a" * 64, version=3)
    verify_recorded_state(recorded, moment, **at)        # unchanged: fine
    verify_recorded_state(recorded, moment.copy(), **at)  # equal bytes: fine

    drifted = moment.copy()
    drifted[2, 3] = 1e-7                                 # one element
    with pytest.raises(RecordedProductError, match="the VALUE changed"):
        verify_recorded_state(recorded, drifted, **at)

    # a different buffer of the SAME metadata shape/dtype is equally rejected
    with pytest.raises(RecordedProductError, match="the VALUE changed"):
        verify_recorded_state(recorded, np.ones((4, 8), dtype=np.float32), **at)


def test_replay_rejects_the_right_bytes_under_the_wrong_identity():
    """PR #630 review, P2 — the other direction, which a content-only check
    cannot see at all.

    Zero-initialised optimizer state is the everyday instance: every lineage's
    first moment is the same bytes, so a content digest cannot distinguish
    them. If a replay reattaches a product to the wrong buffer — or to the
    right buffer at the wrong version — the digests agree and the old check
    called the replay faithful, which is precisely the claim it must not make.
    """
    import numpy as np

    from tessera.compiler.recorded_product import (
        mutation_product_for_buffer,
        verify_recorded_state,
    )

    mine = np.zeros((4, 8), dtype=np.float32)
    theirs = np.zeros((4, 8), dtype=np.float32)      # a DIFFERENT state...
    recorded = mutation_product_for_buffer(
        op="tessera.optimizer_step", occurrence_id="bb0.op1",
        lineage_id="a" * 64, version=3, buffer=mine, write_set=("moment",))

    from tessera.compiler.recorded_product import content_digest
    assert content_digest(mine) == content_digest(theirs)  # ...same bytes

    with pytest.raises(RecordedProductError, match="lineage"):
        verify_recorded_state(recorded, theirs, lineage_id="b" * 64, version=3)
    # right lineage, wrong version — a no-op update leaves bytes identical
    with pytest.raises(RecordedProductError, match="version"):
        verify_recorded_state(recorded, mine, lineage_id="a" * 64, version=4)


def test_mutation_replay_is_bit_identical_end_to_end():
    """Record a stateful step, replay it from the product, require bit
    identity — the acceptance bar the plan sets for E3."""
    import numpy as np

    from tessera.compiler.recorded_product import (
        mutation_product_for_buffer,
        verify_recorded_state,
    )

    def step(param, grad, moment, lr=0.1, beta=0.9):
        new_moment = beta * moment + (1.0 - beta) * grad
        return param - lr * new_moment, new_moment

    rng = np.random.default_rng(3)
    param = rng.standard_normal((4, 8)).astype(np.float32)
    grad = rng.standard_normal((4, 8)).astype(np.float32)
    moment = np.zeros((4, 8), dtype=np.float32)

    new_param, new_moment = step(param, grad, moment)
    recorded = mutation_product_for_buffer(
        op="tessera.optimizer_step", occurrence_id="bb0.op1",
        lineage_id="a" * 64, version=1, buffer=new_moment,
        write_set=("moment",))

    replay_param, replay_moment = step(param, grad, moment)
    assert np.array_equal(new_param, replay_param)
    assert np.array_equal(new_moment, replay_moment)
    at = dict(lineage_id="a" * 64, version=1)
    verify_recorded_state(recorded, replay_moment, **at)  # the product agrees

    # a replay that started from the wrong version state is caught
    _, wrong = step(param, grad, moment + np.float32(1e-6))
    with pytest.raises(RecordedProductError, match="the VALUE changed"):
        verify_recorded_state(recorded, wrong, **at)


def test_verify_recorded_state_rejects_the_wrong_effect_class():
    from tessera.compiler.recorded_product import verify_recorded_state

    import numpy as np

    with pytest.raises(RecordedProductError, match="applies to"):
        verify_recorded_state(_rng(), np.zeros(4, dtype=np.float32),
                              lineage_id="a" * 64, version=0)


# ── E4: ordered collectives — order AND tree ────────────────────────────────

def _collective_record(sequence=("all_reduce:0", "all_gather:1", "all_reduce:2"),
                       algorithm="ring_f32_pairwise_v1",
                       topology=None):
    from tessera.compiler.recorded_product import collective_product_for_sequence

    return collective_product_for_sequence(
        op="tessera.all_reduce", occurrence_id="bb0.op2",
        communicator="dp:0-7", sequence=sequence,
        reduction_algorithm=algorithm,
        topology=topology if topology is not None else {"ranks": 8, "chunks": 4},
        write_set=("grad",))


def test_collective_replay_requires_the_same_order():
    from tessera.compiler.recorded_product import verify_collective_replay

    recorded = _collective_record()
    order = ["all_reduce:0", "all_gather:1", "all_reduce:2"]
    verify_collective_replay(recorded, order,
                             reduction_algorithm="ring_f32_pairwise_v1",
                             topology={"ranks": 8, "chunks": 4})

    permuted = ["all_gather:1", "all_reduce:0", "all_reduce:2"]
    with pytest.raises(RecordedProductError, match="different collective sequence"):
        verify_collective_replay(recorded, permuted,
                                 reduction_algorithm="ring_f32_pairwise_v1",
                                 topology={"ranks": 8, "chunks": 4})


def test_a_changed_reduction_tree_fails_closed_even_when_the_order_matches():
    """The case order-only checking cannot see. Floating-point addition is not
    associative, so the tree is part of the value — verified numerically in
    `test_reduction_tree_changes_the_bits` below, which is WHY the product
    binds the algorithm and the topology that selects it."""
    from tessera.compiler.recorded_product import verify_collective_replay

    recorded = _collective_record()
    order = ["all_reduce:0", "all_gather:1", "all_reduce:2"]

    with pytest.raises(RecordedProductError, match="not associative"):
        verify_collective_replay(recorded, order,
                                 reduction_algorithm="tree_f32_v2",
                                 topology={"ranks": 8, "chunks": 4})
    with pytest.raises(RecordedProductError, match="topology selects the tree"):
        verify_collective_replay(recorded, order,
                                 reduction_algorithm="ring_f32_pairwise_v1",
                                 topology={"ranks": 4, "chunks": 4})


def test_reduction_tree_changes_the_bits():
    """The measurement the E4 contract rests on: identical inputs and an
    identical issue order still give different results under different
    reduction trees, so binding the order alone would not give (R)."""
    import numpy as np

    rng = np.random.default_rng(0)
    values = (rng.standard_normal(1024).astype(np.float32) * np.float32(1e3))

    def sequential(v):
        acc = np.float32(0)
        for element in v:
            acc = np.float32(acc + element)
        return acc

    def pairwise(v):
        v = v.copy()
        while len(v) > 1:
            if len(v) % 2:
                v = np.append(v, np.float32(0))
            v = (v[0::2] + v[1::2]).astype(np.float32)
        return v[0]

    def ring(v, partials):
        return sequential(np.array([sequential(c) for c in
                                    np.array_split(v, partials)],
                                   dtype=np.float32))

    results = {sequential(values), pairwise(values), ring(values, 8)}
    assert len(results) == 3, "expected three distinct bit patterns"
    # and the ring result depends on the rank count, i.e. on the topology
    assert ring(values, 2) != ring(values, 8)


def test_empty_sequence_records_no_order():
    from tessera.compiler.recorded_product import collective_product_for_sequence

    with pytest.raises(RecordedProductError, match="non-empty sequence"):
        collective_product_for_sequence(
            op="tessera.all_reduce", occurrence_id="bb0.op2",
            communicator="dp:0-7", sequence=(),
            reduction_algorithm="ring_f32_pairwise_v1",
            topology={"ranks": 8}, write_set=("grad",))


def test_collective_product_records_the_real_mock_mesh_order():
    """E4 against the actual W5.4 executor rather than a synthetic list: run a
    placement graph on the deterministic mock mesh, record the order it really
    issued, and require the replay to reproduce it. A permutation of that same
    order is rejected.

    Scope, stated because it bounds the claim: this establishes ORDER. Bit
    identity of a collective RESULT needs native deterministic evidence on
    real transport — a mock mesh cannot provide it, and E4 does not claim it.
    """
    import numpy as np

    from tessera.compiler.graph_ir import GraphIRFunction, IRArg, IROp, tensor_ir_type
    from tessera.compiler.recorded_product import (
        collective_product_for_sequence,
        verify_collective_replay,
    )
    from tessera.compiler.sharding_propagation import (
        execute_resharded_graph_on_mock_mesh,
    )

    ty = tensor_ir_type(("4", "2"), "fp32")
    op = IROp(result="out", op_name="tessera.all_reduce", operands=["%x"],
              operand_types=[str(ty)], result_type=str(ty), inferred_type=ty,
              kwargs={"axis": "data", "op": "sum"})
    fn = GraphIRFunction("mock_collective", args=[IRArg("x", ty)],
                         result_types=[ty], body=[op], return_values=["%out"])
    ranks = [np.arange(8, dtype=np.float32).reshape(4, 2),
             np.arange(8, dtype=np.float32).reshape(4, 2) + 10]

    execution = execute_resharded_graph_on_mock_mesh(
        fn, {"x": ranks}, mesh_shape={"data": 2})
    issued = list(execution.executed_reshards)
    assert issued, "the mock mesh should have issued at least one movement"

    recorded = collective_product_for_sequence(
        op="tessera.all_reduce", occurrence_id="bb0.op0",
        communicator="data:0-1", sequence=issued,
        reduction_algorithm="mock_mesh_sum_v1",
        topology={"ranks": 2, "axis": "data"}, write_set=("out",))

    # replaying the same graph issues the same order
    replay = execute_resharded_graph_on_mock_mesh(
        fn, {"x": ranks}, mesh_shape={"data": 2})
    verify_collective_replay(recorded, list(replay.executed_reshards),
                             reduction_algorithm="mock_mesh_sum_v1",
                             topology={"ranks": 2, "axis": "data"})

    # a permutation of the very same movements is refused
    if len(issued) > 1:
        with pytest.raises(RecordedProductError,
                           match="different collective sequence"):
            verify_collective_replay(recorded, list(reversed(issued)),
                                     reduction_algorithm="mock_mesh_sum_v1",
                                     topology={"ranks": 2, "axis": "data"})
    # and the tree is bound even though the mock cannot prove result bits
    with pytest.raises(RecordedProductError, match="not associative"):
        verify_collective_replay(recorded, issued,
                                 reduction_algorithm="tree_v2",
                                 topology={"ranks": 2, "axis": "data"})
