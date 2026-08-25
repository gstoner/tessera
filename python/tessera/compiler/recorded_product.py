"""W4-EFFECTS-1 slice E1 — the recorded-product carrier and its verifier.

`docs/audit/compiler/W4_ADMISSIBLE_EFFECTS_PLAN.md` states when an effectful
operation may enter a differentiated region. This module is that contract in
executable form: one content-addressed carrier per admitted effect, and a
verifier that fails closed rather than trusting the producer.

The criterion, restated because every check below implements one half of it.
An effectful operation ``E`` is admissible iff there is a recorded product
``pi(E)`` such that:

**(R) Reproducibility**
    replay equals the recorded execution *bit-for-bit*, not merely in
    distribution. Each effect class declares which fields make replay a
    function of the product; absent fields fail closed.

**(C) Confinement**
    ``E``'s write-set is contained in values ``pi`` names. A write outside
    the declared set is rejected — (R) alone would admit an operation that
    reproduces its own value while corrupting a neighbour's state.

Two of the four classes bind a *value*, not merely a name, because binding
the name was measured to be insufficient (plan sections 3.2 and 3.3):

* ``recorded_mutation`` must carry a **content digest**. The existing
  ``state_buffer_lineage.v1`` identity hashes name/role/shape/dtype/version/
  access/parents and NOT contents, so two buffers with identical metadata and
  different bytes share an id; a replay binding "version N" could bind
  different bytes and produce a different gradient silently.
* ``ordered_collective`` must carry the **reduction tree/algorithm**. Floating
  point addition is not associative, so the tree is part of the value:
  identical inputs under an identical issue order give different bits for
  sequential, pairwise, and ring reductions. `LANGUAGE_AND_IR_SPEC` section 11
  already requires "fixed collective ordering and reduction trees".

I/O is deliberately absent from the admitted classes. An external read is not
a function of any recorded value, so no product satisfies (R); it is closed by
argument, not pending better bookkeeping.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence


RECORDED_PRODUCT_SCHEMA = "tessera.recorded_product.v1"

#: Effect classes that CAN satisfy (R). `io` is intentionally not a member.
ADMITTED_EFFECT_CLASSES = (
    "keyed_rng",
    "recorded_mutation",
    "ordered_collective",
    "observational",
)

#: Fields each class must carry for replay to be a function of the product.
#: These are the (R) requirements; a missing one is a fail-closed error.
_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    # The S4 generator is counter-based, so a draw is a pure function of its
    # key: the key IS the product. `shape`/`dtype` pin the consumed extent.
    "keyed_rng": ("key", "shape", "dtype"),
    # `lineage_id` + `version` name the buffer; `content_digest` binds what
    # those bytes actually are (plan section 3.2).
    "recorded_mutation": ("lineage_id", "version", "content_digest"),
    # Order alone does not determine the value; the tree does (plan 3.3).
    "ordered_collective": (
        "communicator",
        "sequence_digest",
        "reduction_algorithm",
        "topology",
    ),
    # A compiler-generated observation writes nothing; its only effect is the
    # abort decision, which the STATUS/trap contract already makes explicit.
    "observational": ("origin",),
}

#: Fields whose EMPTINESS is meaningless, per class. `shape` is absent here
#: on purpose: `()` is the scalar draw.
_MUST_BE_NON_EMPTY: dict[str, tuple[str, ...]] = {
    "keyed_rng": ("key",),
    "recorded_mutation": ("lineage_id", "content_digest"),
    "ordered_collective": (
        "communicator",
        "sequence_digest",
        "reduction_algorithm",
        "topology",
    ),
    "observational": ("origin",),
}

#: Classes whose write-set must be empty, by definition of the class.
_MUST_NOT_WRITE = frozenset({"observational", "keyed_rng"})


class RecordedProductError(ValueError):
    """A recorded product does not satisfy (R) or (C)."""


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _freeze(value: Any) -> Any:
    """Deeply immutable snapshot.

    A frozen dataclass only stops rebinding the FIELD; without this the
    caller could still mutate `product["shape"][0]` after construction and,
    because the digest is derived from that state, change the carrier's
    content address after it had been indexed or serialized. Mappings become
    read-only proxies and sequences become tuples all the way down.
    """
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(k): _freeze(v) for k, v in sorted(value.items())}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    """Plain JSON containers for hashing and serialization."""
    if isinstance(value, Mapping):
        return {str(k): _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


@dataclass(frozen=True)
class RecordedProduct:
    """One operation's recorded product: `pi(E)` plus its declared write-set.

    ``product`` is the class-specific recorded value. ``write_set`` is the
    (C) declaration: every value the operation may write, named. Construction
    validates (R)'s required fields and the class's write discipline; the
    dynamic half of (C) is checked by :func:`verify_confinement` when the
    actual write-set is known.
    """

    op: str
    occurrence_id: str
    effect_class: str
    product: Mapping[str, Any]
    write_set: tuple[str, ...] = ()
    schema: str = RECORDED_PRODUCT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RECORDED_PRODUCT_SCHEMA:
            raise RecordedProductError(
                f"unsupported recorded-product schema {self.schema!r}"
            )
        if not self.op:
            raise RecordedProductError("recorded product requires an op name")
        if not self.occurrence_id:
            raise RecordedProductError(
                f"{self.op}: recorded product requires an occurrence id. Two "
                f"calls of one operation in a region share an op NAME, so "
                f"keying by name would reject the second as a duplicate and "
                f"let a single product satisfy both — admitting an unchecked "
                f"effect. Use the region carrier's operation id "
                f"(`StructuredOperation.operation_id`)"
            )
        if self.effect_class not in ADMITTED_EFFECT_CLASSES:
            raise RecordedProductError(
                f"effect class {self.effect_class!r} is not admissible; "
                f"admitted classes are {sorted(ADMITTED_EFFECT_CLASSES)}. An "
                f"external read has no product satisfying reproducibility and "
                f"stays closed by argument, not by omission"
            )
        if not isinstance(self.product, Mapping) or not self.product:
            raise RecordedProductError(
                f"{self.op}: {self.effect_class} requires a non-empty recorded "
                f"product; an effect admitted without one is exactly the "
                f"silent-divergence case the gate exists to prevent"
            )
        # ABSENCE, not emptiness: `shape = ()` is the canonical scalar draw
        # and `tessera.rng.normal` both accepts and defaults to it, so an
        # empty tuple is a legitimate value rather than a missing field.
        # Fields whose emptiness IS meaningless are listed per class below.
        missing = [
            name
            for name in _REQUIRED_FIELDS[self.effect_class]
            if name not in self.product
            or self.product[name] is None
            or self.product[name] == ""
        ]
        if missing:
            raise RecordedProductError(
                f"{self.op}: {self.effect_class} product is missing "
                f"{missing} — replay would not be a function of the product "
                f"(reproducibility)"
            )
        empty = [
            name
            for name in _MUST_BE_NON_EMPTY.get(self.effect_class, ())
            if name in self.product
            and hasattr(self.product[name], "__len__")
            and len(self.product[name]) == 0
        ]
        if empty:
            raise RecordedProductError(
                f"{self.op}: {self.effect_class} product has empty {empty}; "
                f"these fields carry the identity replay depends on "
                f"(reproducibility)"
            )
        if self.effect_class in _MUST_NOT_WRITE and self.write_set:
            raise RecordedProductError(
                f"{self.op}: {self.effect_class} declares a write-set "
                f"{list(self.write_set)} but this class writes nothing beyond "
                f"its own result (confinement)"
            )
        if len(set(self.write_set)) != len(self.write_set):
            raise RecordedProductError(
                f"{self.op}: duplicate names in the declared write-set"
            )
        object.__setattr__(self, "product", _freeze(self.product))
        object.__setattr__(self, "write_set", tuple(self.write_set))
        # Fix the content address at construction. Recomputing it on every
        # access would let any later mutation of nested state move the
        # address of a carrier that has already been indexed or serialized.
        object.__setattr__(self, "_digest", _digest(self.canonical_payload()))

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "op": self.op,
            "effect_class": self.effect_class,
            "op_occurrence": self.occurrence_id,
            "product": _thaw(self.product),
            "write_set": list(self.write_set),
        }

    @property
    def digest(self) -> str:
        """Content address, fixed at construction (see `__post_init__`)."""
        return str(getattr(self, "_digest"))

    def to_mlir_attr(self) -> str:
        """Render as one MLIR dictionary attribute for the region carrier."""
        payload = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":")
        )
        escaped = payload.replace("\\", "\\\\").replace('"', '\\"')
        return (
            f'{{tessera.recorded_product.schema = "{self.schema}", '
            f'tessera.recorded_product.effect_class = "{self.effect_class}", '
            f'tessera.recorded_product.digest = "{self.digest}", '
            f'tessera.recorded_product.payload = "{escaped}"}}'
        )


def verify_confinement(
    recorded: RecordedProduct, actual_write_set: Sequence[str]
) -> None:
    """Check (C): the operation wrote nothing it did not declare.

    Declaring MORE than was written is allowed — an over-declaration is
    conservative and cannot hide a stray write. Writing something undeclared
    is rejected.
    """
    undeclared = sorted(set(actual_write_set) - set(recorded.write_set))
    if undeclared:
        raise RecordedProductError(
            f"{recorded.op}: wrote {undeclared} outside its declared "
            f"write-set {list(recorded.write_set)} (confinement)"
        )


def verify_region_products(
    products: Sequence[RecordedProduct],
    *,
    effectful_occurrences: Sequence[str],
) -> dict[str, RecordedProduct]:
    """Every effectful OCCURRENCE on the path must carry exactly one product.

    Keyed by occurrence, never by op name. A region containing two
    `tessera.dropout` calls has two occurrences with two keys and two
    distinct products; keying by name would reject the second as a duplicate
    and — worse — let one product satisfy both, since a set of names
    collapses the repetition and the totality check would pass with an
    effect left unchecked.

    This is the region-level half of the gate: an occurrence with no product
    is not admitted, which keeps "admissible" from silently meaning
    "unchecked".
    """
    by_occurrence: dict[str, RecordedProduct] = {}
    for item in products:
        if item.occurrence_id in by_occurrence:
            raise RecordedProductError(
                f"{item.occurrence_id}: more than one recorded product for "
                f"one occurrence"
            )
        by_occurrence[item.occurrence_id] = item
    requested = list(effectful_occurrences)
    duplicates = sorted(
        {name for name in requested if requested.count(name) > 1}
    )
    if duplicates:
        raise RecordedProductError(
            f"occurrence ids {duplicates} are repeated; each effectful "
            f"occurrence needs a distinct id or totality cannot be checked"
        )
    missing = sorted(set(requested) - set(by_occurrence))
    if missing:
        raise RecordedProductError(
            f"effectful occurrences {missing} entered a differentiated region "
            f"without a recorded product; they remain fail-closed"
        )
    extra = sorted(set(by_occurrence) - set(requested))
    if extra:
        raise RecordedProductError(
            f"recorded products {extra} name occurrences that are not on the "
            f"region's effectful path"
        )
    return by_occurrence


def region_digest(products: Sequence[RecordedProduct]) -> str:
    """One content address for a region's whole recorded frame."""
    return _digest(
        [item.canonical_payload() for item in sorted(products, key=lambda p: p.op)]
    )


__all__ = [
    "ADMITTED_EFFECT_CLASSES",
    "STOCHASTIC_REFUSALS",
    "RECORDED_PRODUCT_SCHEMA",
    "RecordedProduct",
    "RecordedProductError",
    "collective_product_for_sequence",
    "collective_sequence_digest",
    "content_digest",
    "mutation_product_for_buffer",
    "region_digest",
    "stochastic_product_for_call",
    "verify_collective_replay",
    "verify_confinement",
    "verify_recorded_state",
    "verify_region_products",
]


# ── W4-EFFECTS-1 slice E2: classifying a stochastic call form ───────────────
#
# "Keyed RNG is admissible" is a claim about a CALL FORM, not about an op
# name. Measured on `tessera.dropout`, whose three forms differ:
#
#   dropout(x, p, seed=N)          replay bit-identical      -> admissible
#   dropout(x, p)                  ambient OS entropy        -> closed
#   dropout(x, p, rng=<generator>) generator ADVANCES per
#                                  call; its position is not
#                                  in the product            -> closed
#
# The third is the one worth naming: it looks keyed, and is not. Recording
# the generator object identity would not help — replay needs its position,
# which the object does not expose and the product does not carry.

#: Reasons a stochastic call form cannot be admitted, as stable strings.
STOCHASTIC_REFUSALS = {
    "ambient": (
        "draws from ambient entropy; no recorded value determines the result"
    ),
    "stateful_generator": (
        "draws from a caller-owned generator whose position advances per call; "
        "the product would name the generator but not the state replay needs"
    ),
}


def stochastic_product_for_call(
    *,
    op: str,
    occurrence_id: str,
    shape: Sequence[int],
    dtype: str,
    seed: int | None = None,
    key: Mapping[str, Any] | None = None,
    generator: Any = None,
) -> RecordedProduct:
    """Build the `keyed_rng` product for an admissible draw, or fail closed.

    Exactly one source of randomness may be identified, and it must be one
    whose replay is a function of the recorded value: an S4 ``key`` (the
    counter-based generator, pure in its key) or a ``seed`` that is used to
    construct a fresh generator per call. A caller-owned generator and
    ambient entropy are refused by name.
    """
    if key is not None and seed is not None:
        raise RecordedProductError(
            f"{op}: a draw may identify either a key or a seed, not both; "
            f"two sources make the recorded product ambiguous"
        )
    if generator is not None:
        raise RecordedProductError(
            f"AUTODIFF_STOCHASTIC_UNKEYED: {op} "
            f"{STOCHASTIC_REFUSALS['stateful_generator']}"
        )
    if key is None and seed is None:
        raise RecordedProductError(
            f"AUTODIFF_STOCHASTIC_UNKEYED: {op} "
            f"{STOCHASTIC_REFUSALS['ambient']}"
        )
    identity: dict[str, Any] = {"shape": list(shape), "dtype": dtype}
    if key is not None:
        identity["key"] = dict(key)
    else:
        assert seed is not None  # the ambient case returned above
        identity["key"] = {"seed": int(seed)}
    return RecordedProduct(
        op=op,
        occurrence_id=occurrence_id,
        effect_class="keyed_rng",
        product=identity,
    )


# ── W4-EFFECTS-1 slice E3: recorded state ───────────────────────────────────
#
# A mutation product binds the buffer's IDENTITY (lineage id + version) and
# its VALUE (a content digest). Identity alone is not enough:
# `state_buffer_lineage` hashes name/role/shape/dtype/version/access/parents
# and not contents, so two buffers with identical metadata and different
# bytes share a lineage id. A replay binding "version N" could then bind
# different bytes and produce a different gradient — silently, which is the
# failure the whole gate exists to prevent.


def content_digest(buffer: Any) -> str:
    """Content address of a buffer's VALUE.

    Covers dtype and shape as well as bytes: the same bit pattern read as
    f32 and as int32 is two different values, and a reshape is a different
    buffer even when the bytes are identical.
    """
    import numpy as np

    array = np.ascontiguousarray(buffer)
    header = f"{array.dtype.str}|{array.shape}".encode()
    hasher = hashlib.sha256()
    hasher.update(header)
    hasher.update(array.tobytes())
    return hasher.hexdigest()


def mutation_product_for_buffer(
    *,
    op: str,
    occurrence_id: str,
    lineage_id: str,
    version: int,
    buffer: Any,
    write_set: Sequence[str],
) -> RecordedProduct:
    """Record a mutation: what was written, which version, and what it held."""
    return RecordedProduct(
        op=op,
        occurrence_id=occurrence_id,
        effect_class="recorded_mutation",
        product={
            "lineage_id": lineage_id,
            "version": int(version),
            "content_digest": content_digest(buffer),
        },
        write_set=tuple(write_set),
    )


def verify_recorded_state(recorded: RecordedProduct, buffer: Any) -> None:
    """Check (R) for a mutation: the bytes at replay are the bytes recorded.

    Rejects the case metadata identity cannot see — an unchanged lineage id
    and version over CHANGED bytes.
    """
    if recorded.effect_class != "recorded_mutation":
        raise RecordedProductError(
            f"{recorded.op}: verify_recorded_state applies to "
            f"recorded_mutation, not {recorded.effect_class}"
        )
    actual = content_digest(buffer)
    expected = str(recorded.product["content_digest"])
    if actual != expected:
        raise RecordedProductError(
            f"{recorded.op}: recorded state for lineage "
            f"{recorded.product['lineage_id']} version "
            f"{recorded.product['version']} hashes to {actual[:12]}… but the "
            f"product recorded {expected[:12]}…; the identity matched while "
            f"the VALUE changed, so replay would not reproduce the recorded "
            f"execution (reproducibility)"
        )


# ── W4-EFFECTS-1 slice E4: ordered collectives ──────────────────────────────
#
# An ordered collective's requirement is that every rank issues collectives in
# the same relative order. That is necessary and NOT sufficient for (R):
# floating-point addition is not associative, so the reduction TREE is part of
# the value. Measured: 1024 f32 values, identical inputs and identical issue
# order, reduced sequentially / by pairwise tree / by ring give three
# different bit patterns, and the ring result changes again with rank count.
# `LANGUAGE_AND_IR_SPEC` section 11 says the same thing normatively —
# "Deterministic profiles require fixed collective ordering and reduction
# trees" — so the product binds both.
#
# Scope boundary, deliberately narrow: a recorded sequence proves ORDER.
# Bit-identity of a collective RESULT additionally requires native
# deterministic evidence on real transport; a deterministic mock mesh cannot
# establish it, and this module does not pretend otherwise.


def collective_sequence_digest(sequence: Sequence[str]) -> str:
    """Content address of an ordered collective sequence.

    Order-SENSITIVE by construction: a permutation is a different sequence,
    which is the whole point.
    """
    return _digest({"sequence": [str(item) for item in sequence]})


def collective_product_for_sequence(
    *,
    op: str,
    occurrence_id: str,
    communicator: str,
    sequence: Sequence[str],
    reduction_algorithm: str,
    topology: Mapping[str, Any],
    write_set: Sequence[str],
) -> RecordedProduct:
    """Record a collective: who, in what order, and under which tree."""
    if not sequence:
        raise RecordedProductError(
            f"{op}: an ordered-collective product needs a non-empty sequence; "
            f"an empty one records no order at all"
        )
    return RecordedProduct(
        op=op,
        occurrence_id=occurrence_id,
        effect_class="ordered_collective",
        product={
            "communicator": communicator,
            "sequence_digest": collective_sequence_digest(sequence),
            "reduction_algorithm": reduction_algorithm,
            "topology": dict(topology),
        },
        write_set=tuple(write_set),
    )


def verify_collective_replay(
    recorded: RecordedProduct,
    observed_sequence: Sequence[str],
    *,
    reduction_algorithm: str,
    topology: Mapping[str, Any],
) -> None:
    """Check (R) for a collective: same order, same tree, same topology.

    A changed tree is rejected even when the order and inputs match — that is
    the case order-only checking cannot see, and the one that silently moves
    the result's bits.
    """
    if recorded.effect_class != "ordered_collective":
        raise RecordedProductError(
            f"{recorded.op}: verify_collective_replay applies to "
            f"ordered_collective, not {recorded.effect_class}"
        )
    observed = collective_sequence_digest(observed_sequence)
    if observed != str(recorded.product["sequence_digest"]):
        raise RecordedProductError(
            f"{recorded.op}: replay issued a different collective sequence "
            f"({observed[:12]}… vs recorded "
            f"{str(recorded.product['sequence_digest'])[:12]}…); ranks would "
            f"not agree on the order (reproducibility)"
        )
    if reduction_algorithm != str(recorded.product["reduction_algorithm"]):
        raise RecordedProductError(
            f"{recorded.op}: replay used reduction algorithm "
            f"{reduction_algorithm!r} but the product recorded "
            f"{recorded.product['reduction_algorithm']!r}. Floating-point "
            f"addition is not associative, so a different tree is a different "
            f"value even under an identical order (reproducibility)"
        )
    if _thaw(dict(topology)) != _thaw(recorded.product["topology"]):
        raise RecordedProductError(
            f"{recorded.op}: replay topology {dict(topology)!r} differs from "
            f"the recorded {_thaw(recorded.product['topology'])!r}; the "
            f"topology selects the tree"
        )
