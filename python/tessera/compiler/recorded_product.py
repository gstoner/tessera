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
    "RECORDED_PRODUCT_SCHEMA",
    "RecordedProduct",
    "RecordedProductError",
    "region_digest",
    "verify_confinement",
    "verify_region_products",
]
