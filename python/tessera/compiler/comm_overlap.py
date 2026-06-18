"""Tile-granular compute/communication overlap + SC-HRF scope/ordering (C3).

AMD's Iris pattern (and the SC-HRF — *Sequential Consistency for Heterogeneous
Race-Free* — memory model behind it) communicates a tile *the instant it is
produced* rather than after a whole-tensor barrier.  A producer workgroup
finishes a tile, **releases** it with an atomic signal, and a consumer workgroup
**acquires** that signal (an acquire-spin) before scattering / consuming the
tile.  The release/acquire pairing across the right *memory scope* is what makes
the handoff race-free, so overlap is correct without a global sync.

This module ports those ideas as Tessera contract metadata — pure data, no
kernel import.  It is a **leaf module** (stdlib only — ``enum`` + ``dataclasses``)
so the audit registry (``primitive_coverage``), ``op_catalog``, the backend
manifest, and the runtime can all import it without a cycle.

Four pieces:

* :class:`MemoryScope` — the SC-HRF scope hierarchy (wavefront ⊂ workgroup ⊂
  agent ⊂ system).  A signal's scope must be at least as broad as the widest
  set of agents that race on the tile; :meth:`MemoryScope.covers` answers "is
  this scope at least as broad as that one".

* :class:`MemoryOrdering` — relaxed / acquire / release / acq_rel, the SC-HRF
  orderings a signal op carries.

* :class:`SignalOp` — one producer-side or consumer-side synchronization op
  (atomic_cas / atomic_add / store / load / fence) with a scope + ordering.  A
  producer signal must *release*; a consumer wait must *acquire*.

* :class:`OverlapPlan` — a tile-granular overlap strategy: which overlap kind
  (sequential-fused / workgroup-specialized / unfused producer-consumer), how
  many producer vs consumer workgroups, and the matched release-signal /
  acquire-wait pair that hands tiles across.

Everything flattens to plain JSON-style dicts via ``as_metadata_dict`` (enums
serialize to their string values) so the contract rides on
``PrimitiveCoverage.metadata`` and (later) Schedule/Tile/Target IR attrs.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MemoryScope",
    "MemoryOrdering",
    "SignalOp",
    "OverlapKind",
    "OverlapPlan",
    "SIGNAL_KINDS",
    "plan_overlap",
]


# ── SC-HRF memory scopes (nested) ────────────────────────────────────────────
class MemoryScope(enum.Enum):
    """SC-HRF synchronization scope — the set of agents an atomic synchronizes.

    The four scopes are *nested* in increasing breadth — a release/acquire pair
    at a broader scope synchronizes a strict superset of the agents a narrower
    pair does:

    * ``WAVEFRONT`` — lanes of a single wavefront / warp.
    * ``WORKGROUP`` — work-items of one workgroup (CTA / threadgroup).
    * ``AGENT``     — all workgroups on one device (AMD "agent" / device scope).
    * ``SYSTEM``    — every agent in the system (multi-GPU + host).

    Tile-granular cross-workgroup handoff (the Iris pattern) needs at least
    ``AGENT`` scope, because producer and consumer live in different workgroups
    on the same device.
    """

    WAVEFRONT = "wavefront"
    WORKGROUP = "workgroup"
    AGENT = "agent"
    SYSTEM = "system"

    @property
    def _rank(self) -> int:
        return _SCOPE_RANK[self]

    def covers(self, other: "MemoryScope") -> bool:
        """True iff ``self`` is at least as broad as ``other``.

        ``SYSTEM.covers(AGENT)`` is ``True``; ``WORKGROUP.covers(AGENT)`` is
        ``False``.  A scope always covers itself.
        """
        if not isinstance(other, MemoryScope):
            raise ValueError(
                f"covers expects a MemoryScope; got {type(other).__name__}")
        return self._rank >= other._rank


# Nesting order (narrow → broad).  ``covers`` compares these ranks.
_SCOPE_RANK: dict[MemoryScope, int] = {
    MemoryScope.WAVEFRONT: 0,
    MemoryScope.WORKGROUP: 1,
    MemoryScope.AGENT: 2,
    MemoryScope.SYSTEM: 3,
}


# ── SC-HRF orderings ─────────────────────────────────────────────────────────
class MemoryOrdering(enum.Enum):
    """SC-HRF memory ordering carried by a synchronization op.

    * ``RELAXED`` — atomicity only, no ordering (a plain counter bump).
    * ``ACQUIRE`` — no later access may be reordered before this load.
    * ``RELEASE`` — no earlier access may be reordered after this store.
    * ``ACQ_REL`` — both (an atomic read-modify-write that does both halves).
    """

    RELAXED = "relaxed"
    ACQUIRE = "acquire"
    RELEASE = "release"
    ACQ_REL = "acq_rel"

    @property
    def has_acquire(self) -> bool:
        return self in (MemoryOrdering.ACQUIRE, MemoryOrdering.ACQ_REL)

    @property
    def has_release(self) -> bool:
        return self in (MemoryOrdering.RELEASE, MemoryOrdering.ACQ_REL)


# Signal op kinds — the atomic / memory primitive the signal lowers to.
#   atomic_cas — compare-and-swap (the canonical flag set / acquire-spin probe)
#   atomic_add — fetch-and-add (a producer-count / arrival counter)
#   store      — a plain release store of a flag
#   load       — a plain acquire load of a flag
#   fence      — a standalone memory fence (no address)
SIGNAL_KINDS = ("atomic_cas", "atomic_add", "store", "load", "fence")


@dataclass(frozen=True)
class SignalOp:
    """One producer-side or consumer-side synchronization op.

    ``role`` is ``"producer"`` (the signal that *releases* a finished tile) or
    ``"consumer"`` (the wait that *acquires* it before consuming).  A producer
    signal must carry a release (``RELEASE`` / ``ACQ_REL``) ordering; a consumer
    wait must carry an acquire (``ACQUIRE`` / ``ACQ_REL``) ordering — anything
    else is a data race and is rejected at construction.

    A ``fence`` op carries no data and is allowed in either role as long as its
    ordering matches the role.
    """

    role: str
    scope: MemoryScope
    ordering: MemoryOrdering
    kind: str = "atomic_cas"

    def __post_init__(self) -> None:
        if self.role not in ("producer", "consumer"):
            raise ValueError(
                f"role must be 'producer' or 'consumer'; got {self.role!r}")
        if not isinstance(self.scope, MemoryScope):
            raise ValueError(
                f"scope must be a MemoryScope; got {type(self.scope).__name__}")
        if not isinstance(self.ordering, MemoryOrdering):
            raise ValueError(
                f"ordering must be a MemoryOrdering; got "
                f"{type(self.ordering).__name__}")
        if self.kind not in SIGNAL_KINDS:
            raise ValueError(
                f"signal kind must be one of {SIGNAL_KINDS}; got {self.kind!r}")
        if self.role == "producer" and not self.is_valid_producer():
            raise ValueError(
                f"a producer signal must use RELEASE or ACQ_REL ordering "
                f"(release semantics publish the finished tile); got "
                f"{self.ordering.value!r}")
        if self.role == "consumer" and not self.is_valid_consumer():
            raise ValueError(
                f"a consumer wait must use ACQUIRE or ACQ_REL ordering "
                f"(acquire semantics observe the published tile); got "
                f"{self.ordering.value!r}")

    def is_valid_producer(self) -> bool:
        """True iff this op may publish a tile (carries release semantics)."""
        return self.ordering.has_release

    def is_valid_consumer(self) -> bool:
        """True iff this op may observe a published tile (acquire semantics)."""
        return self.ordering.has_acquire

    def as_metadata_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "scope": self.scope.value,
            "ordering": self.ordering.value,
            "kind": self.kind,
        }


# ── Overlap strategies ───────────────────────────────────────────────────────
class OverlapKind(enum.Enum):
    """Tile-granular compute/communication overlap strategies (Iris families).

    * ``SEQUENTIAL_FUSED`` — store each output tile inside the GEMM loop, in the
      *same* workgroups that produced it.  No cross-workgroup handoff, so there
      is no separate consumer workgroup pool (``consumer_workgroups == 0``).

    * ``WORKGROUP_SPECIALIZED`` — partition the workgroups: a producer pool
      computes + releases tiles via release-atomics; a distinct consumer pool
      acquire-spins then scatters them.  Needs a matched release/acquire pair.

    * ``UNFUSED_PRODUCER_CONSUMER`` — producer and consumer run as separate
      kernels / streams with the CUs partitioned between them.
    """

    SEQUENTIAL_FUSED = "sequential_fused"
    WORKGROUP_SPECIALIZED = "workgroup_specialized"
    UNFUSED_PRODUCER_CONSUMER = "unfused_producer_consumer"


@dataclass(frozen=True)
class OverlapPlan:
    """A tile-granular overlap plan: kind, workgroup split, and signal pair.

    ``signal`` is the producer-side release; ``wait`` is the consumer-side
    acquire.  ``cu_partition`` optionally records ``(producer_cus,
    consumer_cus)`` for the strategies that physically partition the CUs.

    Consistency rules (enforced at construction):

    * ``signal`` must be a valid producer (release) and ``wait`` a valid
      consumer (acquire).
    * ``WORKGROUP_SPECIALIZED`` requires both ``producer_workgroups`` and
      ``consumer_workgroups`` to be positive (it physically splits the pools)
      and a release-signal + acquire-wait pair across matching scopes.
    * ``SEQUENTIAL_FUSED`` runs the store in the producing workgroups, so it has
      no separate consumer pool — ``consumer_workgroups`` must be ``0``.
    """

    kind: OverlapKind
    producer_workgroups: int
    consumer_workgroups: int
    signal: SignalOp
    wait: SignalOp
    cu_partition: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, OverlapKind):
            raise ValueError(
                f"kind must be an OverlapKind; got {type(self.kind).__name__}")
        if self.producer_workgroups < 0 or self.consumer_workgroups < 0:
            raise ValueError(
                f"workgroup counts must be non-negative; got "
                f"producer={self.producer_workgroups}, "
                f"consumer={self.consumer_workgroups}")
        if not isinstance(self.signal, SignalOp) or self.signal.role != "producer":
            raise ValueError("signal must be a producer SignalOp")
        if not isinstance(self.wait, SignalOp) or self.wait.role != "consumer":
            raise ValueError("wait must be a consumer SignalOp")
        if not self.signal.is_valid_producer():
            raise ValueError("signal must carry release (or acq_rel) semantics")
        if not self.wait.is_valid_consumer():
            raise ValueError("wait must carry acquire (or acq_rel) semantics")

        if self.kind is OverlapKind.SEQUENTIAL_FUSED:
            if self.consumer_workgroups != 0:
                raise ValueError(
                    "SEQUENTIAL_FUSED stores tiles in the producing workgroups; "
                    f"it has no separate consumer pool, so consumer_workgroups "
                    f"must be 0; got {self.consumer_workgroups}")
        if self.kind is OverlapKind.WORKGROUP_SPECIALIZED:
            if self.producer_workgroups <= 0 or self.consumer_workgroups <= 0:
                raise ValueError(
                    "WORKGROUP_SPECIALIZED splits the workgroups into producer "
                    "and consumer pools; both producer_workgroups and "
                    "consumer_workgroups must be positive; got "
                    f"producer={self.producer_workgroups}, "
                    f"consumer={self.consumer_workgroups}")
            if not self.signal.scope.covers(self.wait.scope):
                raise ValueError(
                    "WORKGROUP_SPECIALIZED requires the producer release scope to "
                    "cover the consumer acquire scope (cross-workgroup handoff "
                    f"needs ≥ agent scope); got signal scope "
                    f"{self.signal.scope.value!r} not covering "
                    f"{self.wait.scope.value!r}")

        if self.cu_partition is not None:
            if (len(self.cu_partition) != 2
                    or self.cu_partition[0] < 0 or self.cu_partition[1] < 0):
                raise ValueError(
                    "cu_partition must be a (producer_cus, consumer_cus) pair of "
                    f"non-negative ints; got {self.cu_partition}")

    def as_metadata_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "producer_workgroups": self.producer_workgroups,
            "consumer_workgroups": self.consumer_workgroups,
            "signal": self.signal.as_metadata_dict(),
            "wait": self.wait.as_metadata_dict(),
            "cu_partition": list(self.cu_partition)
            if self.cu_partition is not None else None,
        }


# ── Factory ──────────────────────────────────────────────────────────────────
def plan_overlap(kind: OverlapKind, *, producer_wgs: int, consumer_wgs: int,
                 scope: MemoryScope = MemoryScope.AGENT) -> OverlapPlan:
    """Build an :class:`OverlapPlan` with a matched release/acquire signal pair.

    The producer signals via an ``atomic_cas`` release at ``scope`` (publish the
    finished tile's ready-flag); the consumer waits via an ``atomic_cas``
    acquire at the same scope (the acquire-spin that observes the flag).  Using
    one scope for both halves keeps the release→acquire edge well-formed.

    Pattern notes:

    * ``SEQUENTIAL_FUSED`` — the store rides inside the GEMM loop in the
      producing workgroups; pass ``consumer_wgs=0`` (the plan still carries a
      well-formed signal pair describing the in-loop release/acquire of the
      shared tile, but there is no separate consumer pool).
    * ``WORKGROUP_SPECIALIZED`` — producer workgroups release tiles; consumer
      workgroups acquire-spin then scatter.  ``scope`` should be ``AGENT`` (or
      broader) so the cross-workgroup handoff is race-free.
    * ``UNFUSED_PRODUCER_CONSUMER`` — separate streams with CU partitioning; the
      release/acquire still guards the shared buffer.
    """
    signal = SignalOp(role="producer", scope=scope,
                      ordering=MemoryOrdering.RELEASE, kind="atomic_cas")
    wait = SignalOp(role="consumer", scope=scope,
                    ordering=MemoryOrdering.ACQUIRE, kind="atomic_cas")
    return OverlapPlan(
        kind=kind,
        producer_workgroups=producer_wgs,
        consumer_workgroups=consumer_wgs,
        signal=signal,
        wait=wait,
    )
