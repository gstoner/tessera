"""Table-driven tuned-config dispatch DB + correctness-gated autotuning (AITER-style).

AMD's AITER selects GEMM / MoE kernels from CSV tuned-config tables keyed on the
problem *signature* ``(gfx, cu_num, M, N, K, dtype)`` — one tuned row per signature,
de-duped on the lowest measured latency so dispatch is deterministic.  Its pipeline
is **untune → tune → CSV**: an untuned worklist enumerates the problem shapes a model
hits, an offline tuner sweeps kernel candidates per shape, and the winning configs
are baked into the CSV table the runtime then dispatches against in O(1).

This module ports that design as pure-data Tessera compiler metadata.  It is a leaf
module (stdlib only — ``csv`` / ``dataclasses`` / ``pathlib`` / ``io``; nothing from
tessera) so the runtime, the audit registry, and ``op_catalog`` can all import it
without a cycle.

CRITICAL DESIGN LESSON (from AITER): the dispatch table is keyed on the problem
*signature*, **never** on the opaque ``solidx`` (solution index).  Solution indices
are non-portable across arch / library version — they are payload, not key.  Keying
on them would silently mis-dispatch after a library bump.  Here the key is the
:class:`ProblemSignature` and the ``solidx`` rides only inside :class:`TunedConfig`.

Two-tier override (hipBLASLt model): a runtime-loadable override table wins over the
base table per matching signature without recompiling — :meth:`TunedDispatchTable.with_override`.

Correctness-gated autotuning (the magellan / alphaevolve "perf gated behind
correctness" invariant): :func:`tune` measures latency **only** for candidates that
first pass a ``correctness_fn`` gate, and refuses to ever return a fast-but-wrong
winner.  If no candidate passes correctness it raises rather than dispatch a wrong
kernel.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

# AITER's library backends.  ``libtype`` is payload (which kernel family produced the
# winning config), never part of the dispatch key.
LIBTYPES = ("hipblaslt", "asm", "triton", "flydsl", "ck", "cktile")

# The flat CSV column order (AITER's tuned-config schema).
CSV_COLUMNS = (
    "gfx",
    "cu_num",
    "M",
    "N",
    "K",
    "dtype",
    "libtype",
    "solidx",
    "splitK",
    "kernelName",
    "latency_us",
)

# Worklist CSV is just the signature columns.
WORKLIST_COLUMNS = ("gfx", "cu_num", "M", "N", "K", "dtype")


SignatureKey = tuple[str, int, int, int, int, str]


@dataclass(frozen=True)
class ProblemSignature:
    """The dispatch KEY: the problem signature a tuned config is selected for.

    ``(gfx, cu_num, M, N, K, dtype)`` — the architecture + compute-unit count + the
    GEMM problem dims + the dtype.  Frozen, so it is hashable and usable as a dict
    key.  This — and **never** the opaque ``solidx`` — is what the table keys on.
    """

    gfx: str
    cu_num: int
    m: int
    n: int
    k: int
    dtype: str

    def __post_init__(self) -> None:
        if not self.gfx:
            raise ValueError("ProblemSignature.gfx must be a non-empty string")
        if not self.dtype:
            raise ValueError("ProblemSignature.dtype must be a non-empty string")
        if self.cu_num <= 0:
            raise ValueError(
                f"ProblemSignature.cu_num must be positive; got {self.cu_num}")
        for name, val in (("m", self.m), ("n", self.n), ("k", self.k)):
            if val <= 0:
                raise ValueError(
                    f"ProblemSignature.{name} must be positive; got {val}")

    def as_key(self) -> SignatureKey:
        """The hashable tuple key used for table lookup."""
        return (self.gfx, self.cu_num, self.m, self.n, self.k, self.dtype)

    def as_metadata_dict(self) -> dict[str, Any]:
        return {
            "gfx": self.gfx,
            "cu_num": self.cu_num,
            "M": self.m,
            "N": self.n,
            "K": self.k,
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class TunedConfig:
    """A tuned config: a :class:`ProblemSignature` key + the kernel payload.

    The payload (``libtype`` / ``solidx`` / ``split_k`` / ``kernel_name`` /
    ``latency_us``) describes the winning kernel for that signature.  ``solidx`` is
    payload only — it is *never* used as a dispatch key (non-portable across arch /
    library version).
    """

    signature: ProblemSignature
    libtype: str
    solidx: int
    split_k: int
    kernel_name: str
    latency_us: float

    def __post_init__(self) -> None:
        if self.libtype not in LIBTYPES:
            raise ValueError(
                f"TunedConfig.libtype must be one of {LIBTYPES}; got {self.libtype!r}")
        if self.latency_us < 0:
            raise ValueError(
                f"TunedConfig.latency_us must be >= 0; got {self.latency_us}")
        if self.split_k < 1:
            raise ValueError(
                f"TunedConfig.split_k must be >= 1; got {self.split_k}")
        if not self.kernel_name:
            raise ValueError("TunedConfig.kernel_name must be a non-empty string")

    def to_row(self) -> dict[str, Any]:
        """The flat 11-column CSV row (signature columns + payload columns)."""
        return {
            "gfx": self.signature.gfx,
            "cu_num": self.signature.cu_num,
            "M": self.signature.m,
            "N": self.signature.n,
            "K": self.signature.k,
            "dtype": self.signature.dtype,
            "libtype": self.libtype,
            "solidx": self.solidx,
            "splitK": self.split_k,
            "kernelName": self.kernel_name,
            "latency_us": self.latency_us,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "TunedConfig":
        """Parse a flat 11-column CSV row back into a :class:`TunedConfig`."""
        try:
            sig = ProblemSignature(
                gfx=str(row["gfx"]),
                cu_num=int(row["cu_num"]),
                m=int(row["M"]),
                n=int(row["N"]),
                k=int(row["K"]),
                dtype=str(row["dtype"]),
            )
            return cls(
                signature=sig,
                libtype=str(row["libtype"]),
                solidx=int(row["solidx"]),
                split_k=int(row["splitK"]),
                kernel_name=str(row["kernelName"]),
                latency_us=float(row["latency_us"]),
            )
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"TunedConfig.from_row: missing CSV column {exc}; "
                f"expected columns {CSV_COLUMNS}") from exc


class TunedDispatchTable:
    """Signature → tuned-config table with deterministic de-dup dispatch.

    Holds one :class:`TunedConfig` per :class:`ProblemSignature`, de-duped on the
    **lowest** ``latency_us`` so dispatch is deterministic regardless of insertion
    order.  Lookup keys on the signature (never on ``solidx``).
    """

    def __init__(self, configs: Iterable[TunedConfig] | None = None) -> None:
        self._by_key: dict[SignatureKey, TunedConfig] = {}
        for cfg in configs or ():
            self.add(cfg)

    def add(self, cfg: TunedConfig) -> None:
        """Insert a config, keeping the lowest-latency entry for its signature.

        De-dup is deterministic: a higher-latency config for an already-present
        signature is dropped; a lower-latency one replaces the incumbent.
        """
        key = cfg.signature.as_key()
        incumbent = self._by_key.get(key)
        if incumbent is None or cfg.latency_us < incumbent.latency_us:
            self._by_key[key] = cfg

    def lookup(self, sig: ProblemSignature) -> TunedConfig | None:
        """The tuned config for ``sig``, or ``None`` if untuned.  Keys on the
        signature — never on the opaque ``solidx``."""
        return self._by_key.get(sig.as_key())

    def lookup_or_default(
        self, sig: ProblemSignature, default: TunedConfig
    ) -> TunedConfig:
        """The tuned config for ``sig`` if present, else ``default`` (the fallback
        kernel the runtime uses for an untuned shape)."""
        found = self._by_key.get(sig.as_key())
        return found if found is not None else default

    def signatures(self) -> list[ProblemSignature]:
        """Every tuned signature in the table."""
        return [cfg.signature for cfg in self._by_key.values()]

    def configs(self) -> list[TunedConfig]:
        """Every tuned config in the table."""
        return list(self._by_key.values())

    def with_override(
        self, override_table: "TunedDispatchTable"
    ) -> "TunedDispatchTable":
        """A new merged table where ``override_table`` rows WIN over this base table
        per matching signature (the hipBLASLt two-tier override model).

        Signatures present only in the base survive; signatures present in the
        override replace the base unconditionally (the override is authoritative —
        it is *not* re-de-duped on latency against the base).
        """
        merged = TunedDispatchTable()
        merged._by_key = dict(self._by_key)
        for key, cfg in override_table._by_key.items():
            merged._by_key[key] = cfg
        return merged

    def to_csv(self, path: str | Path) -> None:
        """Write the table to a tuned-config CSV (deterministic row order)."""
        rows = sorted(
            (cfg.to_row() for cfg in self._by_key.values()),
            key=lambda r: (r["gfx"], r["cu_num"], r["M"], r["N"], r["K"], r["dtype"]),
        )
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)

    def to_csv_string(self) -> str:
        """The tuned-config CSV as a string (for in-memory round-trips)."""
        buf = io.StringIO()
        rows = sorted(
            (cfg.to_row() for cfg in self._by_key.values()),
            key=lambda r: (r["gfx"], r["cu_num"], r["M"], r["N"], r["K"], r["dtype"]),
        )
        writer = csv.DictWriter(buf, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
        return buf.getvalue()

    @classmethod
    def load_csv(cls, path: str | Path) -> "TunedDispatchTable":
        """Load a tuned-config CSV, de-duping on load (lowest latency per signature
        wins — so a hand-edited CSV with duplicate signatures still dispatches
        deterministically)."""
        table = cls()
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                table.add(TunedConfig.from_row(row))
        return table

    @classmethod
    def load_csv_string(cls, text: str) -> "TunedDispatchTable":
        """Load a tuned-config CSV from a string (de-duping on load)."""
        table = cls()
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            table.add(TunedConfig.from_row(row))
        return table

    def __len__(self) -> int:
        return len(self._by_key)

    def __contains__(self, sig: object) -> bool:
        if not isinstance(sig, ProblemSignature):
            return False
        return sig.as_key() in self._by_key

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"TunedDispatchTable({len(self._by_key)} signatures)"


@dataclass(frozen=True)
class TuneResult:
    """The outcome of one correctness-gated autotune sweep.

    ``config`` is the winning tuned config (lowest latency among the candidates that
    passed correctness).  ``rejected`` is how many candidates failed the correctness
    gate (auditability — the perf-gated-behind-correctness invariant is observable).
    """

    config: TunedConfig
    candidates_total: int
    candidates_passed: int
    rejected: int


def tune(
    signature: ProblemSignature,
    candidates: list[Any],
    *,
    correctness_fn: Callable[[Any], bool],
    latency_fn: Callable[[Any], float],
    libtype_fn: Callable[[Any], str] | None = None,
    solidx_fn: Callable[[Any], int] | None = None,
    split_k_fn: Callable[[Any], int] | None = None,
    kernel_name_fn: Callable[[Any], str] | None = None,
) -> TuneResult:
    """Correctness-gated autotune: pick the lowest-latency candidate that is CORRECT.

    For each candidate the **correctness gate runs first** (``correctness_fn`` — the
    ``checkAllclose`` analog).  Only candidates that PASS are then measured via
    ``latency_fn`` (microseconds).  The winner is the passing candidate with the
    lowest measured latency, returned as a :class:`TunedConfig` keyed on
    ``signature``.

    This is the magellan / alphaevolve **perf-gated-behind-correctness** invariant:
    a fast-but-wrong candidate can never win because it never reaches the latency
    measurement.  If *no* candidate passes correctness, raises ``ValueError`` rather
    than dispatch a wrong kernel.

    The ``*_fn`` extractors map an arbitrary candidate object to the tuned-config
    payload fields; sensible defaults read common attribute / dict keys.
    """
    if not candidates:
        raise ValueError("tune: candidates must be a non-empty list")

    lib = libtype_fn or (lambda c: _candidate_field(c, "libtype", "hipblaslt"))
    sol = solidx_fn or (lambda c: int(_candidate_field(c, "solidx", 0)))
    spk = split_k_fn or (lambda c: int(_candidate_field(c, "split_k", 1)))
    name = kernel_name_fn or (lambda c: str(_candidate_field(c, "kernel_name", "")))

    best: tuple[float, Any] | None = None
    passed = 0
    rejected = 0
    for cand in candidates:
        if not correctness_fn(cand):
            # The gate: a wrong kernel is never measured, so it can never win.
            rejected += 1
            continue
        passed += 1
        lat = float(latency_fn(cand))
        if lat < 0:
            raise ValueError(
                f"tune: latency_fn returned a negative latency {lat} for a "
                f"candidate — latencies are microseconds and must be >= 0")
        if best is None or lat < best[0]:
            best = (lat, cand)

    if best is None:
        raise ValueError(
            f"tune: no candidate passed the correctness gate for signature "
            f"{signature.as_key()} ({rejected}/{len(candidates)} rejected). "
            f"Refusing to dispatch a fast-but-wrong kernel.")

    lat, winner = best
    config = TunedConfig(
        signature=signature,
        libtype=lib(winner),
        solidx=sol(winner),
        split_k=spk(winner),
        kernel_name=name(winner),
        latency_us=lat,
    )
    return TuneResult(
        config=config,
        candidates_total=len(candidates),
        candidates_passed=passed,
        rejected=rejected,
    )


def _candidate_field(cand: Any, key: str, default: Any) -> Any:
    """Read ``key`` from a candidate that is either a mapping or an object."""
    if isinstance(cand, dict):
        return cand.get(key, default)
    return getattr(cand, key, default)


@dataclass
class UntunedWorklist:
    """The auditable "what still needs tuning" artifact (AITER's untune stage).

    Holds the set of :class:`ProblemSignature` a model touches.  Diffing it against
    a :class:`TunedDispatchTable` yields the tuned-vs-pending split — the worklist is
    deliberately separate from the tuned-results table so "what shapes exist" and
    "what shapes are tuned" stay independently auditable.
    """

    signatures: list[ProblemSignature] = field(default_factory=list)

    def __post_init__(self) -> None:
        # De-dup while preserving first-seen order (deterministic worklist).
        seen: set[SignatureKey] = set()
        unique: list[ProblemSignature] = []
        for sig in self.signatures:
            key = sig.as_key()
            if key not in seen:
                seen.add(key)
                unique.append(sig)
        self.signatures = unique

    def add(self, sig: ProblemSignature) -> None:
        """Add a signature (idempotent — duplicates are ignored)."""
        if sig not in self.signatures:
            self.signatures.append(sig)

    def remove(self, sig: ProblemSignature) -> None:
        """Remove a signature.  Raises ``KeyError`` if it is not present."""
        key = sig.as_key()
        for i, existing in enumerate(self.signatures):
            if existing.as_key() == key:
                del self.signatures[i]
                return
        raise KeyError(f"UntunedWorklist.remove: signature {key} not in worklist")

    def pending_against(self, table: TunedDispatchTable) -> list[ProblemSignature]:
        """Signatures in the worklist that are NOT yet tuned in ``table``."""
        return [sig for sig in self.signatures if sig not in table]

    def tuned_against(self, table: TunedDispatchTable) -> list[ProblemSignature]:
        """Signatures in the worklist that ARE already tuned in ``table``."""
        return [sig for sig in self.signatures if sig in table]

    def to_csv(self, path: str | Path) -> None:
        """Write the worklist signatures to a CSV (deterministic row order)."""
        rows = sorted(
            (sig.as_metadata_dict() for sig in self.signatures),
            key=lambda r: (r["gfx"], r["cu_num"], r["M"], r["N"], r["K"], r["dtype"]),
        )
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(WORKLIST_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)

    @classmethod
    def load_csv(cls, path: str | Path) -> "UntunedWorklist":
        """Load a worklist CSV (de-duped via ``__post_init__``)."""
        sigs: list[ProblemSignature] = []
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                sigs.append(
                    ProblemSignature(
                        gfx=str(row["gfx"]),
                        cu_num=int(row["cu_num"]),
                        m=int(row["M"]),
                        n=int(row["N"]),
                        k=int(row["K"]),
                        dtype=str(row["dtype"]),
                    )
                )
        return cls(signatures=sigs)

    def __len__(self) -> int:
        return len(self.signatures)
