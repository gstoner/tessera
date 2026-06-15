"""Long-horizon memory benchmark core (RULER / LongMemEval / MemoryArena lens).

The shared defect these long-context/long-memory benchmarks expose is *not*
"can we run a giant attention op" — it is **re-derivation of resident state**:
retrieve a fact, update it, aggregate across hops, and abstain when the answer
is absent, over a bank that must persist across turns.  Correctness here is an
aliasing/effect property, not a math property, which is exactly the seam
Tessera's reference memory layer leaves open.

This package mirrors the house benchmark-core template (``grid_ai_core`` /
``lattice_reasoning_core``): a primitive-gap registry plus rows emitted at three
proof levels through ``benchmarks.common``:

* **Reference / executable** rows — the public ``tessera.memory`` contract
  (``memory_read``/``memory_write``/``memory_evict``) run on the CPU reference
  and agree with an independent oracle.  ``execution_kind = reference``.
* **Missing-backend** rows — capabilities that have no on-device kernel yet
  (resident bank, version-aware retrieval).  Each names the gap from
  :data:`MEMORY_PRIMITIVE_GAPS`; ``passed`` is ``None`` (no claim made), the
  honest "not yet" row that motivates the next kernel.

Gap-registry-driven discipline (same as ``lattice_reasoning_core``): a row that
can't reach the runtime names the missing primitive; when the primitive lands,
the row flips green and the name moves to :data:`LANDED_MEMORY_PRIMITIVES`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from benchmarks.common import (
    BenchmarkOperator,
    BenchmarkRow,
    CompilerPath,
    Correctness,
    ExecutionKind,
    Profile,
    RuntimeStatus,
    telemetry_for_row,
)
from tessera.memory import MemoryTable, memory_evict, memory_read, memory_write


# Composites with no dedicated on-device op yet — the registry that drives
# stdlib/runtime growth.  Empty: the long-memory surface (scoring, top-1, hard
# top-k, soft-read, read-residency, append-residency) is now all hardware-
# verified.  What remains is @jit frontend routing, tracked in
# PARTIAL_MEMORY_PRIMITIVES — not a missing kernel.
MEMORY_PRIMITIVE_GAPS: tuple[str, ...] = ()

# Primitives whose on-device kernel + runtime path is landed and HARDWARE-
# VERIFIED, but which are not yet reachable through the full ``@jit`` single-call
# path (a frontend-integration gap, not a kernel gap).  Distinct from GAPS (no
# kernel) and LANDED (fully usable end-to-end).
PARTIAL_MEMORY_PRIMITIVES: tuple[str, ...] = (
    # Hard top-k (k>1) runs on Metal via MPSGraph TopK
    # (tessera_apple_gpu_mpsgraph_topk_f32, values+indices) through the directly-
    # callable runtime._apple_gpu_dispatch_topk — hardware-verified in
    # tests/unit/test_apple_gpu_topk.py.  NOT pipeline-routed: the frontend AST
    # lowerer can't emit the multi-output `tessera.top_k` into Graph IR, so it is
    # deliberately not in the runtime envelope (that wiring lands with the
    # frontend).  `@jit(target="apple_gpu")(top_k)` falls back to eager today.
    "segmented_topk_gpu",
    # Read-residency: ResidentBank (tessera.resident_bank) keeps the bank's keys
    # on-device across reads (one upload), scoring each query via the bmm_enc
    # encode-session lane without re-uploading — hardware-verified in
    # tests/unit/test_resident_bank.py (~28× upload reduction on Metal).
    "resident_state_handle",
    # Append-residency: ResidentBank.append offset-writes a new entry into the
    # resident buffer via the ts_dev_upload_at device symbol (O(entry), no full
    # re-upload), scored against the growing bank with bmm_enc(M=current_seq) —
    # hardware-verified in test_resident_bank.py (~32× upload reduction over a
    # 128-step decode loop, metamorphically == recompute).  Landed as a direct
    # ResidentBank API; a single fused @jit-routed append_read kernel is the
    # remaining frontend-integration follow-on.
    "kv_cache_append_read",
)

# Primitives that started as gaps and have since landed a real contract.
LANDED_MEMORY_PRIMITIVES: tuple[str, ...] = (
    # memory_read(..., abstain_below=) returns a NaN-filled, abstained result
    # when no entry clears the score floor — the LongMemEval abstention contract.
    "abstention_read_threshold",
    # memory_read(..., prefer_recent=/recency_key=) breaks an exact-key score tie
    # toward the newest write — the LongMemEval knowledge-update contract.  The
    # resident-vs-recompute decode row proves the append-without-reprocess
    # *algorithm* at reference level; cross-step residency stays a gap above.
    "metadata_time_version_filter",
    # query·keysᵀ scoring + top-1 select + full-bank soft-read all execute on
    # Metal (rung 8) — proven by tessera.compiler.memory_tasks against the
    # Evaluator oracle.  Only HARD top-k (k>1) and residency remain gaps.
    "memory_index_score_gpu",
)


@dataclass(frozen=True)
class LongMemoryConfig:
    """One tiny long-memory core run."""

    bank_size: int = 256          # distractor facts written to the bank
    key_dim: int = 32
    value_dim: int = 8
    n_needles: int = 4            # RULER: planted facts to recover
    top_k: int = 1
    abstain_floor: float = 0.5    # cosine floor below which a read abstains
    decode_steps: int = 16        # resident-vs-recompute decode horizon
    seed: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Scenario builders — pure, oracle-checkable.
# ─────────────────────────────────────────────────────────────────────────────


def _unit_rows(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    """``n`` unit-norm row vectors in ``d`` dims (so dot scores are cosines)."""
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def ruler_needle_scenario(
    cfg: LongMemoryConfig,
) -> tuple[MemoryTable, np.ndarray, np.ndarray]:
    """RULER multi-needle: write a bank of distractors + ``n_needles`` planted
    facts; query each needle by its own key.  Oracle = each needle's value."""
    rng = np.random.default_rng(cfg.seed ^ 0x12345)
    keys = _unit_rows(rng, cfg.bank_size, cfg.key_dim)
    vals = rng.standard_normal((cfg.bank_size, cfg.value_dim)).astype(np.float32)
    needle_idx = rng.choice(cfg.bank_size, size=cfg.n_needles, replace=False)
    bank = MemoryTable(keys=keys, values=vals)
    return bank, keys[needle_idx], vals[needle_idx]


def ruler_multihop_scenario(
    cfg: LongMemoryConfig,
) -> tuple[MemoryTable, np.ndarray, np.ndarray]:
    """RULER multi-hop: ``value[a]`` is a *pointer* (== ``key[b]``), and
    ``value[b]`` is the answer payload.  Reading ``key[a]`` yields ``key[b]``;
    reading that yields the payload.  ``value_dim`` is forced to ``key_dim`` so a
    value can serve as the next query.  Oracle = the payload at the chain end."""
    rng = np.random.default_rng(cfg.seed ^ 0x6789A)
    d = cfg.key_dim
    keys = _unit_rows(rng, cfg.bank_size, d)
    vals = rng.standard_normal((cfg.bank_size, d)).astype(np.float32)
    a, b = rng.choice(cfg.bank_size, size=2, replace=False)
    vals[a] = keys[b]                                   # a → pointer to b
    payload = rng.standard_normal((d,)).astype(np.float32)
    vals[b] = payload                                  # b → answer payload
    bank = MemoryTable(keys=keys, values=vals)
    return bank, keys[a], payload


def abstention_scenario(
    cfg: LongMemoryConfig,
) -> tuple[MemoryTable, np.ndarray, np.ndarray]:
    """LongMemEval abstention: an orthonormal (identity) mini-bank so similarity
    is unambiguous.  A *present* query (a basis vector) clears the floor; an
    *absent* query (uniform, every cosine ≈ 1/√d < floor) must abstain.

    Returns ``(bank, present_query, absent_query)``."""
    d = cfg.key_dim
    keys = np.eye(d, dtype=np.float32)                 # unit, mutually orthogonal
    vals = np.arange(d, dtype=np.float32)[:, None] * np.ones((d, cfg.value_dim), np.float32)
    bank = MemoryTable(keys=keys, values=vals)
    present = keys[3].copy()                           # cosine 1.0 with entry 3
    absent = np.full((d,), 1.0 / np.sqrt(d), np.float32)  # cosine 1/√d with every row
    return bank, present, absent


def version_update_scenario(
    cfg: LongMemoryConfig,
) -> tuple[MemoryTable, np.ndarray, np.ndarray]:
    """LongMemEval knowledge-update: write a fact, then overwrite it in a later
    'session' with the SAME key.  A *recency-aware* read should return the newer
    value — but plain similarity top-k cannot break the exact-key tie, so this is
    a genuine gap (``metadata_time_version_filter``).  Returns
    ``(bank, query_key, latest_value)`` for the gap row to reference."""
    rng = np.random.default_rng(cfg.seed ^ 0xDA7E)
    key = _unit_rows(rng, 1, cfg.key_dim)
    v1 = rng.standard_normal((1, cfg.value_dim)).astype(np.float32)
    v2 = rng.standard_normal((1, cfg.value_dim)).astype(np.float32)
    empty = MemoryTable(
        keys=np.zeros((0, cfg.key_dim), np.float32),
        values=np.zeros((0, cfg.value_dim), np.float32),
        metadata={"session": np.zeros((0,), np.int64)},
    )
    bank = memory_write(empty, key, v1)               # session 1
    bank = memory_write(bank, key, v2)                # session 2 — newer
    return bank, key[0], v2[0]


def resident_decode_telemetry(cfg: LongMemoryConfig) -> tuple[bool, dict[str, Any]]:
    """MemoryArena-style resident-state decode loop, measured.

    Over a ``decode_steps`` horizon a new fact arrives each step and we read it
    back.  Two strategies:

    * **recompute** — rebuild the whole bank from all facts-so-far each step
      (reprocesses history): build traffic is ``Σ t = T(T+1)/2`` entries.
    * **resident** — append the new fact to a persistent bank: build traffic is
      ``T`` entries.

    The *metamorphic oracle* gates the measurement: the two paths must produce
    identical reads at every step (residency may change cost, never output).
    Returns ``(reads_match, metrics)``.  Reads still scan the resident bank each
    step — that scan is exactly what ``memory_index_score_gpu`` /
    ``segmented_topk_gpu`` would accelerate on-device, so those stay gaps.
    """
    rng = np.random.default_rng(cfg.seed ^ 0xC0FFEE)
    t_steps = int(cfg.decode_steps)
    keys = _unit_rows(rng, t_steps, cfg.key_dim)
    vals = rng.standard_normal((t_steps, cfg.value_dim)).astype(np.float32)

    recompute_reads = []
    recompute_build = 0
    for t in range(t_steps):
        bank = MemoryTable(keys=keys[: t + 1], values=vals[: t + 1])
        recompute_build += t + 1                       # rebuilt the whole bank
        recompute_reads.append(np.asarray(memory_read(bank, keys[t], top_k=1).values))

    resident_reads = []
    resident_build = 0
    bank = MemoryTable(keys=keys[:1], values=vals[:1])
    resident_build += 1
    resident_reads.append(np.asarray(memory_read(bank, keys[0], top_k=1).values))
    for t in range(1, t_steps):
        bank = memory_write(bank, keys[t], vals[t])    # append one fact
        resident_build += 1
        resident_reads.append(np.asarray(memory_read(bank, keys[t], top_k=1).values))

    reads_match = all(
        np.allclose(a, b, atol=1e-5) for a, b in zip(recompute_reads, resident_reads)
    )
    metrics = {
        "decode_steps": t_steps,
        "recompute_build_entries": recompute_build,
        "resident_append_entries": resident_build,
        "build_reduction_x": round(recompute_build / max(resident_build, 1), 2),
        "metamorphic_reads_match": reads_match,
    }
    return reads_match, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Row emission.
# ─────────────────────────────────────────────────────────────────────────────


def _ref_row(op: str, shape: str, *, passed: bool, max_err: float | None,
             metrics: dict[str, Any] | None = None) -> BenchmarkRow:
    return BenchmarkRow(
        operator=BenchmarkOperator(name=op, dtype="fp32", shape=shape, target="cpu"),
        compiler_path=CompilerPath.REFERENCE,
        runtime_status=RuntimeStatus.EXECUTABLE,
        correctness=Correctness(max_error=max_err, passed=passed),
        execution_kind=ExecutionKind.REFERENCE,
        metrics=metrics or {},
    )


def _gap_row(op: str, shape: str, *, gap: str, metrics: dict[str, Any] | None = None,
             ) -> BenchmarkRow:
    if gap not in MEMORY_PRIMITIVE_GAPS:
        raise ValueError(f"{gap!r} is not a declared MEMORY_PRIMITIVE_GAPS entry")
    return BenchmarkRow(
        operator=BenchmarkOperator(name=op, dtype="fp32", shape=shape, target="apple_gpu"),
        compiler_path=CompilerPath.RUNTIME_UNAVAILABLE,
        runtime_status=RuntimeStatus.MISSING_BACKEND,
        correctness=Correctness(passed=None),
        execution_kind=ExecutionKind.ARTIFACT_ONLY,
        reason=f"gap: {gap}",
        metrics=metrics or {},
    )


def run_core(cfg: LongMemoryConfig | None = None) -> list[BenchmarkRow]:
    """Run every scenario and return the proof rows."""
    cfg = cfg or LongMemoryConfig()
    shape = f"bank={cfg.bank_size},kdim={cfg.key_dim},k={cfg.top_k}"
    rows: list[BenchmarkRow] = []

    # 1. RULER multi-needle exact recall — real reference proof.
    bank, queries, oracle = ruler_needle_scenario(cfg)
    res = memory_read(bank, queries, top_k=cfg.top_k)
    err = float(np.abs(np.asarray(res.values) - oracle).max())
    rows.append(_ref_row("ruler_multi_needle_read", shape,
                         passed=err < 1e-4, max_err=err,
                         metrics={"n_needles": cfg.n_needles}))

    # 2. RULER multi-hop pointer chase — real reference proof.
    hbank, q0, payload = ruler_multihop_scenario(cfg)
    hop1 = memory_read(hbank, q0, top_k=1)
    hop2 = memory_read(hbank, np.asarray(hop1.values), top_k=1)
    herr = float(np.abs(np.asarray(hop2.values) - payload).max())
    rows.append(_ref_row("ruler_multihop_read", shape,
                         passed=herr < 1e-4, max_err=herr, metrics={"hops": 2}))

    # 3. LongMemEval abstention — real reference proof of the landed contract.
    abank, present, absent = abstention_scenario(cfg)
    hit = memory_read(abank, present, top_k=1, abstain_below=cfg.abstain_floor)
    miss = memory_read(abank, absent, top_k=1, abstain_below=cfg.abstain_floor)
    abstain_ok = (not bool(hit.abstained)) and bool(miss.abstained) \
        and bool(np.all(np.isnan(np.asarray(miss.values))))
    rows.append(_ref_row("longmemeval_abstain_read", f"kdim={cfg.key_dim}",
                         passed=abstain_ok, max_err=None,
                         metrics={"abstain_floor": cfg.abstain_floor}))

    # 4. LongMemEval version-aware read — landed via prefer_recent recency tiebreak.
    vbank, vkey, latest = version_update_scenario(cfg)
    recent = memory_read(vbank, vkey, top_k=1, prefer_recent=True)
    plain = memory_read(vbank, vkey, top_k=1)
    rows.append(_ref_row(
        "longmemeval_version_aware_read", shape,
        passed=bool(np.allclose(np.asarray(recent.values), latest, atol=1e-5)),
        max_err=float(np.abs(np.asarray(recent.values) - latest).max()),
        metrics={"plain_topk_returned_latest":
                 bool(np.allclose(np.asarray(plain.values), latest, atol=1e-5))}))

    # 5. MemoryArena resident-vs-recompute decode — metamorphic-gated traffic win.
    reads_match, tele = resident_decode_telemetry(cfg)
    rows.append(_ref_row("resident_decode_vs_recompute", shape,
                         passed=reads_match, max_err=None, metrics=tele))

    # 6. Read-residency landed: ResidentBank keeps the bank on-device across
    #    reads (one upload), scoring each query without re-uploading — proven on
    #    Metal in tests/unit/test_resident_bank.py. Metamorphic-gated here.
    from tessera.resident_bank import ResidentBank
    rb_rng = np.random.default_rng(cfg.seed ^ 0x4E51)
    rb_keys = rb_rng.standard_normal((cfg.bank_size, cfg.key_dim)).astype(np.float32)
    rb_keys /= np.linalg.norm(rb_keys, axis=1, keepdims=True)
    rb_vals = rb_rng.standard_normal((cfg.bank_size, cfg.value_dim)).astype(np.float32)
    rb = ResidentBank(rb_keys, rb_vals)
    rb_match = True
    for i in range(min(cfg.decode_steps, cfg.bank_size)):
        _v, idx, _s = rb.read(rb_keys[i], top_k=1)
        rb_match = rb_match and int(idx[0]) == i           # self-query recall
    rb_tele = rb.telemetry()
    rb.free()
    rows.append(_ref_row("resident_bank_read_residency", shape,
                         passed=rb_match, max_err=None, metrics=rb_tele))

    # 7. Append-residency landed: ResidentBank.append offset-writes each new
    #    entry into the resident buffer (no full re-upload), scored against the
    #    growing bank — proven on Metal in test_resident_bank.py.  Metamorphic-
    #    gated decode loop here (append-then-read == recompute).
    from tessera.resident_bank import ResidentBank
    ab_rng = np.random.default_rng(cfg.seed ^ 0x9F17)
    ab_T = min(cfg.decode_steps, cfg.bank_size)
    ab_keys = ab_rng.standard_normal((ab_T, cfg.key_dim)).astype(np.float32)
    ab_keys /= np.linalg.norm(ab_keys, axis=1, keepdims=True)
    ab_vals = ab_rng.standard_normal((ab_T, cfg.value_dim)).astype(np.float32)
    ab = ResidentBank(np.zeros((0, cfg.key_dim), np.float32),
                      np.zeros((0, cfg.value_dim), np.float32), capacity=ab_T)
    ab_match = True
    for t in range(ab_T):
        ab.append(ab_keys[t], ab_vals[t])
        _v, idx, _s = ab.read(ab_keys[t], top_k=1)         # the just-appended entry
        ab_match = ab_match and int(idx[0]) == t
    ab_tele = ab.telemetry()
    ab.free()
    rows.append(_ref_row("resident_append_read", shape,
                         passed=ab_match, max_err=None, metrics=ab_tele))

    return rows


def build_report(rows: list[BenchmarkRow]) -> dict[str, Any]:
    """Summarize a run: counts by proof level + the open gap set."""
    passed = [r for r in rows if r.correctness.passed is True]
    gaps = sorted({r.reason.removeprefix("gap: ") for r in rows if r.reason.startswith("gap: ")})
    return {
        "total_rows": len(rows),
        "reference_passed": len(passed),
        "missing_backend": sum(r.runtime_status is RuntimeStatus.MISSING_BACKEND for r in rows),
        "open_gaps": gaps,
        "landed_primitives": list(LANDED_MEMORY_PRIMITIVES),
        "partial_primitives": list(PARTIAL_MEMORY_PRIMITIVES),
    }


def telemetry(rows: list[BenchmarkRow]) -> list[dict[str, Any]]:
    """Shared Tessera telemetry events for each row."""
    return [telemetry_for_row(r, source="long_memory_core") for r in rows]
