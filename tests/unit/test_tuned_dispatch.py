"""Tests for the AITER-style tuned-config dispatch DB + correctness-gated autotuning.

Covers: signature validation + hashing/equality; TunedConfig validation; de-dup
(keep-min-latency) on both ``add`` and ``load_csv``; lookup hit/miss/default; CSV
round-trip; two-tier override (override wins, base survives elsewhere); correctness-
gated ``tune`` (picks lowest-latency PASSING candidate, rejects a fast-but-wrong one,
raises when all fail); untuned-worklist pending/tuned diff.
"""

from __future__ import annotations

import pytest

from tessera.compiler.tuned_dispatch import (
    CSV_COLUMNS,
    LIBTYPES,
    ProblemSignature,
    TunedConfig,
    TunedDispatchTable,
    UntunedWorklist,
    tune,
)


# ── ProblemSignature ────────────────────────────────────────────────────────
def _sig(m: int = 64, n: int = 64, k: int = 64, dtype: str = "bf16") -> ProblemSignature:
    return ProblemSignature(gfx="gfx942", cu_num=304, m=m, n=n, k=k, dtype=dtype)


def test_signature_validation_rejects_bad_fields() -> None:
    with pytest.raises(ValueError):
        ProblemSignature(gfx="", cu_num=304, m=64, n=64, k=64, dtype="bf16")
    with pytest.raises(ValueError):
        ProblemSignature(gfx="gfx942", cu_num=304, m=64, n=64, k=64, dtype="")
    with pytest.raises(ValueError):
        ProblemSignature(gfx="gfx942", cu_num=0, m=64, n=64, k=64, dtype="bf16")
    with pytest.raises(ValueError):
        ProblemSignature(gfx="gfx942", cu_num=304, m=0, n=64, k=64, dtype="bf16")
    with pytest.raises(ValueError):
        ProblemSignature(gfx="gfx942", cu_num=304, m=64, n=-1, k=64, dtype="bf16")
    with pytest.raises(ValueError):
        ProblemSignature(gfx="gfx942", cu_num=304, m=64, n=64, k=0, dtype="bf16")


def test_signature_hashing_and_equality() -> None:
    a = _sig()
    b = _sig()
    c = _sig(m=128)
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    # Hashable / usable as dict + set keys.
    d = {a: "x"}
    assert d[b] == "x"
    assert len({a, b, c}) == 2


def test_signature_as_key_and_metadata() -> None:
    a = _sig()
    assert a.as_key() == ("gfx942", 304, 64, 64, 64, "bf16")
    meta = a.as_metadata_dict()
    assert meta == {"gfx": "gfx942", "cu_num": 304, "M": 64, "N": 64, "K": 64,
                    "dtype": "bf16"}


# ── TunedConfig ─────────────────────────────────────────────────────────────
def _cfg(latency: float = 10.0, sig: ProblemSignature | None = None,
         libtype: str = "hipblaslt", solidx: int = 7,
         split_k: int = 1, kernel_name: str = "k0") -> TunedConfig:
    return TunedConfig(
        signature=sig or _sig(),
        libtype=libtype,
        solidx=solidx,
        split_k=split_k,
        kernel_name=kernel_name,
        latency_us=latency,
    )


def test_tunedconfig_validation() -> None:
    with pytest.raises(ValueError):
        _cfg(libtype="not_a_lib")
    with pytest.raises(ValueError):
        _cfg(latency=-0.1)
    with pytest.raises(ValueError):
        _cfg(split_k=0)
    with pytest.raises(ValueError):
        _cfg(kernel_name="")
    # Every documented libtype is accepted.
    for lt in LIBTYPES:
        assert _cfg(libtype=lt).libtype == lt


def test_tunedconfig_row_roundtrip() -> None:
    cfg = _cfg(latency=12.5, solidx=42, split_k=4, kernel_name="gemm_bf16")
    row = cfg.to_row()
    assert set(row) == set(CSV_COLUMNS)
    assert row["solidx"] == 42
    assert row["splitK"] == 4
    assert row["kernelName"] == "gemm_bf16"
    # All-string row (as it would arrive from csv.DictReader) parses back equal.
    str_row = {k: str(v) for k, v in row.items()}
    assert TunedConfig.from_row(str_row) == cfg


# ── de-dup keeps min latency ────────────────────────────────────────────────
def test_add_dedup_keeps_min_latency() -> None:
    table = TunedDispatchTable()
    sig = _sig()
    table.add(_cfg(latency=20.0, sig=sig, kernel_name="slow"))
    table.add(_cfg(latency=5.0, sig=sig, kernel_name="fast"))
    table.add(_cfg(latency=15.0, sig=sig, kernel_name="mid"))
    assert len(table) == 1
    winner = table.lookup(sig)
    assert winner is not None
    assert winner.latency_us == 5.0
    assert winner.kernel_name == "fast"


def test_add_dedup_is_order_independent() -> None:
    sig = _sig()
    t1 = TunedDispatchTable([_cfg(20.0, sig), _cfg(5.0, sig)])
    t2 = TunedDispatchTable([_cfg(5.0, sig), _cfg(20.0, sig)])
    w1, w2 = t1.lookup(sig), t2.lookup(sig)
    assert w1 is not None and w2 is not None
    assert w1.latency_us == w2.latency_us == 5.0


# ── lookup hit / miss / default ─────────────────────────────────────────────
def test_lookup_hit_miss_and_default() -> None:
    sig = _sig()
    other = _sig(m=128)
    table = TunedDispatchTable([_cfg(8.0, sig)])
    hit = table.lookup(sig)
    assert hit is not None and hit.latency_us == 8.0
    assert table.lookup(other) is None
    assert sig in table
    assert other not in table
    default = _cfg(99.0, other, kernel_name="fallback")
    assert table.lookup_or_default(sig, default).latency_us == 8.0
    assert table.lookup_or_default(other, default).kernel_name == "fallback"


def test_signatures_listing() -> None:
    a, b = _sig(), _sig(m=128)
    table = TunedDispatchTable([_cfg(1.0, a), _cfg(2.0, b)])
    keys = {s.as_key() for s in table.signatures()}
    assert keys == {a.as_key(), b.as_key()}
    assert len(table) == 2


# ── CSV round-trip ──────────────────────────────────────────────────────────
def test_csv_roundtrip_yields_equal_table(tmp_path) -> None:
    a, b, c = _sig(), _sig(m=128), _sig(dtype="fp8_e4m3")
    table = TunedDispatchTable([
        _cfg(8.0, a, libtype="hipblaslt", solidx=1, kernel_name="ka"),
        _cfg(3.5, b, libtype="triton", solidx=2, split_k=2, kernel_name="kb"),
        _cfg(11.0, c, libtype="ck", solidx=3, kernel_name="kc"),
    ])
    path = tmp_path / "tuned.csv"
    table.to_csv(path)
    loaded = TunedDispatchTable.load_csv(path)
    assert len(loaded) == len(table) == 3
    for sig in (a, b, c):
        orig, back = table.lookup(sig), loaded.lookup(sig)
        assert orig is not None and back is not None
        assert orig == back


def test_csv_string_roundtrip() -> None:
    table = TunedDispatchTable([_cfg(4.0, _sig(), kernel_name="k")])
    text = table.to_csv_string()
    assert "latency_us" in text.splitlines()[0]
    back = TunedDispatchTable.load_csv_string(text)
    assert len(back) == 1


def test_load_csv_dedups_duplicate_signatures(tmp_path) -> None:
    # Hand-written CSV with two rows for the same signature — load must keep min.
    path = tmp_path / "dupes.csv"
    path.write_text(
        "gfx,cu_num,M,N,K,dtype,libtype,solidx,splitK,kernelName,latency_us\n"
        "gfx942,304,64,64,64,bf16,hipblaslt,1,1,slow,20.0\n"
        "gfx942,304,64,64,64,bf16,triton,2,1,fast,4.0\n"
    )
    loaded = TunedDispatchTable.load_csv(path)
    assert len(loaded) == 1
    win = loaded.lookup(_sig())
    assert win is not None and win.latency_us == 4.0 and win.kernel_name == "fast"


# ── two-tier override ───────────────────────────────────────────────────────
def test_override_wins_but_base_survives_elsewhere() -> None:
    a, b = _sig(), _sig(m=128)
    base = TunedDispatchTable([
        _cfg(5.0, a, kernel_name="base_a"),
        _cfg(6.0, b, kernel_name="base_b"),
    ])
    # Override re-tunes signature `a` only — and deliberately with a *higher*
    # latency to prove the override is authoritative (not re-de-duped on latency).
    override = TunedDispatchTable([_cfg(99.0, a, libtype="asm", kernel_name="override_a")])
    merged = base.with_override(override)

    win_a = merged.lookup(a)
    assert win_a is not None
    assert win_a.kernel_name == "override_a"   # override wins
    assert win_a.latency_us == 99.0            # even though it is slower

    win_b = merged.lookup(b)
    assert win_b is not None
    assert win_b.kernel_name == "base_b"       # base survives where not overridden

    # Base table is unmutated by the merge.
    base_a = base.lookup(a)
    assert base_a is not None and base_a.kernel_name == "base_a"
    assert len(merged) == 2


# ── correctness-gated tune ──────────────────────────────────────────────────
def test_tune_picks_lowest_latency_passing_candidate() -> None:
    sig = _sig()
    candidates = [
        {"kernel_name": "ok_mid", "latency": 10.0, "ok": True, "libtype": "ck"},
        {"kernel_name": "ok_fast", "latency": 3.0, "ok": True, "libtype": "triton"},
        {"kernel_name": "ok_slow", "latency": 20.0, "ok": True, "libtype": "asm"},
    ]
    result = tune(
        sig, candidates,
        correctness_fn=lambda c: c["ok"],
        latency_fn=lambda c: c["latency"],
        kernel_name_fn=lambda c: c["kernel_name"],
        libtype_fn=lambda c: c["libtype"],
    )
    assert result.config.kernel_name == "ok_fast"
    assert result.config.latency_us == 3.0
    assert result.config.libtype == "triton"
    assert result.config.signature == sig
    assert result.candidates_total == 3
    assert result.candidates_passed == 3
    assert result.rejected == 0


def test_tune_rejects_fast_but_wrong_candidate() -> None:
    """The perf-gated-behind-correctness invariant: the lowest-latency candidate
    is WRONG, so it must NOT be chosen — a correct slower one wins instead."""
    sig = _sig()
    candidates = [
        {"kernel_name": "fast_wrong", "latency": 1.0, "ok": False},   # cheapest but WRONG
        {"kernel_name": "correct", "latency": 8.0, "ok": True},
        {"kernel_name": "correct_slow", "latency": 30.0, "ok": True},
    ]
    result = tune(
        sig, candidates,
        correctness_fn=lambda c: c["ok"],
        latency_fn=lambda c: c["latency"],
        kernel_name_fn=lambda c: c["kernel_name"],
    )
    assert result.config.kernel_name == "correct"   # NOT the cheaper fast_wrong
    assert result.config.latency_us == 8.0
    assert result.rejected == 1
    assert result.candidates_passed == 2


def test_tune_never_measures_rejected_candidates() -> None:
    """Correctness runs FIRST; latency_fn must never be called on a failing
    candidate (so a wrong kernel cannot even register a fast time)."""
    sig = _sig()
    measured: list[str] = []

    def latency_fn(c: dict) -> float:
        measured.append(c["kernel_name"])
        return c["latency"]

    candidates = [
        {"kernel_name": "wrong", "latency": 0.1, "ok": False},
        {"kernel_name": "right", "latency": 5.0, "ok": True},
    ]
    tune(sig, candidates, correctness_fn=lambda c: c["ok"], latency_fn=latency_fn,
         kernel_name_fn=lambda c: c["kernel_name"])
    assert "wrong" not in measured
    assert measured == ["right"]


def test_tune_raises_when_all_candidates_fail_correctness() -> None:
    sig = _sig()
    candidates = [
        {"kernel_name": "a", "latency": 1.0, "ok": False},
        {"kernel_name": "b", "latency": 2.0, "ok": False},
    ]
    with pytest.raises(ValueError, match="no candidate passed"):
        tune(sig, candidates, correctness_fn=lambda c: c["ok"],
             latency_fn=lambda c: c["latency"])


def test_tune_raises_on_empty_candidates() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        tune(_sig(), [], correctness_fn=lambda c: True, latency_fn=lambda c: 1.0)


def test_tune_rejects_negative_latency() -> None:
    with pytest.raises(ValueError, match="negative latency"):
        tune(_sig(), [{"ok": True}], correctness_fn=lambda c: True,
             latency_fn=lambda c: -1.0, kernel_name_fn=lambda c: "k")


def test_tune_default_field_extractors() -> None:
    """With no extractor overrides, common dict keys are read for the payload."""
    sig = _sig()
    result = tune(
        sig,
        [{"libtype": "cktile", "solidx": 9, "split_k": 2, "kernel_name": "kd"}],
        correctness_fn=lambda c: True,
        latency_fn=lambda c: 7.0,
    )
    assert result.config.libtype == "cktile"
    assert result.config.solidx == 9
    assert result.config.split_k == 2
    assert result.config.kernel_name == "kd"


# ── untuned worklist ────────────────────────────────────────────────────────
def test_worklist_pending_and_tuned_diff() -> None:
    a, b, c = _sig(), _sig(m=128), _sig(m=256)
    worklist = UntunedWorklist([a, b, c])
    table = TunedDispatchTable([_cfg(5.0, a)])   # only `a` tuned

    pending = {s.as_key() for s in worklist.pending_against(table)}
    tuned = {s.as_key() for s in worklist.tuned_against(table)}
    assert pending == {b.as_key(), c.as_key()}
    assert tuned == {a.as_key()}
    # The two halves partition the worklist.
    assert pending | tuned == {a.as_key(), b.as_key(), c.as_key()}
    assert pending & tuned == set()


def test_worklist_add_remove_and_dedup() -> None:
    a, b = _sig(), _sig(m=128)
    worklist = UntunedWorklist([a, a, b])   # duplicate `a` collapses
    assert len(worklist) == 2
    worklist.add(a)                          # idempotent
    assert len(worklist) == 2
    worklist.add(_sig(m=512))
    assert len(worklist) == 3
    worklist.remove(b)
    assert len(worklist) == 2
    with pytest.raises(KeyError):
        worklist.remove(b)                   # already gone


def test_worklist_csv_roundtrip(tmp_path) -> None:
    a, b = _sig(), _sig(m=128, dtype="fp8_e4m3")
    worklist = UntunedWorklist([a, b])
    path = tmp_path / "worklist.csv"
    worklist.to_csv(path)
    loaded = UntunedWorklist.load_csv(path)
    assert {s.as_key() for s in loaded.signatures} == {a.as_key(), b.as_key()}


def test_worklist_all_pending_against_empty_table() -> None:
    a, b = _sig(), _sig(m=128)
    worklist = UntunedWorklist([a, b])
    empty = TunedDispatchTable()
    assert len(worklist.pending_against(empty)) == 2
    assert worklist.tuned_against(empty) == []
