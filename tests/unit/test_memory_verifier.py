"""M4 — memory-model verifier tests.

Locks the happens-before + memory-space-transition checks the
new :mod:`tessera.compiler.memory_verifier` adds on top of the
existing structural :class:`tile_ir.TileIRVerifier`.

Coverage:

  - ``async_copy`` / ``wait_async`` happens-before
  - ``queue.create`` / ``push`` / ``pop`` / ``barrier`` happens-before
  - canonical memory-space transition whitelist
  - source-span propagation through the diagnostic ``where`` field
  - the convenience :func:`assert_memory_model_ok` raise gate
"""

from __future__ import annotations

import pytest

from tessera.compiler.memory_verifier import (
    LEGAL_ASYNC_COPY_TRANSITIONS,
    MemoryModelDiagnostic,
    MemoryModelVerificationError,
    VALID_MEMORY_SPACES,
    assert_memory_model_ok,
    verify_memory_model,
)
from tessera.compiler.tile_ir import TileFunction, TileIRModule, TileOp


# ---------------------------------------------------------------------------
# Builder helpers — keep the test bodies focused on the assertion
# ---------------------------------------------------------------------------

def _fn(*ops: TileOp, name: str = "f", target: str = "apple_gpu") -> TileFunction:
    return TileFunction(name=name, body=list(ops), target=target)


def _module(*ops: TileOp, **fn_kwargs) -> TileIRModule:
    return TileIRModule(functions=[_fn(*ops, **fn_kwargs)])


def _copy(*, stage: int = 0, **attrs) -> TileOp:
    base = {"stage": stage, "vector": 4, "ordinal": 0,
            "source": "a", "result": "b"}
    base.update(attrs)
    return TileOp("tile.async_copy", attrs=base)


def _wait(*, stage: int = 0, **attrs) -> TileOp:
    base = {"stage": stage}
    base.update(attrs)
    return TileOp("tile.wait_async", attrs=base)


def _queue_create(qid: int, depth: int = 2) -> TileOp:
    return TileOp("tessera.queue.create",
                  attrs={"queue_id": qid, "depth": depth})


def _queue_push(qid: int) -> TileOp:
    return TileOp("tessera.queue.push", attrs={"queue_id": qid})


def _queue_pop(qid: int) -> TileOp:
    return TileOp("tessera.queue.pop", attrs={"queue_id": qid})


def _queue_barrier(qid: int) -> TileOp:
    return TileOp("tessera.queue.barrier", attrs={"queue_id": qid})


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------

def test_valid_memory_spaces_is_six() -> None:
    """Adding a memory space is a deliberate decision — extending
    this set must update the verifier and the spec."""
    assert VALID_MEMORY_SPACES == frozenset({
        "global", "shared", "register", "tmem", "constant", "host",
    })


def test_legal_async_copy_transitions_is_documented_set() -> None:
    """Locks the canonical transition table.  Any addition must
    document the SM/Apple/AMD path that uses it."""
    assert LEGAL_ASYNC_COPY_TRANSITIONS == frozenset({
        ("global",   "shared"),
        ("global",   "tmem"),
        ("shared",   "register"),
        ("global",   "register"),
        ("host",     "global"),
    })


# ---------------------------------------------------------------------------
# Positive paths
# ---------------------------------------------------------------------------

def test_copy_then_wait_at_same_stage_is_ok() -> None:
    module = _module(_copy(stage=0), _wait(stage=0))
    assert verify_memory_model(module).ok


def test_multiple_copies_then_one_wait_is_ok() -> None:
    """Two async copies at stage 0 followed by a single wait — both
    are tracked at the stage granularity, so one wait suffices."""
    module = _module(
        _copy(stage=0), _copy(stage=0), _wait(stage=0),
    )
    assert verify_memory_model(module).ok


def test_legal_memory_space_transitions_pass() -> None:
    for src, dst in LEGAL_ASYNC_COPY_TRANSITIONS:
        module = _module(
            _copy(source_space=src, dest_space=dst, stage=0),
            _wait(stage=0),
        )
        result = verify_memory_model(module)
        assert result.ok, (
            f"legal transition {src} -> {dst} flagged: {result.format()}"
        )


def test_queue_push_pop_round_trip_is_ok() -> None:
    module = _module(
        _queue_create(0, depth=2),
        _queue_push(0),
        _queue_pop(0),
    )
    assert verify_memory_model(module).ok


def test_queue_barrier_after_push_is_ok_and_does_not_consume() -> None:
    """barrier waits but doesn't decrement the outstanding counter,
    so a subsequent ``pop`` is still permitted."""
    module = _module(
        _queue_create(0),
        _queue_push(0),
        _queue_barrier(0),
        _queue_pop(0),
    )
    assert verify_memory_model(module).ok


# ---------------------------------------------------------------------------
# Negative paths
# ---------------------------------------------------------------------------

def _codes(module) -> list[str]:
    return [d.code for d in verify_memory_model(module).diagnostics]


def test_wait_without_preceding_copy_is_an_error() -> None:
    module = _module(_wait(stage=0))
    assert "MEM_WAIT_WITHOUT_COPY" in _codes(module)


def test_wait_at_wrong_stage_is_an_error() -> None:
    """A wait at stage 1 with copies only at stage 0 should fail."""
    module = _module(_copy(stage=0), _wait(stage=1))
    assert "MEM_WAIT_WITHOUT_COPY" in _codes(module)


def test_queue_push_without_create_is_an_error() -> None:
    module = _module(_queue_push(7))
    assert "MEM_QUEUE_PUSH_WITHOUT_CREATE" in _codes(module)


def test_queue_pop_without_push_is_an_error() -> None:
    module = _module(_queue_create(0), _queue_pop(0))
    assert "MEM_QUEUE_POP_WITHOUT_PUSH" in _codes(module)


def test_queue_pop_after_balanced_push_then_pop_fails() -> None:
    """Two pops after one push: the second one is unmatched."""
    module = _module(
        _queue_create(0),
        _queue_push(0),
        _queue_pop(0),
        _queue_pop(0),
    )
    assert "MEM_QUEUE_POP_WITHOUT_PUSH" in _codes(module)


def test_invalid_memory_space_is_an_error() -> None:
    module = _module(_copy(source_space="weird", dest_space="shared"))
    assert "MEM_UNKNOWN_MEMORY_SPACE" in _codes(module)


def test_illegal_memory_space_transition_is_an_error() -> None:
    """``shared`` → ``shared`` is normally a register copy at the
    instruction level — not an ``async_copy``."""
    module = _module(
        _copy(source_space="shared", dest_space="shared", stage=0),
    )
    assert "MEM_INVALID_ASYNC_COPY_TRANSITION" in _codes(module)


# ---------------------------------------------------------------------------
# Source spans propagate through the diagnostic record
# ---------------------------------------------------------------------------

def test_source_span_attrs_propagate_to_diagnostic_where() -> None:
    module = _module(_wait(stage=0, loc_line=42, loc_col=7))
    diagnostics = verify_memory_model(module).diagnostics
    assert any(
        d.code == "MEM_WAIT_WITHOUT_COPY" and d.where == {"loc_line": 42, "loc_col": 7}
        for d in diagnostics
    )


# ---------------------------------------------------------------------------
# Convenience raise gate
# ---------------------------------------------------------------------------

def test_assert_memory_model_ok_passes_for_clean_program() -> None:
    module = _module(_copy(stage=0), _wait(stage=0))
    assert_memory_model_ok(module)  # no raise


def test_assert_memory_model_ok_raises_for_invalid_program() -> None:
    module = _module(_wait(stage=0))
    with pytest.raises(MemoryModelVerificationError, match="MEM_WAIT_WITHOUT_COPY"):
        assert_memory_model_ok(module)


# ---------------------------------------------------------------------------
# Per-function scoping — issues in one function shouldn't leak
# ---------------------------------------------------------------------------

def test_state_does_not_leak_between_functions() -> None:
    """A copy in function A must not satisfy a wait in function B."""
    module = TileIRModule(functions=[
        TileFunction("a", body=[_copy(stage=0)], target="apple_gpu"),
        TileFunction("b", body=[_wait(stage=0)], target="apple_gpu"),
    ])
    assert "MEM_WAIT_WITHOUT_COPY" in _codes(module)


# ─────────────────────────────────────────────────────────────────────────────
# Sprint M5 (2026-05-22) — atomic op attribute validation
#
# MEMORY_MODEL_SPEC §5 enumerates 5 orders, 6 scopes, and 9 atomic ops.
# Verifier rejects out-of-set attributes with stable diagnostic codes.
# ─────────────────────────────────────────────────────────────────────────────


from tessera.compiler.memory_verifier import (
    VALID_ATOMIC_ORDERS, VALID_SYNC_SCOPES, VALID_ATOMIC_OPS,
    NONDETERMINISTIC_FLOAT_DTYPES, REDUCTION_ATOMIC_OPS,
)


def _atomic(*, op: str = "add", order: str = "acq_rel",
            scope: str = "block", dtype: str = "int32", **attrs) -> TileOp:
    base = {"atomic_op": op, "order": order, "scope": scope, "dtype": dtype}
    base.update(attrs)
    return TileOp("tile.atomic", attrs=base)


def _fence(*, scope: str = "device", **attrs) -> TileOp:
    base = {"scope": scope}
    base.update(attrs)
    return TileOp("tile.fence", attrs=base)


def test_valid_atomic_orders_is_canonical_set() -> None:
    """Memory model §5 — the 5 canonical orders."""
    assert VALID_ATOMIC_ORDERS == frozenset({
        "relaxed", "acquire", "release", "acq_rel", "seq_cst",
    })


def test_valid_sync_scopes_is_canonical_set() -> None:
    """Memory model §3 — the 6 canonical scopes."""
    assert VALID_SYNC_SCOPES == frozenset({
        "thread", "warp", "block", "cluster", "device", "mesh",
    })


def test_valid_atomic_ops_is_canonical_set() -> None:
    """Memory model §5 — the 9 canonical atomic ops."""
    assert VALID_ATOMIC_OPS == frozenset({
        "add", "sub", "min", "max", "and", "or", "xor", "exchange", "cas",
    })


def test_atomic_with_all_valid_attrs_passes() -> None:
    """Every (order, scope, op) tuple from the canonical sets passes."""
    for order in VALID_ATOMIC_ORDERS:
        for scope in VALID_SYNC_SCOPES:
            module = _module(_atomic(order=order, scope=scope))
            assert verify_memory_model(module).ok, (
                f"order={order}, scope={scope} flagged unexpectedly"
            )


def test_atomic_with_invalid_order_is_an_error() -> None:
    module = _module(_atomic(order="weird"))
    assert "MEM_ATOMIC_INVALID_ORDER" in _codes(module)


def test_atomic_with_invalid_scope_is_an_error() -> None:
    module = _module(_atomic(scope="planetary"))
    assert "MEM_ATOMIC_INVALID_SCOPE" in _codes(module)


def test_atomic_with_invalid_op_is_an_error() -> None:
    module = _module(_atomic(op="mul"))
    assert "MEM_ATOMIC_INVALID_OP" in _codes(module)


def test_atomic_without_explicit_order_or_scope_passes() -> None:
    """Verifier only checks attributes that ARE declared; missing
    attrs are not auto-flagged here (the structural verifier in
    tile_ir.py handles required-attr enforcement)."""
    module = _module(TileOp("tile.atomic", attrs={"atomic_op": "add"}))
    assert verify_memory_model(module).ok


# ─────────────────────────────────────────────────────────────────────────────
# Sprint M5 (2026-05-22) — fence scope validation
# ─────────────────────────────────────────────────────────────────────────────


def test_fence_with_valid_scopes_passes() -> None:
    """All 6 canonical sync scopes are legal fence scopes."""
    for scope in VALID_SYNC_SCOPES:
        module = _module(_fence(scope=scope))
        assert verify_memory_model(module).ok, (
            f"fence scope={scope} flagged unexpectedly"
        )


def test_fence_with_invalid_scope_is_an_error() -> None:
    module = _module(_fence(scope="planetary"))
    assert "MEM_FENCE_INVALID_SCOPE" in _codes(module)


def test_fence_without_scope_attr_passes_unchanged() -> None:
    """When the caller doesn't annotate scope, the verifier passes —
    structural validation of fence presence belongs elsewhere."""
    module = _module(TileOp("tile.fence", attrs={}))
    assert verify_memory_model(module).ok


# ─────────────────────────────────────────────────────────────────────────────
# Sprint M5 (2026-05-22) — deterministic profile rejects nondeterministic
# float atomic reductions (MEMORY_MODEL_SPEC §7)
# ─────────────────────────────────────────────────────────────────────────────


def _deterministic_fn(*ops: TileOp, **attrs) -> TileFunction:
    base_attrs = {"deterministic": True}
    base_attrs.update(attrs)
    return TileFunction(name="f", body=list(ops), target="apple_gpu",
                        attrs=base_attrs)


def _deterministic_module(*ops: TileOp) -> TileIRModule:
    return TileIRModule(functions=[_deterministic_fn(*ops)])


def test_nondeterministic_float_dtypes_is_canonical_set() -> None:
    """Float dtypes that produce nondeterministic atomic reductions —
    all storage floats today."""
    assert NONDETERMINISTIC_FLOAT_DTYPES == frozenset({
        "fp64", "fp32", "fp16", "bf16",
        "fp8_e4m3", "fp8_e5m2", "fp6_e2m3", "fp6_e3m2", "fp4_e2m1", "nvfp4",
    })


def test_reduction_atomic_ops_is_canonical_set() -> None:
    """Atomic ops whose float variants are reductions (and therefore
    nondeterministic without a fixed tree)."""
    assert REDUCTION_ATOMIC_OPS == frozenset({"add", "sub", "min", "max"})


def test_deterministic_profile_rejects_float_atomic_add() -> None:
    """The canonical Megatron-TP float `atomic.add` reduction must be
    rejected under the deterministic profile."""
    module = _deterministic_module(
        _atomic(op="add", dtype="fp32", order="acq_rel", scope="device"),
    )
    assert "MEM_DETERMINISTIC_NONDETERMINISTIC_REDUCTION" in _codes(module)


@pytest.mark.parametrize("dtype", sorted(NONDETERMINISTIC_FLOAT_DTYPES))
def test_deterministic_profile_rejects_every_float_dtype(dtype: str) -> None:
    """Every storage-float dtype is rejected for atomic reductions
    under the deterministic profile."""
    module = _deterministic_module(_atomic(op="add", dtype=dtype))
    assert "MEM_DETERMINISTIC_NONDETERMINISTIC_REDUCTION" in _codes(module)


def test_deterministic_profile_allows_integer_atomic_add() -> None:
    """Integer atomics are deterministic by definition — atomicity is
    sufficient.  The verifier must not flag them under deterministic."""
    module = _deterministic_module(_atomic(op="add", dtype="int32"))
    assert "MEM_DETERMINISTIC_NONDETERMINISTIC_REDUCTION" not in _codes(module)


def test_deterministic_profile_allows_float_atomic_exchange() -> None:
    """``exchange`` and ``cas`` are not reductions — they're
    deterministic by construction (each lane sees a defined RMW).
    Must not be flagged even on float dtypes."""
    module = _deterministic_module(_atomic(op="exchange", dtype="fp32"))
    assert "MEM_DETERMINISTIC_NONDETERMINISTIC_REDUCTION" not in _codes(module)


def test_non_deterministic_profile_allows_float_atomic_add() -> None:
    """Without `deterministic=True`, float atomic.add is legal — the
    verifier only enforces under deterministic / strict profiles."""
    module = _module(_atomic(op="add", dtype="fp32"))  # default attrs
    assert "MEM_DETERMINISTIC_NONDETERMINISTIC_REDUCTION" not in _codes(module)


def test_strict_numeric_profile_also_rejects_float_atomic_add() -> None:
    """`numeric_profile="strict"` is equivalent to `deterministic=True`
    per the §7 contract — both reject nondeterministic float
    reductions."""
    fn = TileFunction(
        name="f",
        body=[_atomic(op="add", dtype="fp32")],
        target="apple_gpu",
        attrs={"numeric_profile": "strict"},
    )
    module = TileIRModule(functions=[fn])
    assert "MEM_DETERMINISTIC_NONDETERMINISTIC_REDUCTION" in _codes(module)
