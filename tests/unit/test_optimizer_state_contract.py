"""Optimizer state has ONE documented fresh-start value, and says so.

Correctness-audit finding M-4 (MSW-3): every stateful optimizer read its
slots straight out of the caller's dict, so `state={}` surfaced as a raw
`KeyError('velocity')` raised from inside a tree map — an exception naming
neither the optimizer, the contract, nor the fix.

The refusal is deliberate rather than a convenience gap. An empty or partial
dict is, in practice, state that got dropped between steps: a checkpoint that
saved parameters but not slots, a tree rebuilt by name, a comprehension that
filtered. Silently treating it as a fresh start discards the accumulated
moments and degrades training with no error at all — so it fails closed
(#21a), and only `None` starts fresh.
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera import optim


#: Every stateful optimizer, with the slots its state must carry.
#: `adam` and `lamb` delegate to `adamw`, and are listed because delegation is
#: an implementation detail that a refactor could quietly change.
STATEFUL = {
    "momentum": ("velocity",),
    "nesterov": ("velocity",),
    "adamw": ("m", "v", "step"),
    "adam": ("m", "v", "step"),
    "adafactor": ("v", "step"),
    "lion": ("m", "step"),
    "muon": ("velocity",),
    "lamb": ("m", "v", "step"),
}


def _params():
    return {"w": np.array([1.0, -2.0, 0.5], dtype=np.float32)}


def _grads():
    return {"w": np.array([0.1, 0.2, -0.3], dtype=np.float32)}


def _step(name, state):
    return getattr(optim, name)(_params(), _grads(), state, lr=0.1)


@pytest.mark.parametrize("name", sorted(STATEFUL))
def test_none_starts_fresh_and_round_trips(name):
    """The documented init works, and its own output is accepted back."""
    params1, state1 = _step(name, None)
    assert set(STATEFUL[name]) <= set(state1), (
        f"{name} returned state missing its own required slots: {sorted(state1)}"
    )
    params2, _ = _step(name, state1)
    assert np.all(np.isfinite(params1["w"])) and np.all(np.isfinite(params2["w"]))


@pytest.mark.parametrize("name", sorted(STATEFUL))
def test_empty_state_is_refused_by_name_not_keyerror(name):
    """`state={}` is dropped state, and the diagnostic has to say so."""
    with pytest.raises(ValueError) as excinfo:
        _step(name, {})
    message = str(excinfo.value)
    assert "state=None" in message, "the diagnostic must name the fix"
    # The PUBLIC name, not the delegate's. `adam` and `lamb` forward to
    # `adamw`, and telling their caller that `adamw` rejected the state names
    # an optimizer they never called (review on #693).
    assert name in message, (
        f"the diagnostic named a different optimizer than {name!r}: {message}"
    )
    for slot in STATEFUL[name]:
        assert slot in message, f"the diagnostic must name the missing slot {slot!r}"


@pytest.mark.parametrize("name", sorted(STATEFUL))
def test_partial_state_is_refused(name):
    """Dropping ONE slot is the realistic case, and the dangerous one.

    A wholly empty dict is usually a visible mistake. A dict that lost one
    slot looks valid, and silently restarting just that moment is exactly the
    quiet degradation this contract exists to prevent.
    """
    _, full = _step(name, None)
    for slot in STATEFUL[name]:
        partial = {k: v for k, v in full.items() if k != slot}
        with pytest.raises(ValueError, match=slot):
            _step(name, partial)


@pytest.mark.parametrize("name", sorted(STATEFUL))
def test_refusal_is_not_a_keyerror(name):
    """Pinned separately: `KeyError` IS a `LookupError`, not a `ValueError`.

    Without this, re-introducing the raw lookup would still satisfy the
    tests above if someone loosened them to `Exception`.
    """
    with pytest.raises(Exception) as excinfo:
        _step(name, {})
    assert not isinstance(excinfo.value, KeyError), (
        f"{name} still raises a raw KeyError: {excinfo.value!r}"
    )


def test_extra_slots_are_left_alone():
    """Refusal is about MISSING slots, not unknown ones.

    `master_params` is a real optional slot (present only when
    `master_dtype` is set), so a contract that rejected unknown keys would
    refuse valid state.
    """
    _, state = _step("adamw", None)
    state = dict(state, some_future_slot=object())
    params, _ = _step("adamw", state)
    assert np.all(np.isfinite(params["w"]))


# --- mixed precision: the master slot is not optional (review on #693) ---


def _mixed_precision_state():
    """One completed bf16-storage step with fp32 master weights."""
    params = {"w": np.array([1.0, -2.0, 0.5], dtype=np.float16)}
    grads = {"w": np.array([0.1, 0.2, -0.3], dtype=np.float16)}
    return params, grads, optim.adamw(params, grads, None, lr=0.1,
                                      master_dtype="fp32")[1]


def test_mixed_precision_state_round_trips():
    """The guard must not refuse valid mixed-precision state."""
    params, grads, state = _mixed_precision_state()
    assert "master_params" in state
    out, _ = optim.adamw(params, grads, state, lr=0.1, master_dtype="fp32")
    assert np.all(np.isfinite(np.asarray(out["w"], dtype=np.float32)))


def test_lost_master_params_is_refused_when_master_dtype_is_set():
    params, grads, state = _mixed_precision_state()
    stripped = {k: v for k, v in state.items() if k != "master_params"}
    with pytest.raises(ValueError, match="master_params"):
        optim.adamw(params, grads, stripped, lr=0.1, master_dtype="fp32")


def test_lost_master_params_is_allowed_when_master_dtype_is_absent():
    """The slot is required *because of* master_dtype, not always.

    An fp32 run never has this slot, and refusing it there would reject
    every ordinary checkpoint.
    """
    params, grads, state = _mixed_precision_state()
    stripped = {k: v for k, v in state.items() if k != "master_params"}
    out, _ = optim.adamw(params, grads, stripped, lr=0.1)
    assert np.all(np.isfinite(np.asarray(out["w"], dtype=np.float32)))


def test_the_refused_fallback_really_would_lose_precision():
    """Why the refusal is worth having, demonstrated rather than asserted.

    With the slot absent, `_master_tree` rebuilds the master weights by
    upcasting the ROUNDED fp16 storage. This pins that the reconstruction
    differs from the master weights the run was actually carrying — so
    accepting the stripped state would silently change the trajectory, not
    merely re-derive the same numbers.
    """
    params, _grads, state = _mixed_precision_state()
    true_master = np.asarray(state["master_params"]["w"], dtype=np.float32)
    rebuilt = np.asarray(
        optim._master_tree(params, {}, "fp32")["w"], dtype=np.float32)
    assert not np.array_equal(true_master, rebuilt), (
        "if these agreed, the missing slot would be harmless and the "
        "refusal would be pointless — the test itself would be the bug"
    )


def test_attach_master_params_is_the_explicit_migration():
    """Enabling mixed precision mid-run is legitimate, and has one door."""
    params = {"w": np.array([1.0, -2.0, 0.5], dtype=np.float16)}
    grads = {"w": np.array([0.1, 0.2, -0.3], dtype=np.float16)}
    _, fp32_state = optim.adamw(params, grads, None, lr=0.1)
    assert "master_params" not in fp32_state

    migrated = optim.attach_master_params(fp32_state, params, master_dtype="fp32")
    out, _ = optim.adamw(params, grads, migrated, lr=0.1, master_dtype="fp32")
    assert np.all(np.isfinite(np.asarray(out["w"], dtype=np.float32)))


def test_attach_master_params_does_not_clobber_existing_master():
    """Applied twice it must be a no-op, not a silent downgrade.

    A migration helper that re-derived the master weights from rounded
    storage on every call would reintroduce the defect it exists to close.
    """
    params, _grads, state = _mixed_precision_state()
    again = optim.attach_master_params(state, params, master_dtype="fp32")
    assert np.array_equal(
        np.asarray(again["master_params"]["w"], dtype=np.float32),
        np.asarray(state["master_params"]["w"], dtype=np.float32))
