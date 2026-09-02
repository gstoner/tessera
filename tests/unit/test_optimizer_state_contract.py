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
