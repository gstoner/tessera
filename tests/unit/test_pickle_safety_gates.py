"""Regression: pickle-based deserialization is gated behind an
explicit opt-in/out parameter.

Findings audit (2026-05-19) flagged that ``aot.load`` and
``checkpoint.load_state`` would silently call ``pickle.loads`` on
their respective sidecar / treedef blobs.  Either entry point
loading a maliciously-crafted artifact could execute arbitrary
code.

The fix:

  * ``aot.load(path, allow_pickle=False)`` — default ``False``;
    refuses to load the ``callable.pkl`` sidecar.  Setting
    ``allow_pickle=True`` is required to materialize the callable.
  * ``checkpoint.load_state(path, trust_treedef=True)`` — default
    ``True`` (preserves same-process ergonomics); passing
    ``trust_treedef=False`` makes the function raise loudly
    instead of unpickling the treedef.

Both surfaces also carry prominent docstring warnings about
the security implications.
"""

from __future__ import annotations

import inspect

import pytest


def test_aot_load_defaults_allow_pickle_to_false() -> None:
    from tessera import aot

    sig = inspect.signature(aot.load)
    assert "allow_pickle" in sig.parameters, (
        "aot.load must expose an `allow_pickle` keyword to gate the "
        "pickle.loads of callable.pkl"
    )
    assert sig.parameters["allow_pickle"].default is False, (
        "default must be False so untrusted artifacts are safe to "
        "inspect without executing embedded code"
    )


def test_aot_load_docstring_warns_about_pickle() -> None:
    from tessera import aot

    doc = aot.load.__doc__ or ""
    assert "pickle" in doc.lower()
    assert "arbitrary code" in doc.lower(), (
        "aot.load docstring must explicitly warn about arbitrary code "
        "execution risk; that's the M0/M3 honest-reporting contract"
    )


def test_checkpoint_load_state_exposes_trust_treedef() -> None:
    from tessera import checkpoint

    sig = inspect.signature(checkpoint.load_state)
    assert "trust_treedef" in sig.parameters, (
        "checkpoint.load_state must expose a `trust_treedef` keyword "
        "so callers loading untrusted checkpoints can opt out"
    )


def test_checkpoint_load_sharded_forwards_trust_treedef() -> None:
    # The sharded loader unpickles a __treedef__ exactly like load_state, so it must
    # expose the same opt-out (it forwards to load_state internally).
    from tessera import checkpoint

    sig = inspect.signature(checkpoint.load_sharded)
    assert "trust_treedef" in sig.parameters, (
        "checkpoint.load_sharded must forward `trust_treedef` to load_state "
        "so sharded checkpoints from untrusted sources can opt out of pickle"
    )


def test_checkpoint_load_state_refuses_untrusted_treedef(tmp_path) -> None:
    """``trust_treedef=False`` is the strict path — even on a clean
    artifact it refuses to unpickle the treedef."""
    import numpy as np

    import tessera
    from tessera import checkpoint

    state = {"params": {"w": np.zeros((2, 2), dtype=np.float32)}}
    ckpt = tmp_path / "state.npz"
    checkpoint.save_state(state, ckpt)

    # Explicit-trust path works (and does not warn).
    loaded = checkpoint.load_state(ckpt, trust_treedef=True)
    assert "params" in loaded

    # Explicit opt-out raises with a precise diagnostic.
    with pytest.raises(checkpoint.CheckpointError, match="trust_treedef=False"):
        checkpoint.load_state(ckpt, trust_treedef=False)


def test_checkpoint_load_state_implicit_default_warns(tmp_path) -> None:
    """Calling ``load_state`` without an explicit ``trust_treedef=`` emits a
    DeprecationWarning (the implicit-trust default is on its way out). Locking it
    here keeps the contract covered AND keeps the warning from leaking as test
    noise elsewhere — every other call site passes ``trust_treedef=`` explicitly."""
    import numpy as np

    from tessera import checkpoint

    state = {"params": {"w": np.zeros((2, 2), dtype=np.float32)}}
    ckpt = tmp_path / "state.npz"
    checkpoint.save_state(state, ckpt)

    with pytest.warns(DeprecationWarning, match="trust_treedef"):
        loaded = checkpoint.load_state(ckpt)
    assert "params" in loaded


def test_checkpoint_load_state_docstring_warns_about_pickle() -> None:
    from tessera import checkpoint

    doc = checkpoint.load_state.__doc__ or ""
    assert "pickle" in doc.lower()
    assert "arbitrary code" in doc.lower()
