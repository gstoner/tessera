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


def test_checkpoint_load_state_refuses_untrusted_treedef(tmp_path) -> None:
    """``trust_treedef=False`` is the strict path — even on a clean
    artifact it refuses to unpickle the treedef."""
    import numpy as np

    import tessera
    from tessera import checkpoint

    state = {"params": {"w": np.zeros((2, 2), dtype=np.float32)}}
    ckpt = tmp_path / "state.npz"
    checkpoint.save_state(state, ckpt)

    # Default path still works (trust_treedef=True).
    loaded = checkpoint.load_state(ckpt)
    assert "params" in loaded

    # Explicit opt-out raises with a precise diagnostic.
    with pytest.raises(checkpoint.CheckpointError, match="trust_treedef=False"):
        checkpoint.load_state(ckpt, trust_treedef=False)


def test_checkpoint_load_state_docstring_warns_about_pickle() -> None:
    from tessera import checkpoint

    doc = checkpoint.load_state.__doc__ or ""
    assert "pickle" in doc.lower()
    assert "arbitrary code" in doc.lower()
