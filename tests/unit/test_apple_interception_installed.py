"""The Apple GPU trace interceptor must actually be installed.

PR #492 review caught that a refactor of `_enforce_storage_dtype_preservation`
replaced a region of `tessera/__init__.py` that spanned the
`install_apple_gpu_interception(ops)` call, deleting it. Nothing else in the
repo installs it, so the eight canonical intercepted ops silently reverted to
the numpy reference: `ops.rmsnorm(..., rows=..., cols=...)` then rejects the
trace-only kwargs and never returns the `TraceRef` the Apple encode path needs.

The whole suite stayed green, because every test either runs without an
`@auto_batch` trace (where the interceptor is a pass-through by design) or is
Apple-gated and skips on this box. A load-bearing install with no assertion is
one refactor away from vanishing — so this asserts it directly.
"""

from __future__ import annotations

import pytest


CANONICAL_INTERCEPTED = ("rmsnorm", "layer_norm", "softmax", "gelu", "bmm")


def test_interception_module_is_imported_and_installed():
    import tessera
    from tessera import ops

    installer = getattr(
        tessera, "_agpu_intercept", None
    ) or pytest.importorskip("tessera.apple_gpu_ops_interception")
    assert hasattr(installer, "install_apple_gpu_interception"), (
        "the apple_gpu interception module no longer exposes its installer"
    )

    # Every canonical op must be a wrapper, not the bare reference. The
    # interceptor and the dtype enforcement both wrap, so `__wrapped__` is
    # present either way; what matters is that the module was imported and
    # applied at import time rather than silently skipped.
    for name in CANONICAL_INTERCEPTED:
        fn = getattr(ops, name, None)
        assert fn is not None, f"tessera.ops.{name} is missing"
        assert hasattr(fn, "__wrapped__"), (
            f"tessera.ops.{name} is the bare reference — the Apple "
            "interceptor and/or dtype enforcement did not install"
        )


def test_installer_is_referenced_from_package_init():
    """Guard the call site itself, not just the module's importability.

    The regression was the *call* going missing while the module stayed
    importable, so an import-only check would have passed straight through it.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2]
        / "python"
        / "tessera"
        / "__init__.py"
    ).read_text(encoding="utf-8")
    assert "install_apple_gpu_interception(ops)" in source, (
        "tessera/__init__.py no longer calls install_apple_gpu_interception; "
        "nothing else installs it, so the Apple trace path is dead"
    )


def test_dtype_enforcement_runs_after_interception():
    """Ordering is load-bearing in both directions.

    The interceptor rebinds rmsnorm / layer_norm / softmax / gelu / bmm, so
    enforcing dtype first leaves exactly those unprotected. Enforcing last is
    safe for traces because a TraceRef has no `.dtype`, so the outer wrapper
    returns early and passes the call through untouched.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2]
        / "python"
        / "tessera"
        / "__init__.py"
    ).read_text(encoding="utf-8")
    install_at = source.index("install_apple_gpu_interception(ops)")
    enforce_at = source.index("_enforce_storage_dtype_preservation(ops)")
    assert install_at < enforce_at, (
        "dtype enforcement must run AFTER the Apple interceptor so it wraps "
        "outermost; otherwise the five rebound ops lose their dtype contract"
    )
