"""Guard: there is exactly ONE `TesseraJitError` class across the JIT lanes.

Historically the `@tessera.jit` decoration lane (`compiler.jit`) and the
production GraphFn / runtime lane (`_jit_boundary`) each defined their own
`TesseraJitError`. Same name, different classes — so `except TesseraJitError` /
`pytest.raises(TesseraJitError)` silently missed errors from the "other" lane
depending on which module the name was imported from. They are now unified
(`compiler.jit` re-exports `_jit_boundary`'s class); this test locks that so the
footgun can't reappear.
"""

import importlib

import tessera


def test_single_canonical_jit_error_class():
    jb = importlib.import_module("tessera._jit_boundary")
    jitmod = importlib.import_module("tessera.compiler.jit")
    canonical = jb.TesseraJitError
    assert jitmod.TesseraJitError is canonical          # decoration lane
    assert tessera.TesseraJitError is canonical          # public export
    # RuntimeError base → every `except Exception` / `except RuntimeError` matches.
    assert issubclass(canonical, RuntimeError)


def test_decoration_error_caught_via_either_import_path():
    """An error raised by the @jit decoration lane is catchable through the name
    imported from EITHER module (the footgun this guards against)."""
    from tessera._jit_boundary import TesseraJitError as FromBoundary

    def bad(x, w):
        h = tessera.ops.silu(tessera.ops.matmul(x, w))
        # post-loop residual the AST emission rejects on a non-apple_gpu target
        for _ in range(2):
            h = tessera.ops.silu(tessera.ops.matmul(h, w))
        return tessera.ops.add(h, x if x.shape == h.shape else h)

    import pytest
    with pytest.raises(FromBoundary):  # caught via the _jit_boundary name
        tessera.jit(target="cpu")(bad)
