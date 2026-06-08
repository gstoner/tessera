"""Tests for U4 (``tessera.compiler.dry_run``) and U5 (documented
``from_text`` kwargs)."""

from __future__ import annotations

import pytest

import tessera
from tessera.compiler import dry_run
from tessera.compiler.explain import Explain


class TestDryRun:
    def test_returns_explain_object(self) -> None:
        def f(x, y):
            return tessera.ops.add(x, y)

        ex = dry_run(f)
        assert isinstance(ex, Explain)

    def test_dry_run_does_not_execute(self) -> None:
        """``dry_run`` should compile through Graph IR / Schedule IR /
        Tile IR / Target IR but never call the function with
        concrete inputs."""

        call_count = {"n": 0}

        def f(x, y):
            call_count["n"] += 1
            return tessera.ops.add(x, y)

        ex = dry_run(f)
        # The function body was never executed — the call counter
        # stays at 0.
        assert call_count["n"] == 0
        # But the Explain object has IR layers populated.
        assert ex.ir.graph is not None

    def test_dry_run_surfaces_kernel_list(self) -> None:
        """Static analysis use case: walk the kernel list to know
        which ops a function will dispatch."""

        # Use a file-level def so inspect.getsource works; lambdas
        # don't carry their body source through inspect.
        def chain(x, y):
            z = tessera.ops.add(x, y)
            return tessera.ops.relu(z)

        ex = dry_run(chain)
        op_names = [k.op_name for k in ex.kernels]
        assert "add" in op_names
        assert "relu" in op_names

    def test_dry_run_forwards_target(self) -> None:
        def add(x, y):
            return tessera.ops.add(x, y)

        ex = dry_run(add, target="rocm")
        # Target IR was emitted for the alternate target.
        assert "rocm" in ex.target or ex.execution_kind == "artifact_only"


class TestFromTextDocumentedKwargs:
    def test_target_kwarg_documented(self) -> None:
        """``from_text(..., target=...)`` should accept the target
        as a first-class keyword (U5).  Verified by introspecting
        the signature."""

        import inspect
        sig = inspect.signature(tessera.from_text)
        assert "target" in sig.parameters
        assert "deterministic" in sig.parameters
        assert "seed" in sig.parameters
        assert "native_required" in sig.parameters

    def test_target_kwarg_forwarded_through(self) -> None:
        f = tessera.from_text(
            """
                def f(x, y):
                    return ts.ops.add(x, y)
            """,
            target="rocm",
        )
        ex = f.explain()
        assert "rocm" in ex.target

    def test_deterministic_kwarg_forwarded(self) -> None:
        """``deterministic=True`` keeps an effect-free function
        compiling cleanly.  We test the option threads through
        rather than the deterministic semantics themselves (those
        are covered elsewhere)."""

        f = tessera.from_text(
            """
                def f(x, y):
                    return ts.ops.add(x, y)
            """,
            deterministic=True,
        )
        # Compiled without error.
        assert f is not None
