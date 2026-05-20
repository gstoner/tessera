"""Tests for ``tessera.from_text`` — the P1-6 notebook-safe factory.

Replaces the ``exec(source); ts.jit(ns[name], source=source)`` dance
with a single blessed call.
"""

from __future__ import annotations

import pytest

import tessera as ts
from tessera.compiler.diagnostics import JitDiagnosticCode


class TestSingleDef:
    def test_single_def_name_auto_detected(self) -> None:
        f = ts.from_text("""
            def f(x, y):
                return ts.ops.add(x, y)
        """)
        # The function compiles end-to-end; explain() succeeds.
        ex = f.explain()
        assert "f" in ex.function_name

    def test_single_def_emits_source_provided_diagnostic(self) -> None:
        f = ts.from_text("""
            def f(x):
                return ts.ops.relu(x)
        """)
        codes = [d.code_value for d in f.explain().diagnostics]
        # The JIT pipeline emits JIT_SOURCE_PROVIDED when source= is
        # passed explicitly — that's the contract from_text relies on.
        assert JitDiagnosticCode.SOURCE_PROVIDED.value in codes


class TestMultiDef:
    def test_multi_def_requires_name(self) -> None:
        source = """
            def helper(x):
                return x

            def main(x):
                return helper(x)
        """
        with pytest.raises(ValueError, match="multiple top-level"):
            ts.from_text(source)

    def test_multi_def_with_named_function(self) -> None:
        f = ts.from_text("""
            def helper(x):
                return ts.ops.relu(x)

            def main(x, y):
                return helper(ts.ops.add(x, y))
        """, name="main")
        assert "main" in f.explain().function_name


class TestErrors:
    def test_no_def_raises(self) -> None:
        with pytest.raises(ValueError, match="no top-level"):
            ts.from_text("x = 1")

    def test_syntax_error_raises(self) -> None:
        with pytest.raises(ValueError, match="does not parse"):
            ts.from_text("def f(x: this is not valid: return x")

    def test_name_not_in_source_raises(self) -> None:
        with pytest.raises(KeyError, match="not defined in the source"):
            ts.from_text("""
                def f(x):
                    return x
            """, name="g")

    def test_passing_source_keyword_raises(self) -> None:
        """Python's own TypeError fires when ``source=`` is passed
        twice (once positional + once keyword).  Either way, the user
        can't accidentally bypass the factory's exec semantics."""

        with pytest.raises(TypeError):
            ts.from_text(
                "def f(x): return x",
                source="def f(x): return x",  # type: ignore[arg-type]
            )


class TestNamespaceDefaults:
    def test_ts_alias_is_available(self) -> None:
        # The source references ``ts.ops.add`` without an explicit
        # import — from_text pre-populates the namespace.
        f = ts.from_text("""
            def f(x, y):
                return ts.ops.add(x, y)
        """)
        assert f is not None

    def test_tessera_alias_is_available(self) -> None:
        f = ts.from_text("""
            def f(x, y):
                return tessera.ops.add(x, y)
        """)
        assert f is not None

    def test_custom_namespace_extras_are_exposed(self) -> None:
        sentinel = {"answer": 42}
        # The source references SENTINEL_ANSWER — succeeds only if
        # the custom namespace was wired through.
        f = ts.from_text("""
            def f(x):
                _ = SENTINEL_ANSWER["answer"]  # noqa: F841
                return ts.ops.relu(x)
        """, namespace={"SENTINEL_ANSWER": sentinel})
        # Calling the function exercises the namespace.
        import numpy as np
        out = f(np.array([1.0, -1.0]))
        assert out is not None


class TestNotebookHappyPath:
    def test_minimal_quickstart_from_README(self) -> None:
        """The exact pattern documented in the API reference must work."""

        f = ts.from_text("""
            def f(x):
                return ts.ops.relu(x)
        """)
        ex = f.explain()
        # Standard summary works.
        text = str(ex)
        assert "tessera.jit" in text
        assert ex.execution_kind in (
            "native_cpu", "reference_cpu", "fallback_eager",
        )
