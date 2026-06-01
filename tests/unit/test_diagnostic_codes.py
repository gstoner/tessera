"""Tests for the public diagnostic-code taxonomy (P0-2).

Locks two contracts:

1. ``JitDiagnosticCode`` values are stable and match the strings
   emitted at every call site under ``python/tessera/compiler``.
2. ``FallbackReason`` round-trips through ``Diagnostic.code_value``
   to a stable string.

The strings are public API — renaming one is a breaking change
that ripples through CI artifacts and benchmark JSON.
"""

from __future__ import annotations

import re
from pathlib import Path

from tessera.compiler import (
    Diagnostic,
    FallbackReason,
    JitDiagnosticCode,
    TesseraNativeRequiredError,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPILER_DIR = REPO_ROOT / "python" / "tessera" / "compiler"


class TestJitDiagnosticCode:
    def test_values_are_stable_strings(self) -> None:
        # Lock the public values.  Adding new enum members is fine;
        # changing one of these strings is a breaking change.
        expected = {
            "SOURCE_UNAVAILABLE": "JIT_SOURCE_UNAVAILABLE",
            "SOURCE_PROVIDED": "JIT_SOURCE_PROVIDED",
            "EAGER_FALLBACK_EMPTY": "JIT_EAGER_FALLBACK_EMPTY",
            "EAGER_FALLBACK_UNSUPPORTED_OP": (
                "JIT_EAGER_FALLBACK_UNSUPPORTED_OP"
            ),
            "EAGER_FALLBACK_ARITY": "JIT_EAGER_FALLBACK_ARITY",
            "EAGER_FALLBACK_UNSUPPORTED_BODY": (
                "JIT_EAGER_FALLBACK_UNSUPPORTED_BODY"
            ),
            "EAGER_FALLBACK_CONTROL_FLOW": (
                "JIT_EAGER_FALLBACK_CONTROL_FLOW"
            ),
            "COMPILED_CPU": "JIT_COMPILED_CPU",
            "TARGET_IR_ARTIFACT_ONLY": "JIT_TARGET_IR_ARTIFACT_ONLY",
        }
        for name, value in expected.items():
            assert JitDiagnosticCode[name].value == value

    def test_every_emitted_string_has_an_enum_member(self) -> None:
        """Every ``JIT_*`` code emitted in the compiler tree must
        correspond to an enum member."""

        # Inventory: walk all .py files under python/tessera/compiler/
        # and find strings matching ``"JIT_[A-Z_]+"`` (excluding the
        # diagnostics.py file that defines them).
        pattern = re.compile(r'"(JIT_[A-Z_]+)"')
        used: set[str] = set()
        for path in COMPILER_DIR.glob("*.py"):
            if path.name == "diagnostics.py":
                continue
            for match in pattern.finditer(path.read_text(encoding="utf-8")):
                used.add(match.group(1))
        # Every used string must be in the enum (the test file itself
        # is excluded so this gate doesn't fire on docstrings).
        defined = {c.value for c in JitDiagnosticCode}
        missing = used - defined
        assert missing == set(), (
            f"Compiler tree emits JIT codes not in JitDiagnosticCode: "
            f"{sorted(missing)}.  Either add the enum member or fix "
            f"the typo."
        )


class TestFallbackReason:
    def test_values_are_stable_strings(self) -> None:
        expected = {
            "NON_DARWIN_HOST": "non_darwin_host",
            "APPLE_GPU_RUNTIME_UNAVAILABLE": "apple_gpu_runtime_unavailable",
            "MANIFEST_MISS": "manifest_miss",
            "DTYPE_NOT_COVERED": "dtype_not_covered",
            "SHAPE_OUT_OF_ENVELOPE": "shape_out_of_envelope",
            "CAPABILITY_NOT_READY": "capability_not_ready",
            "OPT_OUT": "opt_out",
            "REFERENCE_FORCED": "reference_forced",
        }
        for name, value in expected.items():
            assert FallbackReason[name].value == value

    def test_every_reason_carries_a_message(self) -> None:
        for reason in FallbackReason:
            msg = reason.message()
            assert isinstance(msg, str) and len(msg) > 10


class TestDiagnosticShape:
    def test_from_fallback_lifts_into_diagnostic(self) -> None:
        d = Diagnostic.from_fallback(FallbackReason.NON_DARWIN_HOST)
        assert d.severity == "warning"
        assert d.code is FallbackReason.NON_DARWIN_HOST
        assert d.code_value == "non_darwin_host"
        assert "Darwin" in d.message or "darwin" in d.message.lower()

    def test_from_jit_lifts_into_diagnostic(self) -> None:
        d = Diagnostic.from_jit(
            JitDiagnosticCode.SOURCE_UNAVAILABLE,
            "no source for AST inspection",
        )
        assert d.code is JitDiagnosticCode.SOURCE_UNAVAILABLE
        assert d.code_value == "JIT_SOURCE_UNAVAILABLE"

    def test_code_value_handles_raw_string(self) -> None:
        """Diagnostic still works when callers pass a raw string code
        (backwards compat with pre-enum emission sites)."""

        d = Diagnostic(severity="info", code="LEGACY_CODE", message="x")
        assert d.code_value == "LEGACY_CODE"


class TestTesseraNativeRequiredError:
    def test_carries_reason_and_code(self) -> None:
        try:
            raise TesseraNativeRequiredError(
                FallbackReason.MANIFEST_MISS,
                target="apple_gpu",
                op_name="custom_op",
            )
        except TesseraNativeRequiredError as exc:
            assert exc.reason is FallbackReason.MANIFEST_MISS
            assert exc.target == "apple_gpu"
            assert "manifest_miss" in str(exc)
            assert "apple_gpu/custom_op" in str(exc)


class TestPublicNamespace:
    def test_all_codes_exported_from_tessera_compiler(self) -> None:
        import tessera.compiler as tc

        for name in (
            "Diagnostic", "DiagnosticCode", "FallbackReason",
            "JitDiagnosticCode", "TesseraNativeRequiredError",
            "classify_host",
        ):
            assert name in tc.__all__, f"{name} missing from compiler.__all__"
            assert hasattr(tc, name), f"{name} not importable from tessera.compiler"
