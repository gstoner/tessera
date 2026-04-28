"""
test_error_reporter.py — ErrorReporter + exception hierarchy tests (Phase 6)

API notes (from diagnostics.py):
  - TesseraShapeError(message, expected_shape=..., actual_shape=...)
  - TesseraTargetError(message, target=...)
  - ErrorReporter._error_limit  (private; public access via reporter._error_limit)
  - check_dtype(actual, allowed) — reports error if not in list; no raise for empty list
  - get_shape() returns None on miss (uses .get()), not KeyError
  - raise_if_errors() raises TesseraShapeError but without expected/actual stored
  - error_limit stored as _error_limit
"""
from __future__ import annotations

import pytest
from tessera.diagnostics import (
    ErrorReporter,
    DiagnosticLevel,
    TesseraError,
    TesseraShapeError,
    TesseraTargetError,
    TesseraTypeError,
    TesseraNotImplementedError,
    SourceLocation,
)


# ---------------------------------------------------------------------------
# Basic report / error / warning / note
# ---------------------------------------------------------------------------

class TestReport:
    def test_no_errors_initially(self, reporter):
        assert not reporter.has_errors()
        assert reporter.error_count() == 0

    def test_report_error(self, reporter):
        reporter.error("test error message")
        assert reporter.has_errors()
        assert reporter.error_count() == 1

    def test_report_warning(self, reporter):
        reporter.warning("test warning")
        assert not reporter.has_errors()
        assert len(reporter.warnings) == 1

    def test_report_note(self, reporter):
        reporter.note("just a note")
        assert not reporter.has_errors()
        assert len(reporter.diagnostics) == 1

    def test_multiple_errors_counted(self, reporter):
        for i in range(5):
            reporter.error(f"err {i}")
        assert reporter.error_count() == 5

    def test_clear_resets_state(self, reporter):
        reporter.error("oops")
        reporter.clear()
        assert not reporter.has_errors()
        assert reporter.error_count() == 0
        assert len(reporter.diagnostics) == 0

    def test_errors_property_filters(self, reporter):
        reporter.error("e1")
        reporter.warning("w1")
        reporter.note("n1")
        assert len(reporter.errors) == 1
        assert len(reporter.warnings) == 1


# ---------------------------------------------------------------------------
# raise_if_errors
# ---------------------------------------------------------------------------

class TestRaiseIfErrors:
    def test_no_error_does_not_raise(self, reporter):
        reporter.raise_if_errors()  # should not raise

    def test_with_error_raises_tessera_error(self, reporter):
        reporter.error("bad shape")
        with pytest.raises(TesseraError):
            reporter.raise_if_errors()

    def test_raises_shape_error_type(self, reporter):
        """raise_if_errors() always raises TesseraShapeError."""
        reporter.check_shape((4, 8), (4, 9))
        with pytest.raises(TesseraShapeError):
            reporter.raise_if_errors()

    def test_shape_error_message_contains_dimensions(self, reporter):
        reporter.check_shape((4, 8), (4, 9))
        try:
            reporter.raise_if_errors()
        except TesseraShapeError as e:
            assert "4" in str(e) or "8" in str(e) or "9" in str(e)


# ---------------------------------------------------------------------------
# check_shape
# ---------------------------------------------------------------------------

class TestCheckShape:
    def test_matching_shapes_no_error(self, reporter):
        reporter.check_shape((4, 8), (4, 8))
        assert not reporter.has_errors()

    def test_mismatching_shapes_adds_error(self, reporter):
        reporter.check_shape((4, 8), (4, 9))
        assert reporter.has_errors()

    def test_rank_mismatch_is_error(self, reporter):
        reporter.check_shape((4, 8), (4, 8, 1))
        assert reporter.has_errors()

    def test_scalar_shape(self, reporter):
        reporter.check_shape((), ())
        assert not reporter.has_errors()

    def test_check_shape_returns_bool(self, reporter):
        ok = reporter.check_shape((4, 4), (4, 4))
        assert ok is True
        fail = reporter.check_shape((4, 4), (4, 5))
        assert fail is False


# ---------------------------------------------------------------------------
# check_rank
# ---------------------------------------------------------------------------

class TestCheckRank:
    def test_correct_rank(self, reporter):
        reporter.check_rank((2, 3, 4), 3)
        assert not reporter.has_errors()

    def test_wrong_rank(self, reporter):
        reporter.check_rank((2, 3), 3)
        assert reporter.has_errors()

    def test_check_rank_returns_bool(self, reporter):
        assert reporter.check_rank((2, 3), 2) is True
        assert reporter.check_rank((2, 3), 3) is False


# ---------------------------------------------------------------------------
# check_dtype
# ---------------------------------------------------------------------------

class TestCheckDtype:
    def test_allowed_dtype(self, reporter):
        reporter.check_dtype("bf16", ["bf16", "fp16"])
        assert not reporter.has_errors()

    def test_disallowed_dtype(self, reporter):
        reporter.check_dtype("fp64", ["bf16", "fp16", "fp32"])
        assert reporter.has_errors()

    def test_check_dtype_returns_bool(self, reporter):
        assert reporter.check_dtype("bf16", ["bf16"]) is True
        assert reporter.check_dtype("fp64", ["bf16"]) is False

    def test_empty_allowed_list_disallows_all(self, reporter):
        # Empty allowed list — every dtype should be "not allowed"
        reporter.check_dtype("bf16", [])
        assert reporter.has_errors()


# ---------------------------------------------------------------------------
# raise_target_error_if_any
# ---------------------------------------------------------------------------

class TestTargetError:
    def test_no_error_no_raise(self, reporter):
        reporter.raise_target_error_if_any("gfx90a")  # must not raise

    def test_error_raises_target_error(self, reporter):
        reporter.error("unsupported target")
        with pytest.raises(TesseraTargetError) as exc_info:
            reporter.raise_target_error_if_any("gfx99x")
        assert exc_info.value.target == "gfx99x"

    def test_target_error_message_preserved(self, reporter):
        reporter.error("my custom error")
        try:
            reporter.raise_target_error_if_any("sm_90")
        except TesseraTargetError as e:
            assert "my custom error" in str(e)


# ---------------------------------------------------------------------------
# format_all
# ---------------------------------------------------------------------------

class TestFormatAll:
    def test_format_includes_message(self, reporter):
        reporter.error("my special error")
        text = reporter.format_all()
        assert "my special error" in text

    def test_format_includes_levels(self, reporter):
        reporter.error("err")
        reporter.warning("warn")
        reporter.note("note")
        text = reporter.format_all()
        assert "error" in text.lower() or "ERROR" in text
        assert "warn" in text.lower() or "WARNING" in text

    def test_format_empty_is_string(self, reporter):
        text = reporter.format_all()
        assert isinstance(text, str)


# ---------------------------------------------------------------------------
# Source location capture
# ---------------------------------------------------------------------------

class TestSourceLocation:
    def test_python_loc_captured(self):
        r = ErrorReporter(capture_python_location=True)
        r.error("loc test")
        diag = r.diagnostics[0]
        assert diag.location is not None
        assert diag.location.line > 0

    def test_no_loc_capture(self):
        r = ErrorReporter(capture_python_location=False)
        r.error("no loc")
        diag = r.diagnostics[0]
        # With capture disabled, location should be None
        assert diag.location is None

    def test_source_location_fields(self):
        loc = SourceLocation(file="test.py", line=42, column=7)
        assert loc.file == "test.py"
        assert loc.line == 42
        assert loc.column == 7

    def test_source_location_str(self):
        loc = SourceLocation(file="foo.py", line=10, column=5)
        s = str(loc)
        assert "foo.py" in s
        assert "10" in s


# ---------------------------------------------------------------------------
# Error limit
# ---------------------------------------------------------------------------

class TestErrorLimit:
    def test_error_limit_stored(self):
        r = ErrorReporter(error_limit=3)
        assert r._error_limit == 3

    def test_error_limit_caps_accumulation(self):
        r = ErrorReporter(error_limit=3)
        for i in range(10):
            r.error(f"error {i}")
        # After limit, additional errors should be dropped
        assert r.error_count() <= 3

    def test_error_limit_zero_blocks_all_errors(self):
        """error_limit=0 means no errors are accumulated (cap is 0)."""
        r = ErrorReporter(error_limit=0)
        for i in range(10):
            r.error(f"error {i}")
        # All errors blocked since limit is 0
        assert r.error_count() == 0

    def test_error_limit_large_accumulates_many(self):
        r = ErrorReporter(error_limit=100)
        for i in range(50):
            r.error(f"error {i}")
        assert r.error_count() == 50


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class TestExceptionHierarchy:
    def test_shape_error_is_tessera_error(self):
        e = TesseraShapeError("bad shapes", expected_shape=(4,), actual_shape=(5,))
        assert isinstance(e, TesseraError)

    def test_target_error_is_tessera_error(self):
        e = TesseraTargetError("unsupported target", target="gfx90a")
        assert isinstance(e, TesseraError)

    def test_type_error_is_tessera_error(self):
        e = TesseraTypeError("bad type")
        assert isinstance(e, TesseraError)

    def test_not_implemented_is_tessera_error(self):
        e = TesseraNotImplementedError("not yet")
        assert isinstance(e, TesseraError)

    def test_shape_error_stores_shapes(self):
        e = TesseraShapeError("mismatch", expected_shape=(2, 3), actual_shape=(2, 4))
        assert e.expected_shape == (2, 3)
        assert e.actual_shape == (2, 4)

    def test_shape_error_message(self):
        e = TesseraShapeError("dim mismatch", expected_shape=(2, 3), actual_shape=(2, 4))
        msg = str(e)
        assert "dim mismatch" in msg

    def test_target_error_stores_target(self):
        e = TesseraTargetError("no support", target="volta")
        assert e.target == "volta"

    def test_target_error_message(self):
        e = TesseraTargetError("no volta support", target="volta")
        assert "volta" in str(e)

    def test_shape_error_none_shapes(self):
        e = TesseraShapeError("generic shape error")
        assert e.expected_shape is None
        assert e.actual_shape is None
