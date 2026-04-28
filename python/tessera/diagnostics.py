"""
diagnostics.py — Tessera error reporting and diagnostics (Phase 6)

Provides structured error types and an ``ErrorReporter`` that collects
diagnostics (errors, warnings, notes) and can attach Python source locations.
The reporter mirrors ``ErrorReporter.cpp`` at the C++ level but is fully
usable from pure Python.

Usage::

    from tessera.diagnostics import (
        ErrorReporter, TesseraShapeError, TesseraTargetError,
        DiagnosticLevel, SourceLocation,
    )

    reporter = ErrorReporter()
    reporter.error("matmul: dim mismatch — got (128,256) × (512,64)",
                   op_name="tessera.matmul")
    reporter.raise_if_errors()   # raises TesseraShapeError
"""

from __future__ import annotations

import inspect
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence


# ---------------------------------------------------------------------------
# Diagnostic level
# ---------------------------------------------------------------------------

class DiagnosticLevel(Enum):
    NOTE    = "note"
    WARNING = "warning"
    ERROR   = "error"


# ---------------------------------------------------------------------------
# Source location
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceLocation:
    """
    A Python / MLIR source location attached to a diagnostic.

    ``file``, ``line``, ``column`` mirror the MLIR ``FileLineColLoc``.
    ``python_frame`` is the Python stack frame string if captured.
    """

    file: str
    line: int
    column: int = 0
    python_frame: str = ""

    def __str__(self) -> str:
        col = f":{self.column}" if self.column else ""
        return f"{self.file}:{self.line}{col}"

    def __repr__(self) -> str:
        return f"SourceLocation({self.file!r}, {self.line}, {self.column})"


# ---------------------------------------------------------------------------
# Diagnostic record
# ---------------------------------------------------------------------------

@dataclass
class TesseraDiagnostic:
    """
    A single diagnostic message with level, location, and op context.
    """

    level: DiagnosticLevel
    message: str
    location: Optional[SourceLocation] = None
    op_name: str = ""
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        parts = [f"{self.level.value.upper()}"]
        if self.location:
            parts.append(f"[{self.location}]")
        if self.op_name:
            parts.append(f"({self.op_name})")
        parts.append(self.message)
        return " ".join(parts)

    def __repr__(self) -> str:
        return (f"TesseraDiagnostic(level={self.level.value!r}, "
                f"msg={self.message!r})")


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class TesseraError(Exception):
    """
    Base class for all structured Tessera compiler errors.

    Parameters
    ----------
    message : str
        Human-readable error description.
    location : SourceLocation, optional
        Source position (MLIR loc or Python frame).
    op_name : str
        Name of the MLIR op that triggered the error.
    """

    def __init__(
        self,
        message: str,
        location: Optional[SourceLocation] = None,
        op_name: str = "",
    ) -> None:
        super().__init__(message)
        self.location = location
        self.op_name = op_name

    def __str__(self) -> str:
        base = super().__str__()
        parts = [base]
        if self.op_name:
            parts.append(f"  op: {self.op_name}")
        if self.location:
            parts.append(f"  at: {self.location}")
        return "\n".join(parts)


class TesseraShapeError(TesseraError):
    """
    Raised when tensor shapes are incompatible.

    Carries ``expected_shape`` and ``actual_shape`` for diagnostics tooling.
    """

    def __init__(
        self,
        message: str,
        expected_shape: Optional[Sequence[int]] = None,
        actual_shape: Optional[Sequence[int]] = None,
        **kwargs,
    ) -> None:
        super().__init__(message, **kwargs)
        self.expected_shape = tuple(expected_shape) if expected_shape else None
        self.actual_shape = tuple(actual_shape) if actual_shape else None

    def __str__(self) -> str:
        base = super().__str__()
        if self.expected_shape is not None or self.actual_shape is not None:
            base += (
                f"\n  expected: {self.expected_shape}"
                f"\n  actual:   {self.actual_shape}"
            )
        return base


class TesseraTargetError(TesseraError):
    """
    Raised when a lowering or codegen step fails for a specific hardware target.

    ``target`` is a string like ``"sm_90"`` or ``"gfx90a"``.
    """

    def __init__(
        self,
        message: str,
        target: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(message, **kwargs)
        self.target = target

    def __str__(self) -> str:
        base = super().__str__()
        if self.target:
            base += f"\n  target: {self.target}"
        return base


class TesseraTypeError(TesseraError):
    """Raised when dtype constraints are violated."""


class TesseraNotImplementedError(TesseraError, NotImplementedError):
    """Raised for unimplemented lowering paths."""


# ---------------------------------------------------------------------------
# Error reporter
# ---------------------------------------------------------------------------

class ErrorReporter:
    """
    Collects diagnostics from the compiler/runtime and surfaces them as
    structured Python errors.

    Thread-safety: not thread-safe (use one reporter per compilation thread).

    Parameters
    ----------
    capture_python_location : bool
        When True, automatically capture the Python call-site (file + line)
        for each diagnostic if no explicit location is provided.
    error_limit : int
        Stop accumulating after this many errors (prevents runaway output).
    """

    def __init__(
        self,
        *,
        capture_python_location: bool = True,
        error_limit: int = 100,
    ) -> None:
        self._diagnostics: List[TesseraDiagnostic] = []
        self._capture_python_location = capture_python_location
        self._error_limit = error_limit

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def _caller_location(self, depth: int = 2) -> Optional[SourceLocation]:
        """Return the SourceLocation of the caller ``depth`` frames up."""
        if not self._capture_python_location:
            return None
        try:
            frame = inspect.stack()[depth]
            return SourceLocation(
                file=frame.filename,
                line=frame.lineno,
                python_frame=f"{frame.filename}:{frame.lineno} in {frame.function}",
            )
        except (IndexError, AttributeError):
            return None

    def report(
        self,
        level: DiagnosticLevel,
        message: str,
        op_name: str = "",
        location: Optional[SourceLocation] = None,
        notes: Optional[List[str]] = None,
    ) -> TesseraDiagnostic:
        """Record a diagnostic.  Returns the created TesseraDiagnostic."""
        if location is None:
            location = self._caller_location(depth=2)

        diag = TesseraDiagnostic(
            level=level,
            message=message,
            location=location,
            op_name=op_name,
            notes=notes or [],
        )

        error_count = sum(
            1 for d in self._diagnostics
            if d.level == DiagnosticLevel.ERROR
        )
        if level != DiagnosticLevel.ERROR or error_count < self._error_limit:
            self._diagnostics.append(diag)

        return diag

    def error(
        self,
        message: str,
        op_name: str = "",
        location: Optional[SourceLocation] = None,
        expected_shape: Optional[Sequence[int]] = None,
        actual_shape: Optional[Sequence[int]] = None,
    ) -> TesseraDiagnostic:
        """Record an error-level diagnostic."""
        return self.report(DiagnosticLevel.ERROR, message, op_name, location)

    def warning(
        self,
        message: str,
        op_name: str = "",
        location: Optional[SourceLocation] = None,
    ) -> TesseraDiagnostic:
        return self.report(DiagnosticLevel.WARNING, message, op_name, location)

    def note(
        self,
        message: str,
        op_name: str = "",
        location: Optional[SourceLocation] = None,
    ) -> TesseraDiagnostic:
        return self.report(DiagnosticLevel.NOTE, message, op_name, location)

    # ------------------------------------------------------------------
    # Shape-check helpers
    # ------------------------------------------------------------------

    def check_shape(
        self,
        actual: Sequence[int],
        expected: Sequence[int],
        op_name: str = "",
    ) -> bool:
        """
        Verify ``actual`` matches ``expected``.

        Returns True if shapes match; records an error and returns False
        otherwise.
        """
        if tuple(actual) != tuple(expected):
            self.report(
                DiagnosticLevel.ERROR,
                f"shape mismatch: expected {tuple(expected)}, got {tuple(actual)}",
                op_name=op_name,
            )
            return False
        return True

    def check_rank(
        self,
        shape: Sequence[int],
        expected_rank: int,
        op_name: str = "",
    ) -> bool:
        if len(shape) != expected_rank:
            self.report(
                DiagnosticLevel.ERROR,
                f"rank mismatch: expected rank {expected_rank}, "
                f"got rank {len(shape)} (shape={tuple(shape)})",
                op_name=op_name,
            )
            return False
        return True

    def check_dtype(
        self,
        actual_dtype: str,
        allowed_dtypes: Sequence[str],
        op_name: str = "",
    ) -> bool:
        if actual_dtype not in allowed_dtypes:
            self.report(
                DiagnosticLevel.ERROR,
                f"dtype {actual_dtype!r} not allowed; expected one of "
                f"{list(allowed_dtypes)}",
                op_name=op_name,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Result queries
    # ------------------------------------------------------------------

    def has_errors(self) -> bool:
        return any(d.level == DiagnosticLevel.ERROR
                   for d in self._diagnostics)

    def has_warnings(self) -> bool:
        return any(d.level == DiagnosticLevel.WARNING
                   for d in self._diagnostics)

    def error_count(self) -> int:
        return sum(1 for d in self._diagnostics
                   if d.level == DiagnosticLevel.ERROR)

    def warning_count(self) -> int:
        return sum(1 for d in self._diagnostics
                   if d.level == DiagnosticLevel.WARNING)

    # ------------------------------------------------------------------
    # Raising
    # ------------------------------------------------------------------

    def raise_if_errors(self) -> None:
        """Raise ``TesseraShapeError`` with the first error if any."""
        errs = [d for d in self._diagnostics
                if d.level == DiagnosticLevel.ERROR]
        if errs:
            first = errs[0]
            raise TesseraShapeError(
                first.message,
                location=first.location,
                op_name=first.op_name,
            )

    def raise_target_error_if_any(self, target: str = "") -> None:
        """Raise ``TesseraTargetError`` with the first error if any."""
        errs = [d for d in self._diagnostics
                if d.level == DiagnosticLevel.ERROR]
        if errs:
            first = errs[0]
            raise TesseraTargetError(
                first.message,
                target=target,
                location=first.location,
                op_name=first.op_name,
            )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def diagnostics(self) -> List[TesseraDiagnostic]:
        return list(self._diagnostics)

    @property
    def errors(self) -> List[TesseraDiagnostic]:
        return [d for d in self._diagnostics
                if d.level == DiagnosticLevel.ERROR]

    @property
    def warnings(self) -> List[TesseraDiagnostic]:
        return [d for d in self._diagnostics
                if d.level == DiagnosticLevel.WARNING]

    def clear(self) -> None:
        self._diagnostics.clear()

    def format_all(self) -> str:
        return "\n".join(str(d) for d in self._diagnostics)

    def __repr__(self) -> str:
        return (f"ErrorReporter(errors={self.error_count()}, "
                f"warnings={self.warning_count()})")


# ---------------------------------------------------------------------------
# Shape inference engine (Python layer)
# ---------------------------------------------------------------------------

class ShapeInferenceEngine:
    """
    Forward-propagate static shapes through a layer graph.

    Mirrors ``ShapeInferencePass.cpp`` at the Python layer.  Each call to
    ``infer()`` checks the output shape of an op against the shapes that
    would be produced by the standard broadcasting / matmul rules.

    Parameters
    ----------
    reporter : ErrorReporter, optional
        If provided, shape mismatches are reported there instead of raised
        immediately.
    """

    def __init__(
        self,
        reporter: Optional[ErrorReporter] = None,
    ) -> None:
        self._reporter = reporter or ErrorReporter(capture_python_location=False)
        self._shapes: dict = {}

    def set_shape(self, name: str, shape: Sequence[int]) -> None:
        """Register a known shape for a named value."""
        self._shapes[name] = tuple(shape)

    def get_shape(self, name: str) -> Optional[tuple]:
        return self._shapes.get(name)

    # ------------------------------------------------------------------
    # Shape rules
    # ------------------------------------------------------------------

    def infer_matmul(
        self,
        lhs_shape: Sequence[int],
        rhs_shape: Sequence[int],
        op_name: str = "tessera.matmul",
    ) -> Optional[tuple]:
        """
        Infer output shape of a 2-D matmul: (M, K) × (K, N) → (M, N).
        Supports batched: (B, M, K) × (B, K, N) → (B, M, N).
        """
        L, R = tuple(lhs_shape), tuple(rhs_shape)
        if len(L) < 2 or len(R) < 2:
            self._reporter.error(
                f"matmul requires rank >= 2; got {L} × {R}", op_name=op_name
            )
            return None
        if L[-1] != R[-2]:
            self._reporter.error(
                f"matmul K-dim mismatch: lhs[-1]={L[-1]} != rhs[-2]={R[-2]}",
                op_name=op_name,
                expected_shape=None,
                actual_shape=None,
            )
            return None
        batch = L[:-2]
        return batch + (L[-2], R[-1])

    def infer_elementwise(
        self,
        *shapes: Sequence[int],
        op_name: str = "tessera.elementwise",
    ) -> Optional[tuple]:
        """All shapes must be identical for element-wise ops."""
        if not shapes:
            return None
        ref = tuple(shapes[0])
        for s in shapes[1:]:
            if tuple(s) != ref:
                self._reporter.error(
                    f"element-wise shape mismatch: {ref} vs {tuple(s)}",
                    op_name=op_name,
                )
                return None
        return ref

    def infer_flash_attn(
        self,
        q_shape: Sequence[int],
        k_shape: Sequence[int],
        v_shape: Sequence[int],
        op_name: str = "tessera.flash_attn",
    ) -> Optional[tuple]:
        """
        Infer output of flash attention: output has same shape as Q.
        Validates (B, H, S, D) layout and K/V sequence lengths match.
        """
        Q, K, V = tuple(q_shape), tuple(k_shape), tuple(v_shape)
        for name, shape in [("Q", Q), ("K", K), ("V", V)]:
            if len(shape) != 4:
                self._reporter.error(
                    f"flash_attn {name} must be rank-4, got {shape}",
                    op_name=op_name,
                )
                return None
        if K[2] != V[2]:
            self._reporter.error(
                f"flash_attn K/V sequence length mismatch: K_seq={K[2]}, V_seq={V[2]}",
                op_name=op_name,
            )
            return None
        if Q[3] != K[3]:
            self._reporter.error(
                f"flash_attn Q/K head-dim mismatch: Q_d={Q[3]}, K_d={K[3]}",
                op_name=op_name,
            )
            return None
        return Q  # output has same shape as Q

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def infer_graph(self, ops: List[dict]) -> dict:
        """
        Infer shapes for a list of op descriptors.

        Each dict must have:
          "name"   : str — op name
          "op"     : str — op kind ("matmul", "elementwise", "flash_attn")
          "inputs" : list[str] — input value names (must be registered)
          "output" : str — output value name

        Returns a dict mapping output name → inferred shape.
        """
        results = {}
        for op in ops:
            inputs = [self._shapes[n] for n in op.get("inputs", [])]
            kind = op.get("op", "")
            output = op.get("output", "")
            op_name = op.get("name", kind)

            shape = None
            if kind == "matmul" and len(inputs) >= 2:
                shape = self.infer_matmul(inputs[0], inputs[1], op_name)
            elif kind == "elementwise":
                shape = self.infer_elementwise(*inputs, op_name=op_name)
            elif kind == "flash_attn" and len(inputs) >= 3:
                shape = self.infer_flash_attn(
                    inputs[0], inputs[1], inputs[2], op_name
                )

            if shape is not None:
                self._shapes[output] = shape
                results[output] = shape

        return results

    @property
    def reporter(self) -> ErrorReporter:
        return self._reporter
