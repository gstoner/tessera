"""PK7 — ArgumentLayout artifact (audit Action 5).

Audit framing:

  > Emit a first-class ArgumentLayout artifact beside backend IR:
  > binding name, index, resource kind, tensor/buffer type, dtype,
  > rank, residency requirement. Replaces hand-counted runtime
  > indices with compiler-emitted layout contract.

PK7 ships:

* ``ArgumentLayoutEntry`` — one row per binding (name, buffer_index,
  kind, dtype, rank, dims, direction, residency).
* ``ArgumentLayout`` — collection of entries + pipeline metadata.
  Carries enough information for the audit dashboard, for PK6
  drift-validation (round-trips through ``ExpectedBinding``), and
  for compiler-side artifact emission.
* ``extract_argument_layout(pipeline)`` — builds the layout from a
  compiled pipeline's reflection.

These tests use the bundled Apple matrix-multiplication package
(inputA / inputB / output, 2D fp32) as the proving ground.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera._apple_gpu_dispatch import apple_gpu_runtime
from tessera.apple_mlpkg import (
    ArgumentLayout,
    ArgumentLayoutEntry,
    ExpectedBinding,
    Pipeline,
    compile_mlpackage,
    extract_argument_layout,
    last_error_kind,
    validate_bindings,
)


_FIXTURES_DIR = (Path(__file__).resolve().parent.parent
                 / "fixtures" / "apple_gpu")


def _find_mtlpackage() -> Path | None:
    if not _FIXTURES_DIR.is_dir():
        return None
    for entry in _FIXTURES_DIR.iterdir():
        if entry.suffix == ".mtlpackage" and entry.is_dir():
            return entry
    return None


def _open_matmul_pipe_or_skip() -> Pipeline:
    if apple_gpu_runtime() is None:
        pytest.skip("Apple GPU runtime not buildable on this host")
    pkg = _find_mtlpackage()
    if pkg is None:
        pytest.skip(f"no .mtlpackage fixture in {_FIXTURES_DIR}")
    pipe = compile_mlpackage(
        pkg, function_name="main",
        input_dimensions={0: (4, 4), 1: (4, 4)})
    if pipe is None:
        pytest.fail(f"compile_mlpackage failed; last_error_kind="
                    f"{last_error_kind()}")
    return pipe


# ---- Extraction roundtrip ----------------------------------------------

def test_extract_returns_layout_with_pipeline_metadata():
    pipe = _open_matmul_pipe_or_skip()
    try:
        layout = extract_argument_layout(pipe)
        assert isinstance(layout, ArgumentLayout)
        assert layout.function_name == "main"
        assert layout.pipeline_path.endswith(".mtlpackage")
        # Apple's sample package has three bindings.
        assert len(layout.entries) == 3
    finally:
        pipe.destroy()


def test_entries_are_sorted_by_buffer_index():
    """Sorting by buffer_index makes diffs in audit dashboards
    deterministic. Catches a regression where the entries reorder."""
    pipe = _open_matmul_pipe_or_skip()
    try:
        layout = extract_argument_layout(pipe)
        indices = [e.buffer_index for e in layout.entries]
        assert indices == sorted(indices)
        # The bundled package's indices are 0, 1, 2 (no gaps).
        assert indices == [0, 1, 2]
    finally:
        pipe.destroy()


def test_per_entry_fields_match_reflection():
    pipe = _open_matmul_pipe_or_skip()
    try:
        layout = extract_argument_layout(pipe)
        by_name = layout.by_name()
        # inputA / inputB / output should all be present.
        assert set(by_name.keys()) == {"inputA", "inputB", "output"}
        for entry in layout.entries:
            assert isinstance(entry, ArgumentLayoutEntry)
            assert entry.kind == "tensor"        # only kind today
            assert entry.dtype == "fp32"
            assert entry.rank == 2
            assert entry.dims == (4, 4)
            assert entry.residency == "shared"
    finally:
        pipe.destroy()


# ---- Direction inference ----------------------------------------------

def test_direction_inferred_from_name_prefix():
    """``input*`` → ``"input"``, ``output*`` → ``"output"``, anything
    else → ``"unknown"``. Heuristic — caller can override post-extract
    if the package uses a different naming convention."""
    pipe = _open_matmul_pipe_or_skip()
    try:
        layout = extract_argument_layout(pipe)
        by_name = layout.by_name()
        assert by_name["inputA"].direction == "input"
        assert by_name["inputB"].direction == "input"
        assert by_name["output"].direction == "output"
    finally:
        pipe.destroy()


def test_inputs_and_outputs_helpers():
    pipe = _open_matmul_pipe_or_skip()
    try:
        layout = extract_argument_layout(pipe)
        inputs = layout.inputs()
        outputs = layout.outputs()
        assert len(inputs) == 2
        assert {e.name for e in inputs} == {"inputA", "inputB"}
        assert len(outputs) == 1
        assert outputs[0].name == "output"
    finally:
        pipe.destroy()


# ---- Round-trip through PK6 validation ---------------------------------

def test_layout_to_expected_bindings_self_validates():
    """The whole point of Action 5 + Action 4 composing: a layout
    EMITTED at compile time should validate cleanly against a layout
    EXTRACTED at runtime (same source-of-truth: reflection).

    Generate the ExpectedBinding list from the layout, then run
    ``validate_bindings`` against the same pipeline — must report
    ok=True with no diffs."""
    pipe = _open_matmul_pipe_or_skip()
    try:
        layout = extract_argument_layout(pipe)
        expected = layout.to_expected_bindings()
        # The expected list must cover all bindings.
        assert len(expected) == len(layout.entries)
        # And validation against the same pipeline must pass clean.
        result = validate_bindings(pipe, expected, strict_extra=True)
        assert result.ok is True, result
        assert result.missing == ()
        assert result.extra == ()
        assert result.mismatched == ()
    finally:
        pipe.destroy()


def test_layout_validates_strictly_against_drift():
    """If the layout is from build N and the pipeline is from build
    N+1 with a re-ordered binding, the strict validation catches it."""
    pipe = _open_matmul_pipe_or_skip()
    try:
        layout = extract_argument_layout(pipe)
        # Synthesize a "drifted" layout: swap inputA's buffer_index.
        drifted_entries = tuple(
            ArgumentLayoutEntry(
                name=e.name,
                buffer_index=(99 if e.name == "inputA" else e.buffer_index),
                kind=e.kind, dtype=e.dtype, rank=e.rank,
                dims=e.dims, direction=e.direction,
                residency=e.residency,
            )
            for e in layout.entries
        )
        drifted = ArgumentLayout(
            pipeline_path=layout.pipeline_path,
            function_name=layout.function_name,
            entries=drifted_entries,
        )
        # Validate the pipeline against the drifted-layout expectation.
        result = validate_bindings(pipe, drifted.to_expected_bindings())
        assert result.ok is False
        m = [x for x in result.mismatched
             if x.name == "inputA" and x.field == "buffer_index"]
        assert len(m) == 1
        assert m[0].expected == 99
        assert m[0].actual == 0
    finally:
        pipe.destroy()


# ---- to_dict serialization -----------------------------------------------

def test_to_dict_round_trip_has_expected_shape():
    """The JSON-friendly representation an audit dashboard / artifact
    sink consumes. Locks the schema so a downstream consumer doesn't
    drift unnoticed."""
    pipe = _open_matmul_pipe_or_skip()
    try:
        layout = extract_argument_layout(pipe)
        d = layout.to_dict()
        # Top-level keys.
        for k in ("pipeline_path", "function_name", "entries"):
            assert k in d
        assert isinstance(d["entries"], list)
        assert len(d["entries"]) == 3
        # Per-entry keys.
        for entry in d["entries"]:
            for k in ("name", "buffer_index", "kind", "dtype",
                      "rank", "dims", "direction", "residency"):
                assert k in entry, f"entry missing {k!r}"
            # ``dims`` must be a list (JSON-friendly), not a tuple.
            assert isinstance(entry["dims"], list)
    finally:
        pipe.destroy()


def test_to_dict_is_json_serializable():
    """No special types in the dict — straight to json.dumps."""
    import json
    pipe = _open_matmul_pipe_or_skip()
    try:
        layout = extract_argument_layout(pipe)
        serialized = json.dumps(layout.to_dict(), sort_keys=True)
        # Round-trip parses cleanly.
        parsed = json.loads(serialized)
        assert parsed["function_name"] == "main"
        assert len(parsed["entries"]) == 3
    finally:
        pipe.destroy()


# ---- Lifecycle contract -------------------------------------------------

def test_extract_on_destroyed_pipeline_raises():
    """The whole class follows the PK3+ lifecycle contract: calls on
    a destroyed handle raise rather than silently returning empty."""
    pipe = Pipeline(handle=0, package_path="<test>", function_name="main")
    import pytest as _p
    with _p.raises(RuntimeError, match="already destroyed"):
        extract_argument_layout(pipe)


# ---- Helper coverage ----------------------------------------------------

def test_direction_unknown_for_arbitrary_names():
    """The direction inference is a best-effort heuristic; names
    without "input" / "output" return ``"unknown"`` so the caller
    knows the answer isn't authoritative."""
    from tessera.apple_mlpkg import _infer_direction
    assert _infer_direction("inputA") == "input"
    assert _infer_direction("input_logits") == "input"
    assert _infer_direction("output") == "output"
    assert _infer_direction("Output_Probs") == "output"
    assert _infer_direction("weights") == "unknown"
    assert _infer_direction("bias_0") == "unknown"
    assert _infer_direction("") == "unknown"
