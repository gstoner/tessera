"""Audit Action 2 — AppleTensorBindingSpec / AppleKernelBindingSpec.

The audit's framing:

  > Emit an Apple tensor-binding spec at the IR layer that runtime
  > drift gates validate against — binding name, index, kind, dtype,
  > rank, dims (possibly wildcarded), direction, residency. The spec
  > is the compiler-side declarative contract; the runtime extract
  > (PK7 ``ArgumentLayout``) is the reflection of what Apple's
  > pipeline actually exposes. The two must round-trip and diff.

These tests pin:

* **Construction invariants** — rank/dims agreement, buffer_index
  non-negative, kind/direction in known sets, dims positive-or-None.
* **Wildcard handling** — wildcard-dims specs propagate as "skip
  the dims check" through PK6; concrete specs pin all four axes.
* **Round-trip with PK7** — spec ↔ ArgumentLayoutEntry round-trips
  cleanly when dims are concrete; raises a precise error when
  wildcards are present.
* **Round-trip with PK6** — ``to_expected_bindings()`` produces a
  list ``validate_bindings`` accepts.
* **JSON round-trip** — ``to_dict``/``from_dict`` survives a full
  JSON serialize/parse without drift.
* **AppleKernelBindingSpec invariants** — buffer_index unique,
  names unique, sorted by buffer_index.
* **BackendKernelEntry integration** — spec attached only when
  status='packaged'; package_path consistency check; rendered into
  ``as_dict``.
* **Live-runtime validation** — on hosts where Metal 4 packaged ML
  is available, the spec validates clean against the bundled Apple
  matrix-multiplication package's reflection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tessera.apple_mlpkg import (
    AppleKernelBindingSpec,
    AppleTensorBindingSpec,
    ArgumentLayout,
    ArgumentLayoutEntry,
    ExpectedBinding,
    compile_mlpackage,
    extract_argument_layout,
    last_error_kind,
    packaged_ml_available,
    packaged_ml_skip_reason,
    validate_bindings,
)
from tessera.compiler.backend_manifest import BackendKernelEntry


_FIXTURES_DIR = (Path(__file__).resolve().parent.parent
                 / "fixtures" / "apple_gpu")


def _find_mtlpackage() -> Path | None:
    if not _FIXTURES_DIR.is_dir():
        return None
    for entry in _FIXTURES_DIR.iterdir():
        if entry.suffix == ".mtlpackage" and entry.is_dir():
            return entry
    return None


# ---- AppleTensorBindingSpec construction invariants --------------------

def test_spec_accepts_well_formed_concrete_binding():
    spec = AppleTensorBindingSpec(
        name="inputA", buffer_index=0, kind="tensor",
        dtype="fp32", rank=2, dims=(4, 4),
        direction="input", residency="shared",
    )
    assert spec.name == "inputA"
    assert spec.has_wildcard_dims is False
    assert spec.concrete_dims() == (4, 4)


def test_spec_accepts_wildcard_dims():
    spec = AppleTensorBindingSpec(
        name="inputA", buffer_index=0, kind="tensor",
        dtype="fp32", rank=2, dims=(None, None),
        direction="input",
    )
    assert spec.has_wildcard_dims is True
    assert spec.concrete_dims() is None


def test_spec_accepts_partial_wildcard():
    """A rank-2 binding where one axis is dynamic + one is fixed."""
    spec = AppleTensorBindingSpec(
        name="inputA", buffer_index=0, kind="tensor",
        dtype="fp32", rank=2, dims=(None, 128),
        direction="input",
    )
    assert spec.has_wildcard_dims is True
    assert spec.concrete_dims() is None


def test_spec_rejects_rank_dims_mismatch():
    with pytest.raises(ValueError, match=r"rank=3 but len\(dims\)=2"):
        AppleTensorBindingSpec(
            name="x", buffer_index=0, kind="tensor",
            dtype="fp32", rank=3, dims=(4, 4),
            direction="input",
        )


def test_spec_rejects_negative_buffer_index():
    with pytest.raises(ValueError, match="buffer_index=-1"):
        AppleTensorBindingSpec(
            name="x", buffer_index=-1, kind="tensor",
            dtype="fp32", rank=1, dims=(4,),
            direction="input",
        )


def test_spec_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind='quaternion'"):
        AppleTensorBindingSpec(
            name="x", buffer_index=0, kind="quaternion",
            dtype="fp32", rank=1, dims=(4,),
            direction="input",
        )


def test_spec_rejects_unknown_direction():
    with pytest.raises(ValueError, match="direction='sideways'"):
        AppleTensorBindingSpec(
            name="x", buffer_index=0, kind="tensor",
            dtype="fp32", rank=1, dims=(4,),
            direction="sideways",
        )


def test_spec_rejects_zero_or_negative_dim():
    with pytest.raises(ValueError, match=r"dims\[0\]=0"):
        AppleTensorBindingSpec(
            name="x", buffer_index=0, kind="tensor",
            dtype="fp32", rank=1, dims=(0,),
            direction="input",
        )


# ---- PK6 ExpectedBinding round-trip ------------------------------------

def test_concrete_spec_yields_full_expected_binding():
    spec = AppleTensorBindingSpec(
        name="inputA", buffer_index=0, kind="tensor",
        dtype="fp32", rank=2, dims=(4, 4),
        direction="input",
    )
    eb = spec.to_expected_binding()
    assert eb == ExpectedBinding(
        name="inputA", rank=2, dtype="fp32",
        buffer_index=0, dims=(4, 4))


def test_wildcard_spec_drops_dims_field_from_expected_binding():
    """Wildcard specs preserve buffer_index / dtype / rank as strict
    checks but skip the ``dims`` axis. This is the whole point of
    wildcards: the compiler can declare the contract before the
    runtime knows the concrete shape."""
    spec = AppleTensorBindingSpec(
        name="inputA", buffer_index=0, kind="tensor",
        dtype="fp32", rank=2, dims=(None, None),
        direction="input",
    )
    eb = spec.to_expected_binding()
    assert eb.dims is None         # the wildcard
    assert eb.rank == 2            # still pinned
    assert eb.dtype == "fp32"      # still pinned
    assert eb.buffer_index == 0    # still pinned


# ---- PK7 ArgumentLayoutEntry round-trip --------------------------------

def test_concrete_spec_round_trips_through_argument_layout_entry():
    spec = AppleTensorBindingSpec(
        name="inputA", buffer_index=2, kind="tensor",
        dtype="fp16", rank=3, dims=(2, 4, 8),
        direction="output", residency="shared",
    )
    entry = spec.to_argument_layout_entry()
    back = AppleTensorBindingSpec.from_argument_layout_entry(entry)
    assert back == spec


def test_wildcard_spec_cannot_be_converted_to_argument_layout_entry():
    """The runtime form needs concrete shapes — wildcards are a
    compiler-only construct."""
    spec = AppleTensorBindingSpec(
        name="x", buffer_index=0, kind="tensor",
        dtype="fp32", rank=1, dims=(None,),
        direction="input",
    )
    with pytest.raises(ValueError, match="cannot convert"):
        spec.to_argument_layout_entry()


# ---- JSON round-trip ---------------------------------------------------

def test_spec_to_dict_is_json_serializable():
    spec = AppleTensorBindingSpec(
        name="output", buffer_index=2, kind="tensor",
        dtype="fp32", rank=2, dims=(None, 128),
        direction="output",
    )
    blob = json.dumps(spec.to_dict())
    parsed = json.loads(blob)
    # null survives JSON round-trip as None.
    assert parsed["dims"] == [None, 128]
    back = AppleTensorBindingSpec.from_dict(parsed)
    assert back == spec


# ---- AppleKernelBindingSpec composition --------------------------------

def test_kernel_spec_sorts_entries_by_buffer_index():
    """A caller can pass entries in any order; the spec sorts by
    buffer_index at construction so round-trips and diffs are
    deterministic."""
    a = AppleTensorBindingSpec(
        name="output", buffer_index=2, kind="tensor",
        dtype="fp32", rank=2, dims=(4, 4), direction="output")
    b = AppleTensorBindingSpec(
        name="inputA", buffer_index=0, kind="tensor",
        dtype="fp32", rank=2, dims=(4, 4), direction="input")
    c = AppleTensorBindingSpec(
        name="inputB", buffer_index=1, kind="tensor",
        dtype="fp32", rank=2, dims=(4, 4), direction="input")
    ks = AppleKernelBindingSpec(
        function_name="main", package_path="foo.mtlpackage",
        entries=(a, b, c))
    indices = [e.buffer_index for e in ks.entries]
    assert indices == [0, 1, 2]


def test_kernel_spec_rejects_duplicate_buffer_indices():
    a = AppleTensorBindingSpec(
        name="x", buffer_index=0, kind="tensor",
        dtype="fp32", rank=1, dims=(4,), direction="input")
    b = AppleTensorBindingSpec(
        name="y", buffer_index=0, kind="tensor",
        dtype="fp32", rank=1, dims=(4,), direction="input")
    with pytest.raises(ValueError, match="buffer_index 0 reused"):
        AppleKernelBindingSpec(
            function_name="main", package_path="foo.mtlpackage",
            entries=(a, b))


def test_kernel_spec_rejects_duplicate_names():
    a = AppleTensorBindingSpec(
        name="x", buffer_index=0, kind="tensor",
        dtype="fp32", rank=1, dims=(4,), direction="input")
    b = AppleTensorBindingSpec(
        name="x", buffer_index=1, kind="tensor",
        dtype="fp32", rank=1, dims=(4,), direction="input")
    with pytest.raises(ValueError, match="binding name 'x' duplicated"):
        AppleKernelBindingSpec(
            function_name="main", package_path="foo.mtlpackage",
            entries=(a, b))


def test_kernel_spec_inputs_outputs_split():
    a = AppleTensorBindingSpec(
        name="inputA", buffer_index=0, kind="tensor",
        dtype="fp32", rank=2, dims=(4, 4), direction="input")
    b = AppleTensorBindingSpec(
        name="output", buffer_index=1, kind="tensor",
        dtype="fp32", rank=2, dims=(4, 4), direction="output")
    ks = AppleKernelBindingSpec(
        function_name="main", package_path="foo.mtlpackage",
        entries=(a, b))
    assert tuple(e.name for e in ks.inputs()) == ("inputA",)
    assert tuple(e.name for e in ks.outputs()) == ("output",)


def test_kernel_spec_to_argument_layout_round_trips():
    a = AppleTensorBindingSpec(
        name="inputA", buffer_index=0, kind="tensor",
        dtype="fp32", rank=2, dims=(4, 4), direction="input")
    b = AppleTensorBindingSpec(
        name="output", buffer_index=1, kind="tensor",
        dtype="fp32", rank=2, dims=(4, 4), direction="output")
    ks = AppleKernelBindingSpec(
        function_name="main", package_path="foo.mtlpackage",
        entries=(a, b))
    layout = ks.to_argument_layout()
    assert isinstance(layout, ArgumentLayout)
    back = AppleKernelBindingSpec.from_argument_layout(layout)
    assert back == ks


def test_kernel_spec_to_argument_layout_fails_on_wildcards():
    a = AppleTensorBindingSpec(
        name="inputA", buffer_index=0, kind="tensor",
        dtype="fp32", rank=2, dims=(None, 4), direction="input")
    ks = AppleKernelBindingSpec(
        function_name="main", package_path="foo.mtlpackage",
        entries=(a,))
    assert ks.has_wildcard_dims
    with pytest.raises(ValueError, match="wildcard"):
        ks.to_argument_layout()


def test_kernel_spec_json_round_trip():
    a = AppleTensorBindingSpec(
        name="inputA", buffer_index=0, kind="tensor",
        dtype="fp32", rank=2, dims=(None, 128), direction="input")
    ks = AppleKernelBindingSpec(
        function_name="main", package_path="foo.mtlpackage",
        entries=(a,))
    blob = json.dumps(ks.to_dict())
    back = AppleKernelBindingSpec.from_dict(json.loads(blob))
    assert back == ks


# ---- BackendKernelEntry integration ------------------------------------

def test_manifest_entry_accepts_apple_binding_spec_for_packaged_status():
    spec = AppleKernelBindingSpec(
        function_name="main", package_path="kernels/matmul.mtlpackage",
        entries=(
            AppleTensorBindingSpec(
                name="inputA", buffer_index=0, kind="tensor",
                dtype="fp32", rank=2, dims=(4, 4), direction="input"),
        ))
    entry = BackendKernelEntry(
        target="apple_gpu",
        status="packaged",
        dtypes=("fp32",),
        feature_flags=(),
        notes="audit Action 2 smoke",
        packaged_pipeline_path="kernels/matmul.mtlpackage",
        apple_binding_spec=spec,
    )
    assert entry.apple_binding_spec is spec
    rendered = entry.as_dict()
    assert "apple_binding_spec" in rendered
    assert (rendered["apple_binding_spec"]["function_name"]  # type: ignore[index]
            == "main")


def test_manifest_entry_rejects_apple_binding_spec_when_not_packaged():
    spec = AppleKernelBindingSpec(
        function_name="main", package_path="foo.mtlpackage",
        entries=())
    with pytest.raises(ValueError, match="only valid when status='packaged'"):
        BackendKernelEntry(
            target="apple_gpu",
            status="reference",
            dtypes=("fp32",),
            feature_flags=(),
            notes="x",
            apple_binding_spec=spec,
        )


def test_manifest_entry_rejects_non_spec_object():
    with pytest.raises(TypeError, match="must be an AppleKernelBindingSpec"):
        BackendKernelEntry(
            target="apple_gpu",
            status="packaged",
            dtypes=("fp32",),
            feature_flags=(),
            notes="x",
            packaged_pipeline_path="foo.mtlpackage",
            apple_binding_spec="not a spec",  # type: ignore[arg-type]
        )


def test_manifest_entry_rejects_path_mismatch():
    """The spec's package_path must match the manifest entry's
    packaged_pipeline_path — drift here would silently route the
    runtime to a different artifact than what the compiler emitted
    the spec for."""
    spec = AppleKernelBindingSpec(
        function_name="main", package_path="A.mtlpackage", entries=())
    with pytest.raises(ValueError, match="does not match"):
        BackendKernelEntry(
            target="apple_gpu",
            status="packaged",
            dtypes=("fp32",),
            feature_flags=(),
            notes="x",
            packaged_pipeline_path="B.mtlpackage",
            apple_binding_spec=spec,
        )


def test_manifest_entry_omits_apple_binding_spec_in_as_dict_when_absent():
    entry = BackendKernelEntry(
        target="apple_gpu",
        status="packaged",
        dtypes=("fp32",),
        feature_flags=(),
        notes="x",
        packaged_pipeline_path="foo.mtlpackage",
    )
    assert "apple_binding_spec" not in entry.as_dict()


# ---- Live runtime validation -------------------------------------------

def test_kernel_spec_validates_against_real_apple_matmul_package():
    """End-to-end Action 2: declare the compiler-side spec for Apple's
    matrix-multiplication sample, then validate it against the live
    runtime reflection. This is the audit's headline contract — the
    compiler EMITS a spec, the runtime LOADS the same package, and
    the diff is clean."""
    if not packaged_ml_available():
        pytest.skip(packaged_ml_skip_reason() or "packaged ML unavailable")
    pkg = _find_mtlpackage()
    if pkg is None:
        pytest.skip(f"no .mtlpackage fixture in {_FIXTURES_DIR}")

    # Spec written by hand as if from the compiler emitter — the
    # 4x4 case Apple's sample uses.
    spec = AppleKernelBindingSpec(
        function_name="main", package_path=str(pkg),
        entries=(
            AppleTensorBindingSpec(
                name="inputA", buffer_index=0, kind="tensor",
                dtype="fp32", rank=2, dims=(4, 4),
                direction="input"),
            AppleTensorBindingSpec(
                name="inputB", buffer_index=1, kind="tensor",
                dtype="fp32", rank=2, dims=(4, 4),
                direction="input"),
            AppleTensorBindingSpec(
                name="output", buffer_index=2, kind="tensor",
                dtype="fp32", rank=2, dims=(4, 4),
                direction="output"),
        ))

    pipe = compile_mlpackage(pkg, function_name="main",
                             input_dimensions={0: (4, 4), 1: (4, 4)})
    if pipe is None:
        pytest.fail(f"compile_mlpackage failed; "
                    f"last_error_kind={last_error_kind()}")
    try:
        result = spec.validate_against(pipe, strict_extra=True)
        assert result.ok is True, result
        # And derived ExpectedBindings agree directly.
        expected = spec.to_expected_bindings()
        result2 = validate_bindings(pipe, expected, strict_extra=True)
        assert result2.ok is True, result2
    finally:
        pipe.destroy()


def test_wildcard_spec_validates_at_runtime_skipping_dims():
    """A compiler that doesn't know the concrete dims at IR-emit time
    can still produce a valid spec — the runtime drift gate falls
    back to checking rank/dtype/buffer_index."""
    if not packaged_ml_available():
        pytest.skip(packaged_ml_skip_reason() or "packaged ML unavailable")
    pkg = _find_mtlpackage()
    if pkg is None:
        pytest.skip(f"no .mtlpackage fixture in {_FIXTURES_DIR}")

    spec = AppleKernelBindingSpec(
        function_name="main", package_path=str(pkg),
        entries=(
            AppleTensorBindingSpec(
                name="inputA", buffer_index=0, kind="tensor",
                dtype="fp32", rank=2, dims=(None, None),  # wildcards
                direction="input"),
            AppleTensorBindingSpec(
                name="inputB", buffer_index=1, kind="tensor",
                dtype="fp32", rank=2, dims=(None, None),
                direction="input"),
            AppleTensorBindingSpec(
                name="output", buffer_index=2, kind="tensor",
                dtype="fp32", rank=2, dims=(None, None),
                direction="output"),
        ))
    assert spec.has_wildcard_dims

    pipe = compile_mlpackage(pkg, function_name="main",
                             input_dimensions={0: (8, 8), 1: (8, 8)})
    if pipe is None:
        pytest.fail(f"compile failed; last_error_kind={last_error_kind()}")
    try:
        result = spec.validate_against(pipe, strict_extra=True)
        assert result.ok is True, result
    finally:
        pipe.destroy()


def test_kernel_spec_from_runtime_layout_round_trips():
    """The compiler can bootstrap a spec FROM a known-good runtime
    extract — useful for golden-file tests."""
    if not packaged_ml_available():
        pytest.skip(packaged_ml_skip_reason() or "packaged ML unavailable")
    pkg = _find_mtlpackage()
    if pkg is None:
        pytest.skip(f"no .mtlpackage fixture in {_FIXTURES_DIR}")
    pipe = compile_mlpackage(pkg, function_name="main",
                             input_dimensions={0: (4, 4), 1: (4, 4)})
    if pipe is None:
        pytest.fail(f"compile failed; last_error_kind={last_error_kind()}")
    try:
        layout = extract_argument_layout(pipe)
        spec = AppleKernelBindingSpec.from_argument_layout(layout)
        # Round-trip back.
        layout2 = spec.to_argument_layout()
        assert layout2 == layout
        # And validates clean against the same pipeline.
        result = spec.validate_against(pipe, strict_extra=True)
        assert result.ok is True, result
    finally:
        pipe.destroy()
