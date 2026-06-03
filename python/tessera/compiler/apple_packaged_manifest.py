"""Populated ``BackendKernelEntry(status="packaged")`` rows.

PK5 follow-on (2026-06-01) — the original PK5 sprint shipped the
data-model scaffolding (``status="packaged"`` + ``packaged_pipeline_path``
+ ``__post_init__`` validation, now joined by Action 2's
``apple_binding_spec``). This module moves it to the next rung: actual
populated entries, drift-gated against the artifacts they reference.

**Status honesty.** Today the only ``.mtlpackage`` checked into the
repo is Apple's bundled sample matrix-multiplication package at
``tests/fixtures/apple_gpu/matrix-multiplication.mtlpackage`` (covered
by Apple's MIT-style sample-code license, attribution preserved).
That's a *test fixture*, not a Tessera-authored production kernel.

Listing it as a production ``BackendKernelEntry`` in
``primitive_coverage.py`` would lie about backend coverage in the
dashboards. So we keep it in a separate ``PACKAGED_TEST_FIXTURES``
table here:

* Real packaged-ML lifecycle (load → reflect → bind → dispatch) is
  proven end-to-end against this entry's artifact, which is what
  the audit's Action 6 / PK5 follow-on actually asks for.
* The standalone-primitive-coverage dashboard stays accurate —
  populating the registry waits for kernels Tessera actually owns.

When Tessera-authored packaged kernels land, the recipe is:

1. Drop the ``.mtlpackage`` somewhere reachable from the repo
   (typically ``runtime_artifacts/apple_gpu/<op_name>.mtlpackage``).
2. Append a ``BackendKernelEntry`` to ``PACKAGED_PRODUCTION_KERNELS``
   below with ``status="packaged"`` + ``packaged_pipeline_path`` +
   ``apple_binding_spec=<AppleKernelBindingSpec>``.
3. Add the entry to the ``OP_SPECS["<op_name>"].metadata["backend_kernel_manifest"]``
   list in ``primitive_coverage.py`` so the dashboard reflects it.
4. The drift gate (``validate_packaged_entry``) will catch path
   typos / binding-spec mismatches at test time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .backend_manifest import BackendKernelEntry
from ..apple_mlpkg import (
    AppleKernelBindingSpec,
    AppleTensorBindingSpec,
)


# ---------------------------------------------------------------------
# Repository-relative paths.
#
# We resolve at import time so the table values are immediately usable;
# the drift gate re-resolves at test time to catch a moved fixture.
# ---------------------------------------------------------------------

def repo_root() -> Path:
    """Return the Tessera repo root (3 levels up from this module:
    python/tessera/compiler/apple_packaged_manifest.py → repo)."""
    return Path(__file__).resolve().parent.parent.parent.parent


_APPLE_MATMUL_FIXTURE_REL = (
    "tests/fixtures/apple_gpu/matrix-multiplication.mtlpackage")


# ---------------------------------------------------------------------
# Apple sample matmul — the only checked-in .mtlpackage today.
# Used to prove PK1–PK7 end-to-end + to lock the manifest-plumbing
# contract. NOT a Tessera-authored kernel; status="packaged" + the
# test-fixture flag in feature_flags makes the role explicit.
# ---------------------------------------------------------------------

_APPLE_MATMUL_BINDING_SPEC = AppleKernelBindingSpec(
    function_name="main",
    package_path=_APPLE_MATMUL_FIXTURE_REL,
    entries=(
        AppleTensorBindingSpec(
            name="inputA", buffer_index=0, kind="tensor",
            dtype="fp32", rank=2, dims=(4, 4), direction="input"),
        AppleTensorBindingSpec(
            name="inputB", buffer_index=1, kind="tensor",
            dtype="fp32", rank=2, dims=(4, 4), direction="input"),
        AppleTensorBindingSpec(
            name="output", buffer_index=2, kind="tensor",
            dtype="fp32", rank=2, dims=(4, 4), direction="output"),
    ),
)


_APPLE_MATMUL_FIXTURE_ENTRY = BackendKernelEntry(
    target="apple_gpu",
    status="packaged",
    dtypes=("fp32",),
    feature_flags=("test_fixture", "apple_sample"),
    notes=(
        "Apple sample matrix-multiplication.mtlpackage bundled at "
        "tests/fixtures/apple_gpu/. Proves the PK1–PK7 packaged-ML "
        "lifecycle (load → reflect → bind → dispatch → ABI-validate) "
        "end-to-end. NOT a Tessera-authored production kernel — see "
        "tests/fixtures/apple_gpu/APPLE_SAMPLE_LICENSE.txt for "
        "provenance. The 'test_fixture' feature_flag keeps this row "
        "out of any dashboard that counts production-ready kernels."
    ),
    packaged_pipeline_path=_APPLE_MATMUL_FIXTURE_REL,
    apple_binding_spec=_APPLE_MATMUL_BINDING_SPEC,
)


#: Populated test-fixture-backed packaged entries. Always at least
#: one entry (the bundled Apple matmul). Keyed by a short fixture name
#: callers can look up by.
PACKAGED_TEST_FIXTURES: dict[str, BackendKernelEntry] = {
    "apple_matmul_4x4_fp32": _APPLE_MATMUL_FIXTURE_ENTRY,
}


# ---------------------------------------------------------------------
# Tessera-authored PRODUCTION packaged kernels (PK8c, 2026-06-02).
#
# These are authored by Tessera itself from the MPSGraph lane (PK8 —
# ``apple_mlpkg.author_matmul_package`` / ``author_chain_package``),
# serialized via ``MPSGraphExecutable serializeToMPSGraphPackageAtURL:``,
# and committed under tests/fixtures/apple_gpu/. Regenerate with
# ``python3 scripts/author_apple_packages.py``.
#
# Binding-spec note: MPSGraph-serialized packages expose *positional,
# unnamed* bindings (``MPSGraphTensor`` has no name property — see PK8).
# The name-based ``AppleKernelBindingSpec`` therefore does NOT apply; these
# rows set ``apple_binding_spec=None`` and rely on positional addressing
# (``fill_input_at`` / ``read_output_at``). ``validate_packaged_entry``
# still gates path existence + .mtlpackage structure; numerical dispatch is
# proven in ``tests/unit/test_apple_mlpkg_pk8c.py``.
#
# Portability caveat: a serialized MPSGraph package targets the current
# platform's latest deployment version and may not load on an *older*
# macOS; the validating test treats a load failure as a skip, not a
# hard failure.
# ---------------------------------------------------------------------

# Production fixtures live in a ``production/`` subdir so the PK1-PK7
# lifecycle tests (which ``iterdir``-scan the top-level fixtures dir and
# compile whatever ``.mtlpackage`` files they find) only ever pick up Apple's
# bundled sample — adding Tessera-authored packages here must not change which
# package those tests exercise, nor push the per-process MTL4-ML-compile count
# past the Metal abort ceiling.
_TESSERA_MATMUL_REL = (
    "tests/fixtures/apple_gpu/production/"
    "tessera_authored_matmul_8x8x8.mtlpackage")
_TESSERA_MATMUL_SOFTMAX_REL = (
    "tests/fixtures/apple_gpu/production/"
    "tessera_authored_matmul_softmax_4x6x5.mtlpackage")


_TESSERA_MATMUL_ENTRY = BackendKernelEntry(
    target="apple_gpu",
    status="packaged",
    dtypes=("fp32",),
    feature_flags=("tessera_authored", "mpsgraph_lane"),
    notes=(
        "Tessera-authored matmul 8x8x8 .mtlpackage — built from the "
        "MPSGraph lane via apple_mlpkg.author_matmul_package, serialized "
        "with MPSGraphExecutable serializeToMPSGraphPackageAtURL:. "
        "Positional (unnamed) bindings: inputs at 0/1, output at 2. "
        "Regenerate via scripts/author_apple_packages.py. Numerically "
        "validated vs numpy in test_apple_mlpkg_pk8c.py."
    ),
    packaged_pipeline_path=_TESSERA_MATMUL_REL,
    apple_binding_spec=None,  # MPSGraph packages are positionally bound
)

_TESSERA_MATMUL_SOFTMAX_ENTRY = BackendKernelEntry(
    target="apple_gpu",
    status="packaged",
    dtypes=("fp32",),
    feature_flags=("tessera_authored", "mpsgraph_lane", "fused_chain"),
    notes=(
        "Tessera-authored fused softmax(A@B) .mtlpackage (4x6x5) — built "
        "from the MPSGraph lane via apple_mlpkg.author_chain_package "
        "('matmul_softmax'). One serialized executable = one GPU dispatch "
        "for the whole chain. Positional bindings: A=0, B=1, output=2. "
        "Regenerate via scripts/author_apple_packages.py."
    ),
    packaged_pipeline_path=_TESSERA_MATMUL_SOFTMAX_REL,
    apple_binding_spec=None,
)


# Expanded coverage (2026-06-02) — one representative per authorable family
# beyond matmul/matmul_softmax: a unary op (silu), a row op (softmax), a
# weighted norm (rmsnorm), and the remaining two fused chains. All
# Tessera-authored from the MPSGraph lane, positionally bound, regenerable via
# scripts/author_apple_packages.py, numerically validated in
# test_apple_mlpkg_pk8c.py.
def _authored_row(rel: str, notes: str,
                  *flags: str) -> BackendKernelEntry:
    return BackendKernelEntry(
        target="apple_gpu",
        status="packaged",
        dtypes=("fp32",),
        feature_flags=("tessera_authored", "mpsgraph_lane", *flags),
        notes=notes,
        packaged_pipeline_path=rel,
        apple_binding_spec=None,  # MPSGraph packages are positionally bound
    )


_FIX = "tests/fixtures/apple_gpu/production/"

_TESSERA_SILU_ENTRY = _authored_row(
    _FIX + "tessera_authored_silu_8x16.mtlpackage",
    "Tessera-authored silu(x) .mtlpackage (8x16) — unary MPSGraph-lane op. "
    "Positional binding: input 0, output 1.")

_TESSERA_SOFTMAX_ENTRY = _authored_row(
    _FIX + "tessera_authored_softmax_8x16.mtlpackage",
    "Tessera-authored softmax(x) .mtlpackage (8x16) — row op over the last "
    "axis. Positional binding: input 0, output 1.")

_TESSERA_RMSNORM_ENTRY = _authored_row(
    _FIX + "tessera_authored_rmsnorm_8x16.mtlpackage",
    "Tessera-authored weighted rmsnorm(x)*gamma .mtlpackage (8x16) — norm "
    "MPSGraph-lane op with a gamma input. Positional bindings: x=0, gamma=1, "
    "output=2.",
    "weighted_norm")

_TESSERA_ATTN_BLOCK_ENTRY = _authored_row(
    _FIX + "tessera_authored_matmul_softmax_matmul_4x6x5x3.mtlpackage",
    "Tessera-authored fused softmax(A@B)@C .mtlpackage (4x6x5x3) — the 3-op "
    "attention block as one serialized executable. Positional bindings: "
    "A=0, B=1, C=2, output=3.",
    "fused_chain")

_TESSERA_RMSNORM_MATMUL_ENTRY = _authored_row(
    _FIX + "tessera_authored_rmsnorm_matmul_4x6x5.mtlpackage",
    "Tessera-authored fused (rmsnorm(x)*gamma)@W .mtlpackage (4x6x5) — a "
    "norm+projection block in one dispatch. Positional bindings: x=0, "
    "gamma=1, W=2, output=3.",
    "fused_chain")


#: Populated PRODUCTION packaged-kernel entries — Tessera-authored
#: ``.mtlpackage`` files (PK8c + 2026-06-02 expansion). Distinct from
#: ``PACKAGED_TEST_FIXTURES`` (Apple's bundled sample) so dashboards count
#: Tessera-owned kernels correctly. One representative per authorable family.
PACKAGED_PRODUCTION_KERNELS: tuple[BackendKernelEntry, ...] = (
    _TESSERA_MATMUL_ENTRY,
    _TESSERA_MATMUL_SOFTMAX_ENTRY,
    _TESSERA_SILU_ENTRY,
    _TESSERA_SOFTMAX_ENTRY,
    _TESSERA_RMSNORM_ENTRY,
    _TESSERA_ATTN_BLOCK_ENTRY,
    _TESSERA_RMSNORM_MATMUL_ENTRY,
)


def lookup_packaged_test_fixture(name: str) -> Optional[BackendKernelEntry]:
    """Look up a populated test-fixture-backed packaged entry by name.
    Returns ``None`` if no such fixture is registered."""
    return PACKAGED_TEST_FIXTURES.get(name)


def all_packaged_entries() -> tuple[BackendKernelEntry, ...]:
    """Return every populated packaged entry, fixtures + production
    together. Useful for the drift gate that walks all of them."""
    return tuple(PACKAGED_TEST_FIXTURES.values()) + PACKAGED_PRODUCTION_KERNELS


def resolve_packaged_path(entry: BackendKernelEntry,
                          *, root: Optional[Path] = None) -> Path:
    """Resolve a manifest entry's ``packaged_pipeline_path`` to an
    absolute filesystem path. ``root`` defaults to the repo root.

    The entry's path may be absolute (use it verbatim) or repo-relative
    (resolved against ``root``). Raises if the entry has no path
    declared (which shouldn't happen — ``__post_init__`` enforces it
    for ``status="packaged"`` — but the helper is robust to it)."""
    if not entry.packaged_pipeline_path:
        raise ValueError(
            f"entry has no packaged_pipeline_path: {entry!r}")
    p = Path(entry.packaged_pipeline_path)
    if p.is_absolute():
        return p
    base = root if root is not None else repo_root()
    return base / p


def validate_packaged_entry(
    entry: BackendKernelEntry, *, root: Optional[Path] = None
) -> tuple[bool, str]:
    """Drift gate — verify an entry's packaged-kernel claims hold up:

    1. The artifact path resolves to an existing directory ending in
       ``.mtlpackage``.
    2. If an ``apple_binding_spec`` is attached and the runtime can
       actually load Metal 4 packaged ML, the spec validates clean
       against the runtime reflection (binding names, buffer indices,
       dtypes, ranks).

    Returns ``(True, "")`` on success. On failure returns
    ``(False, reason)`` with a precise diagnostic — the test layer
    surfaces this in the failure message."""
    if entry.status != "packaged":
        return (False, f"entry status is {entry.status!r}, not 'packaged'")

    # 1. Path resolution + existence.
    try:
        path = resolve_packaged_path(entry, root=root)
    except ValueError as e:
        return (False, str(e))
    if not path.exists():
        return (False, f"packaged path does not exist: {path}")
    if path.suffix != ".mtlpackage":
        return (False, f"packaged path is not a .mtlpackage: {path}")
    if not path.is_dir():
        return (False, f"packaged path is not a directory: {path}")

    # 2. Binding-spec reflection check (only when MTL4 is loadable).
    if entry.apple_binding_spec is None:
        return (True, "")
    # Late import keeps the manifest module importable off-Darwin.
    from ..apple_mlpkg import (
        compile_mlpackage,
        last_error_kind,
        packaged_ml_available,
    )
    if not packaged_ml_available():
        # MTL4 not available — we can verify path + structure but not
        # the reflection contract. Treat this as a successful soft
        # validation (caller can re-check on a Metal-4 host).
        return (True, "")
    spec: AppleKernelBindingSpec = (
        entry.apple_binding_spec)  # type: ignore[assignment]
    # Pre-compute the dims map (in MTLTensorExtents innermost-first
    # convention) the package needs at setInputDimensions: time. We
    # take it from the spec's concrete-dims entries; wildcard specs
    # don't get to drive setInputDimensions and rely on the package's
    # defaults.
    input_dims: dict[int, tuple[int, ...]] = {}
    for ent in spec.entries:
        if ent.direction == "input":
            cd = ent.concrete_dims()
            if cd is not None:
                input_dims[ent.buffer_index] = cd
    pipe = compile_mlpackage(
        path, function_name=spec.function_name,
        input_dimensions=input_dims or None)
    if pipe is None:
        return (False, (
            f"compile_mlpackage failed; last_error_kind="
            f"{last_error_kind()}"))
    try:
        result = spec.validate_against(pipe, strict_extra=True)
        if not result.ok:
            return (False, (
                f"binding-spec drift vs runtime reflection: "
                f"{result.first_failure_reason}"))
    finally:
        pipe.destroy()
    return (True, "")


__all__ = [
    "PACKAGED_PRODUCTION_KERNELS",
    "PACKAGED_TEST_FIXTURES",
    "all_packaged_entries",
    "lookup_packaged_test_fixture",
    "repo_root",
    "resolve_packaged_path",
    "validate_packaged_entry",
]
