"""PK5 follow-on (2026-06-01) — populated packaged-kernel manifest entries.

The original PK5 sprint shipped the DATA MODEL for
``BackendKernelEntry(status="packaged")``. The audit's PK5 follow-on
asks for that scaffold to actually carry POPULATED entries — at least
one — so the manifest → load → reflect → bind → dispatch lifecycle is
exercised end-to-end against a real artifact.

The single populated entry today is Apple's bundled sample
matrix-multiplication package (kept under the ``test_fixture`` /
``apple_sample`` feature flags so it doesn't count as production
backend coverage in the dashboards).

These tests pin:

* **Structural** — the populated entry is well-formed (every
  ``BackendKernelEntry`` invariant holds); its
  ``apple_binding_spec`` round-trips cleanly.
* **Filesystem drift gate** — the entry's
  ``packaged_pipeline_path`` resolves to a real ``.mtlpackage``
  directory on disk. If the fixture gets renamed / removed, this
  test fires before behavior changes silently.
* **Reflection drift gate** — when Metal 4 packaged ML is
  available on the host, the entry's ``apple_binding_spec``
  validates clean against the runtime reflection of the loaded
  package. End-to-end PK5+Action 2 plumbing proof.
* **Lookup API** — the by-name lookup helper returns the populated
  entry; unknown names return ``None``.
* **No silent production claims** — the production-kernels tuple
  is empty today; this test fires if someone adds a Tessera-owned
  production kernel without also wiring it into
  ``primitive_coverage.py``.
"""

from __future__ import annotations

import pytest

from tessera.apple_mlpkg import AppleKernelBindingSpec
from tessera.compiler.apple_packaged_manifest import (
    PACKAGED_PRODUCTION_KERNELS,
    PACKAGED_TEST_FIXTURES,
    all_packaged_entries,
    lookup_packaged_test_fixture,
    repo_root,
    resolve_packaged_path,
    validate_packaged_entry,
)
from tessera.compiler.backend_manifest import BackendKernelEntry


# ---- Structural invariants ---------------------------------------------

def test_at_least_one_populated_packaged_entry_exists():
    """The whole point of the PK5 follow-on: the manifest is no longer
    empty. At minimum the Apple sample matmul fixture is wired in so
    the lifecycle has something to load."""
    assert len(PACKAGED_TEST_FIXTURES) >= 1, (
        "PK5 follow-on requires at least one populated test-fixture "
        "entry — the Apple sample matmul package")


def test_apple_matmul_fixture_is_present_and_well_formed():
    entry = lookup_packaged_test_fixture("apple_matmul_4x4_fp32")
    assert entry is not None
    assert isinstance(entry, BackendKernelEntry)
    assert entry.status == "packaged"
    assert entry.target == "apple_gpu"
    assert entry.dtypes == ("fp32",)
    assert "test_fixture" in entry.feature_flags
    assert "apple_sample" in entry.feature_flags
    # The spec is attached and round-trips.
    assert isinstance(entry.apple_binding_spec, AppleKernelBindingSpec)
    spec = entry.apple_binding_spec
    assert spec.function_name == "main"
    assert len(spec.entries) == 3
    by_name = spec.by_name()
    assert set(by_name.keys()) == {"inputA", "inputB", "output"}


def test_apple_matmul_fixture_path_is_repo_relative():
    """The path stored on the entry should be repo-relative (not
    absolute) so the manifest is portable across machines."""
    entry = lookup_packaged_test_fixture("apple_matmul_4x4_fp32")
    assert entry is not None
    p = entry.packaged_pipeline_path
    assert p is not None
    # Repo-relative paths don't start with /.
    assert not p.startswith("/"), (
        f"packaged_pipeline_path={p!r} should be repo-relative, not absolute")
    assert p.endswith(".mtlpackage")


# ---- Lookup helper -----------------------------------------------------

def test_lookup_returns_none_for_unknown_name():
    assert lookup_packaged_test_fixture("__not_a_real_fixture__") is None


def test_all_packaged_entries_unions_fixtures_and_production():
    all_e = all_packaged_entries()
    # Every entry in PACKAGED_TEST_FIXTURES must show up.
    for e in PACKAGED_TEST_FIXTURES.values():
        assert e in all_e
    # And every entry in PACKAGED_PRODUCTION_KERNELS must show up.
    for e in PACKAGED_PRODUCTION_KERNELS:
        assert e in all_e
    # The total count is the sum.
    assert len(all_e) == (len(PACKAGED_TEST_FIXTURES)
                          + len(PACKAGED_PRODUCTION_KERNELS))


# ---- Filesystem drift gate ---------------------------------------------

def test_apple_matmul_fixture_path_resolves_to_a_real_directory():
    """If the bundled fixture gets moved / renamed / deleted, this
    test fires immediately — the manifest claim becomes a lie."""
    entry = lookup_packaged_test_fixture("apple_matmul_4x4_fp32")
    assert entry is not None
    path = resolve_packaged_path(entry)
    assert path.exists(), (
        f"Apple matmul fixture missing at {path} — check "
        f"tests/fixtures/apple_gpu/ for the bundled sample")
    assert path.is_dir(), f"{path} is not a directory"
    assert path.suffix == ".mtlpackage"
    # An .mtlpackage must contain at least a manifest + binary at the
    # canonical sub-paths Apple uses. We're permissive here — exact
    # internal layout could vary across Apple toolchain versions — but
    # any non-empty directory passes.
    contents = list(path.iterdir())
    assert len(contents) > 0, f"{path} is empty"


def test_resolve_packaged_path_handles_absolute_paths():
    """The resolver shouldn't double-root an absolute path."""
    entry = BackendKernelEntry(
        target="apple_gpu",
        status="packaged",
        dtypes=("fp32",),
        feature_flags=(),
        notes="",
        packaged_pipeline_path="/absolute/path/to/foo.mtlpackage",
    )
    p = resolve_packaged_path(entry)
    assert str(p) == "/absolute/path/to/foo.mtlpackage"


# ---- Drift-gate helper -------------------------------------------------

def test_validate_packaged_entry_accepts_well_formed_fixture():
    """End-to-end PK5+Action 2 proof: the bundled Apple matmul
    fixture's manifest entry validates clean. When Metal 4 packaged ML
    is available, this also exercises the reflection drift gate; when
    it isn't, the structural validation alone passes."""
    entry = lookup_packaged_test_fixture("apple_matmul_4x4_fp32")
    assert entry is not None
    ok, reason = validate_packaged_entry(entry)
    assert ok, f"populated fixture entry failed validation: {reason}"


def test_validate_packaged_entry_rejects_non_packaged_status():
    """Sanity: the drift-gate helper refuses an entry that isn't
    a packaged-status row to begin with."""
    entry = BackendKernelEntry(
        target="apple_gpu",
        status="reference",
        dtypes=("fp32",),
        feature_flags=(),
        notes="",
    )
    ok, reason = validate_packaged_entry(entry)
    assert not ok
    assert "not 'packaged'" in reason


def test_validate_packaged_entry_rejects_missing_path():
    """If a populated entry points at a path that doesn't exist on
    this machine, the drift gate fires with a precise diagnostic."""
    fake = BackendKernelEntry(
        target="apple_gpu",
        status="packaged",
        dtypes=("fp32",),
        feature_flags=(),
        notes="",
        packaged_pipeline_path=(
            "tests/fixtures/apple_gpu/__missing__.mtlpackage"),
    )
    ok, reason = validate_packaged_entry(fake)
    assert not ok
    assert "does not exist" in reason


def test_validate_packaged_entry_rejects_wrong_suffix():
    fake = BackendKernelEntry(
        target="apple_gpu",
        status="packaged",
        dtypes=("fp32",),
        feature_flags=(),
        notes="",
        # Pointing at a real file but with the wrong suffix.
        packaged_pipeline_path=(
            "tests/fixtures/apple_gpu/APPLE_SAMPLE_LICENSE.txt"),
    )
    ok, reason = validate_packaged_entry(fake)
    assert not ok
    # Path exists but the suffix check fires first.
    assert "not a .mtlpackage" in reason or "not a directory" in reason


# ---- Validate every populated entry (catches NEW entries that drift) ---

def test_every_populated_entry_passes_validation():
    """Walk every entry in ``all_packaged_entries()`` and run it
    through the drift gate. As new populated entries are added, this
    test extends automatically — a typo in any future
    packaged_pipeline_path / binding_spec drift fires here."""
    failures: list[str] = []
    for entry in all_packaged_entries():
        ok, reason = validate_packaged_entry(entry)
        if not ok:
            failures.append(
                f"target={entry.target!r} path={entry.packaged_pipeline_path!r}: "
                f"{reason}")
    assert not failures, (
        "Populated packaged-manifest entries failed drift gate:\n"
        + "\n".join(failures))


# ---- Production-kernel population guardrail ---------------------------

def test_production_kernels_tuple_is_empty_today():
    """Honest-coverage check. Today Tessera doesn't ship its own
    ``.mtlpackage``; the only checked-in artifact is Apple's sample
    fixture. When the first Tessera-authored production kernel lands,
    THIS TEST WILL BREAK — the breaking is intentional. The fix is:

    1. Wire the new entry into ``primitive_coverage.py`` so the
       backend-coverage dashboard reflects it.
    2. Bump the lower bound below.

    This guardrail keeps the dashboard honest: a populated production
    kernel that didn't make it into the primitive registry is a
    silent dashboard lie."""
    assert len(PACKAGED_PRODUCTION_KERNELS) == 0, (
        f"PACKAGED_PRODUCTION_KERNELS grew to "
        f"{len(PACKAGED_PRODUCTION_KERNELS)} entries — also wire each "
        f"new entry into primitive_coverage.py's backend_kernel_manifest "
        f"for its op, then update this guardrail")


def test_repo_root_resolves_to_a_real_directory():
    root = repo_root()
    assert root.exists()
    assert root.is_dir()
    # Sanity-check: the resolved root contains the python/tessera/
    # tree the module itself lives in.
    assert (root / "python" / "tessera").exists()
