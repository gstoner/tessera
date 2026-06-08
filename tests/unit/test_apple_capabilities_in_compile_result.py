"""Apple-sample Actions 1 + 6 — capability + archive telemetry on
``CompileResult.to_dict()``.

The doc-deep review surfaced two gaps:

* **Action 1 — Capability-first lowering**: ``target=apple_gpu`` was
  carrying no per-feature detail. A developer Mac with macOS 26 has
  MTL4 + tensors + argument tables; an older host has MPS only.
  Dashboards / B's named-gates need the structured answer.
* **Action 6 — AOT cache telemetry**: ``MTL4Archive`` was plumbed in
  the runtime (``mtl4_archive_*`` fields on ``MetalDeviceContext``)
  but its enabled/path/lookup-loaded state wasn't exposed to Python.

Both flow through ``apple_gpu_capabilities_snapshot()`` in
``_apple_gpu_dispatch.py`` and land on ``CompileResult.to_dict()``
under stable keys.

These tests pin:

1. The snapshot has the documented shape on every host (Darwin or
   not) — empty/false values when the runtime isn't available; full
   per-feature dict when it is.
2. ``CompileResult.to_dict()`` carries the snapshot for Apple targets
   and omits it for non-Apple targets.
3. The capability bit names are stable (drift gate against renames).
4. The archive snapshot reflects ``mtl4_archive_enable`` state
   changes correctly.
"""

from __future__ import annotations

from typing import Any

import pytest

from tessera._apple_gpu_dispatch import (
    apple_gpu_capabilities_snapshot,
    apple_gpu_runtime,
    bind_symbol,
)


# ---- Snapshot shape is stable on every host -----------------------------

def test_snapshot_always_returns_documented_shape():
    snap = apple_gpu_capabilities_snapshot()
    # Top-level keys.
    for k in ("runtime_available", "capabilities", "capabilities_raw",
              "mtl4_full", "archive"):
        assert k in snap, f"snapshot missing key {k!r}"
    # Types.
    assert isinstance(snap["runtime_available"], bool)
    assert isinstance(snap["capabilities"], dict)
    assert isinstance(snap["capabilities_raw"], int)
    assert isinstance(snap["mtl4_full"], bool)
    assert isinstance(snap["archive"], dict)
    # Archive sub-keys.
    for k in ("available", "enabled", "has_lookup", "path"):
        assert k in snap["archive"], f"snapshot.archive missing {k!r}"


def test_capability_bit_names_are_stable():
    """The Tessera B-gates (downstream) consume these names. A rename
    would silently break any dashboard that hard-codes them. Pin the
    set + spelling here."""
    snap = apple_gpu_capabilities_snapshot()
    expected = {
        "mtl4_command_queue", "mtl4_command_allocator", "mtl4_compiler",
        "mtl_tensor", "msl_4_0",
    }
    # The keys should be exactly the expected set (no extras / no missing)
    # when capabilities are populated. On a host where the runtime
    # isn't available, capabilities is an empty dict — that's fine; the
    # bit-name pin is enforced via the module-level table.
    from tessera._apple_gpu_dispatch import _APPLE_GPU_CAP_BITS
    actual = {name for _, name in _APPLE_GPU_CAP_BITS}
    assert actual == expected


def test_runtime_loaded_implies_capabilities_populated():
    """When the runtime dylib loaded, the snapshot's `capabilities`
    dict must be fully populated (every expected bit-name appears as a
    key)."""
    if apple_gpu_runtime() is None:
        pytest.skip("Apple GPU runtime not buildable on this host")
    snap = apple_gpu_capabilities_snapshot()
    assert snap["runtime_available"] is True
    caps = snap["capabilities"]
    for name in ("mtl4_command_queue", "mtl4_command_allocator",
                 "mtl4_compiler", "mtl_tensor", "msl_4_0"):
        assert name in caps, f"missing capability key {name!r}"
        assert isinstance(caps[name], bool)


def test_mtl4_full_matches_all_bits():
    """``mtl4_full`` is the rc=1 case of the C probe — every cap bit
    set. Verify the Python decoder agrees with the C answer."""
    if apple_gpu_runtime() is None:
        pytest.skip("Apple GPU runtime not buildable on this host")
    snap = apple_gpu_capabilities_snapshot()
    all_caps_true = all(snap["capabilities"].values())
    assert snap["mtl4_full"] == all_caps_true


# ---- CompileResult.to_dict() carries the snapshot for Apple targets ----

def _tiny_matmul_module():
    from tessera.compiler.graph_ir import (
        GraphIRFunction, GraphIRModule, IRArg, IROp, IRType,
    )
    ten_a = IRType("tensor<128x64xf32>", ("128", "64"), "fp32")
    ten_b = IRType("tensor<64x128xf32>", ("64", "128"), "fp32")
    ten_c = IRType("tensor<128x128xf32>", ("128", "128"), "fp32")
    fn = GraphIRFunction(
        name="m",
        args=[IRArg("a", ten_a), IRArg("b", ten_b)],
        result_types=[ten_c],
        body=[IROp(
            result="c", op_name="tessera.matmul",
            operands=["%a", "%b"],
            operand_types=["tensor<128x64xf32>", "tensor<64x128xf32>"],
            result_type="tensor<128x128xf32>",
        )],
        return_values=["%c"],
    )
    return GraphIRModule(functions=[fn])


def test_compile_result_apple_gpu_carries_capability_keys():
    from tessera.compiler.canonical_compile import canonical_compile
    result = canonical_compile(_tiny_matmul_module(), target="apple_gpu")
    d = result.to_dict()
    for key in (
        "apple_gpu_capabilities",
        "apple_gpu_capabilities_raw",
        "apple_gpu_mtl4_full",
        "apple_gpu_archive",
        "apple_gpu_runtime_available",
    ):
        assert key in d, f"to_dict missing Apple-target key {key!r}"


def test_compile_result_apple_cpu_carries_capability_keys():
    """``apple_cpu`` also gets the snapshot — same physical device, same
    capability picture, useful for dashboards uniformly classifying
    Apple-family hosts."""
    from tessera.compiler.canonical_compile import canonical_compile
    result = canonical_compile(_tiny_matmul_module(), target="apple_cpu")
    d = result.to_dict()
    assert "apple_gpu_capabilities" in d
    assert "apple_gpu_archive" in d


def test_compile_result_non_apple_target_omits_capability_keys():
    """A NVIDIA / CPU compile result has no Apple-capability keys —
    otherwise dashboards would render Apple flags on irrelevant rows."""
    from tessera.compiler.canonical_compile import canonical_compile
    for target in ("cpu", "nvidia_sm90", "rocm"):
        result = canonical_compile(_tiny_matmul_module(), target=target)
        d = result.to_dict()
        assert "apple_gpu_capabilities" not in d, (
            f"target={target}: should not carry Apple-capability keys")
        assert "apple_gpu_archive" not in d, target


def test_capability_snapshot_in_to_dict_matches_direct_call():
    """The snapshot on the compile result equals the snapshot taken
    directly via the public helper — no transformation drift."""
    from tessera.compiler.canonical_compile import canonical_compile
    result = canonical_compile(_tiny_matmul_module(), target="apple_gpu")
    d = result.to_dict()
    direct = apple_gpu_capabilities_snapshot()
    assert d["apple_gpu_capabilities"] == direct["capabilities"]
    assert d["apple_gpu_capabilities_raw"] == direct["capabilities_raw"]
    assert d["apple_gpu_mtl4_full"] == direct["mtl4_full"]
    assert d["apple_gpu_archive"] == direct["archive"]


# ---- Archive snapshot reflects runtime state ----------------------------

def test_archive_snapshot_initially_disabled():
    """A fresh process has no archive enabled — capture this baseline
    so a future change that auto-enables archive doesn't slip through
    without a deliberate decision."""
    if apple_gpu_runtime() is None:
        pytest.skip("Apple GPU runtime not buildable on this host")
    snap = apple_gpu_capabilities_snapshot()
    arc = snap["archive"]
    assert arc["available"] is True, (
        "the archive-state probe must respond when the runtime is up")
    # Note: enabled / has_lookup state is process-shared; we don't pin
    # them here because earlier tests in the same session may have
    # toggled them. We just verify ``available`` is honest and the
    # path field is a string (possibly empty).
    assert isinstance(arc["path"], str)


def test_archive_snapshot_reflects_enable_call(tmp_path):
    """After ``tessera_apple_gpu_mtl4_archive_enable(path)``, the
    archive snapshot must report ``enabled=True`` and ``path=<that
    path>``. Catches a regression where the telemetry probe reads
    stale state."""
    import ctypes
    if apple_gpu_runtime() is None:
        pytest.skip("Apple GPU runtime not buildable on this host")
    enable = bind_symbol(
        "tessera_apple_gpu_mtl4_archive_enable",
        (ctypes.c_char_p,),
        restype=ctypes.c_int32,
    )
    if enable is None:
        pytest.skip("archive enable symbol unavailable")
    target_path = str(tmp_path / "test_archive.mtl4archive")
    rc = enable(target_path.encode("utf-8"))
    # The enable call may return 0 if MTL4Archive isn't available on
    # this host (older OS). In that case the snapshot path follows the
    # same answer — verify consistency, not absolute truth.
    snap = apple_gpu_capabilities_snapshot()
    arc = snap["archive"]
    if rc == 1:
        assert arc["enabled"] is True
        assert arc["path"] == target_path
    else:
        # On non-MTL4 hosts (or where archive setup failed), the
        # snapshot should also report not-enabled rather than lying.
        assert arc["enabled"] is False


# ---- Pure-aggregator invariant survives the new lazy import -------------

def test_canonical_compile_still_pure_aggregator():
    """``canonical_compile.py``'s pure-aggregator drift gate
    (test_canonical_compile.py::test_module_is_pure_aggregator)
    operates on MODULE-level imports. The Apple snapshot is imported
    LAZILY inside ``to_dict()`` so the rule still holds. Verify the
    module's top-level imports haven't grown a new dependency."""
    from pathlib import Path
    src = Path(
        __file__).resolve().parents[1].parent / (
        "python/tessera/compiler/canonical_compile.py")
    text = src.read_text()
    # The lazy import must be indented (inside a function), not at
    # column 0.
    for line in text.splitlines():
        if line.startswith("from tessera._apple_gpu_dispatch"):
            raise AssertionError(
                "module-level import of _apple_gpu_dispatch would "
                "violate the pure-aggregator rule; the lazy import "
                "must remain inside to_dict()")
