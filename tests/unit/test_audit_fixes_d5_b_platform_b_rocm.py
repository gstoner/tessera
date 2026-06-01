"""Regression guards for three audit-found bugs (2026-05-31):

* **P1 / D.5** — ``c -= b`` was silently dropping the subtract because
  ``visit_AugAssign`` accepted ``Sub``/``Div`` but ``_try_map_binop`` only
  knew ``Add``/``Mult``. The user got the "desugared" info diagnostic but
  no ``tessera.sub`` op was emitted. Fixed by adding real ``Sub``/``Div``
  BinOp lowering in ``_try_map_binop``.

* **P2 / pipeline_gates rocm** — ``_manifest_entries`` only knew
  ``nvidia_*`` family-mapping. A per-arch ROCm target like
  ``rocm_gfx942`` matched zero manifest rows and the codegen gate
  spuriously failed. Fixed by adding the symmetric ROCm family-mapping.

* **P2 / hardware_smoke arm64** — ``_platform_is_darwin_arm64`` returned
  ``True`` for any Darwin host (Intel Macs included). Fixed by also
  checking ``platform.machine() == "arm64"``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tessera.compiler import pipeline_gates as pg
from tessera.compiler.graph_ir import GraphIRBuilder


# ----- P1 / D.5: AugAssign Sub/Div ----------------------------------------

def test_aug_assign_sub_emits_real_tessera_sub_op(tmp_path):
    """Pre-fix: ``c -= b`` emitted the desugared info note but NO
    tessera.sub op (silent drop). Post-fix: the IR body contains a real
    tessera.sub op."""
    src = (
        "import tessera as ts\n"
        "def f(a, b):\n"
        "    c = ts.ops.relu(a)\n"
        "    c -= b\n"
        "    return c\n"
    )
    (tmp_path / "m.py").write_text(src)
    import sys
    sys.path.insert(0, str(tmp_path))
    try:
        import m  # noqa: I001
        builder = GraphIRBuilder()
        fn = builder.lower(m.f)
        ops = [op.op_name for op in fn.body]
        assert "tessera.sub" in ops, (
            f"c -= b must emit a real tessera.sub op; got {ops!r}")
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("m", None)


def test_aug_assign_div_emits_real_tessera_div_op(tmp_path):
    """Symmetric guard for ``/=``."""
    src = (
        "import tessera as ts\n"
        "def f(a, b):\n"
        "    c = ts.ops.relu(a)\n"
        "    c /= b\n"
        "    return c\n"
    )
    (tmp_path / "m.py").write_text(src)
    import sys
    sys.path.insert(0, str(tmp_path))
    try:
        import m
        builder = GraphIRBuilder()
        fn = builder.lower(m.f)
        ops = [op.op_name for op in fn.body]
        assert "tessera.div" in ops, ops
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("m", None)


def test_plain_sub_and_div_binops_lower_too(tmp_path):
    """Non-augmented ``a - b`` / ``a / b`` must now also lower (we widened
    the ``_emit_expr`` accept-set to include Sub/Div)."""
    src = (
        "import tessera as ts\n"
        "def f(a, b):\n"
        "    p = ts.ops.relu(a)\n"
        "    q = p - b\n"
        "    r = q / b\n"
        "    return r\n"
    )
    (tmp_path / "m.py").write_text(src)
    import sys
    sys.path.insert(0, str(tmp_path))
    try:
        import m
        builder = GraphIRBuilder()
        fn = builder.lower(m.f)
        ops = [op.op_name for op in fn.body]
        assert "tessera.sub" in ops and "tessera.div" in ops, ops
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("m", None)


# ----- P2 / pipeline_gates rocm ------------------------------------------

def test_rocm_per_arch_target_finds_family_manifest_entry():
    """Pre-fix: ``rocm_gfx942`` matched zero manifest entries → codegen
    spuriously failed first. Post-fix: per-arch ROCm targets inherit the
    rocm family manifest row, so the *real* first failing gate is the
    one the audit cared about (``toolchain`` — no hipcc)."""
    result = pg.first_failing_gate("rocm_gfx942", "matmul")
    assert result is not None
    assert result.gate == pg.GATE_TOOLCHAIN, (
        f"expected first failing gate to be toolchain (no hipcc); "
        f"got {result.gate} ({result.detail})")
    assert "hipcc" in result.detail


@pytest.mark.parametrize("arch", [
    "rocm_gfx90a", "rocm_gfx940", "rocm_gfx942",
    "rocm_gfx950", "rocm_gfx1100",
])
def test_every_rocm_subarch_inherits_manifest_codegen_pass(arch):
    """Per-arch ROCm targets in the capabilities registry all match the
    ``rocm`` family manifest row. None should report codegen=fail."""
    results = {r.gate: r for r in pg.evaluate(arch, "matmul")}
    assert results["codegen"].status == "pass", (
        f"{arch}: codegen={results['codegen'].status!r} — "
        f"manifest matching regressed")


def test_rocm_family_target_still_matches():
    """Bare ``rocm`` family target keeps working (the fix is additive)."""
    result = pg.first_failing_gate("rocm", "matmul")
    assert result is not None
    assert result.gate == pg.GATE_TOOLCHAIN


def test_nvidia_per_sm_targets_still_match_exactly():
    """The fix preserves the NVIDIA per-SM exact-match behavior."""
    for sm in ("nvidia_sm80", "nvidia_sm90", "nvidia_sm100", "nvidia_sm120"):
        results = {r.gate: r for r in pg.evaluate(sm, "matmul")}
        assert results["codegen"].status == "pass", (
            f"{sm}: codegen regressed to {results['codegen'].status}")


# ----- P2 / hardware_smoke arm64 ------------------------------------------

def test_platform_check_requires_actual_arm64():
    """The helper name promises arm64; the body must actually check it.
    Patch ``platform.machine`` to simulate an Intel Mac and confirm the
    helper returns ``False`` even though ``sys.platform == "darwin"`` is
    still true."""
    with patch("platform.machine", return_value="x86_64"):
        with patch.object(pg, "sys") as mock_sys:
            mock_sys.platform = "darwin"
            assert pg._platform_is_darwin_arm64() is False, (
                "Intel Mac (darwin + x86_64) must NOT pass the "
                "arm64 helper")


def test_platform_check_accepts_real_apple_silicon():
    """Apple Silicon (darwin + arm64) returns True."""
    with patch("platform.machine", return_value="arm64"):
        with patch.object(pg, "sys") as mock_sys:
            mock_sys.platform = "darwin"
            assert pg._platform_is_darwin_arm64() is True


def test_platform_check_rejects_linux():
    """Linux returns False regardless of machine."""
    with patch("platform.machine", return_value="aarch64"):
        with patch.object(pg, "sys") as mock_sys:
            mock_sys.platform = "linux"
            assert pg._platform_is_darwin_arm64() is False


def test_apple_gpu_hardware_smoke_fails_on_intel_mac():
    """The downstream consequence: on an Intel Mac, the apple_gpu
    hardware_smoke gate must NOT report pass — it would falsely greenlight
    Metal MPS execution on a host that can't run it. Patch the helper
    and check the gate evaluates correctly."""
    from tessera.compiler.pipeline_gates import _eval_hardware_smoke
    with patch("platform.machine", return_value="x86_64"):
        with patch.object(pg, "sys") as mock_sys:
            mock_sys.platform = "darwin"
            result = _eval_hardware_smoke("apple_gpu", "matmul")
            assert result.status == pg.STATUS_FAIL, (
                f"Intel Mac must FAIL apple_gpu hardware_smoke; got "
                f"{result.status}")
            assert "Apple silicon" in result.detail
