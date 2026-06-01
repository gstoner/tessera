"""Tests for the named pipeline capability gates (audit recommendation B).

The gate module is a *pure aggregator* over capabilities + backend_manifest +
execution_matrix + primitive_coverage + toolchain probes. These tests lock:

1. **Pure aggregator** — no truth sources beyond the four declared modules +
   stdlib (same allowlist pattern as ``test_op_target_conformance``).
2. **Gate enum coverage** — the seven canonical gate names + four status
   values are stable.
3. **Known-good targets** — cpu / apple_cpu / apple_gpu on Darwin pass every
   gate for ``matmul``. (If the test runs off-Darwin, apple_* hardware_smoke
   becomes fail — handled by the platform-gate spec.)
4. **Known-failing targets** — nvidia / rocm name ``toolchain`` as the first
   failing gate on a developer Mac without CUDA/ROCm toolchains. Metalium
   names ``link`` (its toolchain probe is intentionally not_evaluated).
5. **No silent passes** — every target/op pair either all-pass or has a
   concrete first_failing_gate with a non-empty detail string.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from tessera.compiler import pipeline_gates as pg


def test_gate_order_is_canonical():
    assert pg.GATE_ORDER == (
        pg.GATE_LEGALITY, pg.GATE_CODEGEN, pg.GATE_TOOLCHAIN,
        pg.GATE_LINK, pg.GATE_RUNTIME_ABI, pg.GATE_HARDWARE_SMOKE,
        pg.GATE_NUMERICAL,
    )


def test_module_is_pure_aggregator():
    src = Path(pg.__file__).read_text()
    src_no_strings = re.sub(r'"""[\s\S]*?"""', '', src)
    src_no_strings = re.sub(r"#[^\n]*", "", src_no_strings)
    bare = re.findall(r"^import\s+([\w\.]+)", src_no_strings, flags=re.M)
    froms = re.findall(r"^from\s+([\w\.]+)\s+import\s+([\w\., ]+)",
                       src_no_strings, flags=re.M)
    resolved: list[str] = list(bare)
    for pkg, names in froms:
        for raw in names.split(","):
            leaf = raw.strip().split(" as ")[0].strip()
            if leaf:
                resolved.append(f"{pkg}.{leaf}" if pkg != "__future__"
                                else pkg)
    allowed_prefixes = (
        "tessera.compiler.primitive_coverage",
        "tessera.compiler.backend_manifest",
        "tessera.compiler.execution_matrix",
        "tessera.compiler.capabilities",
        # stdlib + typing
        "__future__", "dataclasses", "pathlib", "typing",
        "shutil", "sys",
    )
    for mod in resolved:
        assert any(mod.startswith(p) for p in allowed_prefixes), (
            f"pipeline_gates.py is supposed to be a pure aggregator; "
            f"import {mod!r} not in allowed truth-source set")


def test_evaluate_returns_seven_results_in_canonical_order():
    results = pg.evaluate("cpu", "matmul")
    assert len(results) == 7
    assert tuple(r.gate for r in results) == pg.GATE_ORDER


@pytest.mark.skipif(sys.platform != "darwin",
                    reason="apple_cpu / apple_gpu hardware_smoke requires Darwin")
@pytest.mark.parametrize("target", ["cpu", "apple_cpu", "apple_gpu"])
def test_known_good_target_passes_all_gates_for_matmul(target):
    """matmul on a host with all the right pieces should pass every gate."""
    assert pg.first_failing_gate(target, "matmul") is None
    results = pg.evaluate(target, "matmul")
    for r in results:
        assert r.status in (pg.STATUS_PASS, pg.STATUS_NOT_APPLICABLE,
                            pg.STATUS_NOT_EVALUATED), (
            f"{target}/{r.gate}: status={r.status} detail={r.detail}")


def test_nvidia_matmul_first_failing_gate_is_toolchain():
    """On a developer Mac without CUDA installed, the *named* gate is
    toolchain — not 'reference_cpu' silently, not 'unsupported' generically."""
    result = pg.first_failing_gate("nvidia", "matmul")
    assert result is not None
    assert result.gate == pg.GATE_TOOLCHAIN
    assert "nvcc" in result.detail


def test_rocm_matmul_first_failing_gate_is_toolchain():
    result = pg.first_failing_gate("rocm", "matmul")
    assert result is not None
    assert result.gate == pg.GATE_TOOLCHAIN
    assert "hipcc" in result.detail


def test_metalium_matmul_first_failing_gate_is_link():
    """Metalium intentionally lists its toolchain probe as not_evaluated
    (separate SDK surface); the first FAIL is therefore link."""
    result = pg.first_failing_gate("metalium", "matmul")
    assert result is not None
    assert result.gate == pg.GATE_LINK


def test_every_failing_gate_has_a_nonempty_detail():
    """No silent fails. The audit's whole point is that 'unsupported' must
    name *why*."""
    targets = ("cpu", "apple_cpu", "apple_gpu", "nvidia", "rocm", "metalium")
    ops = ("matmul", "softmax", "flash_attn", "conv2d", "kv_cache_read")
    for t in targets:
        for op in ops:
            for r in pg.evaluate(t, op):
                if r.status == pg.STATUS_FAIL:
                    assert r.detail, (
                        f"{t}/{op}/{r.gate}: FAIL with empty detail — "
                        "every fail must say why")


def test_apple_gpu_softmax_passes_via_runtime_path():
    """softmax on apple_gpu has a fused MSL kernel + runtime envelope entry.
    Every gate should pass on Darwin."""
    if sys.platform != "darwin":
        pytest.skip("requires Darwin for apple_gpu hardware_smoke")
    assert pg.first_failing_gate("apple_gpu", "softmax") is None


def test_unknown_op_legality_fails():
    """An op not in primitive_coverage fails legality immediately."""
    result = pg.first_failing_gate("apple_gpu", "no_such_op_ever_xyz")
    assert result is not None
    assert result.gate == pg.GATE_LEGALITY
    assert "no_such_op_ever_xyz" in result.detail


def test_unknown_target_codegen_fails_or_legality():
    """An unknown target either fails legality (target not in capabilities)
    or fails codegen (no manifest entry); both are acceptable, but it must
    fail at SOME named gate — never silently pass."""
    result = pg.first_failing_gate("xeno_dsp", "matmul")
    assert result is not None
