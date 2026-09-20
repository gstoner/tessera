"""Host-free: a foreign-target test must not make a device claim this host cannot check.

Three shapes cost a full day on 2026-09-19, and this file exists so none of
them can return silently.

1. **A tautological numerical assertion.** ``test_x86_fft_benchmark`` patched
   ``rt._x86_fft_c2c_rows`` to *be* ``scipy.fft`` and then asserted
   ``max_abs_error == 0.0`` -- that is ``|scipy - scipy| == 0``, which would
   have passed with the real kernel arbitrarily wrong, because the real kernel
   never ran. The tell was exactness: the harness itself allows ``2.0e-4``
   relative, so a genuine cross-library comparison can never be bit-identical.

2. **An undeclared module whose name promises execution.** ``e2e_spine``,
   ``parity``, ``benchmark`` and ``native`` read as device claims while testing
   the compile path. That is not a defect in the test -- it is a defect in what
   a reader concludes from a green run, and it misread that way twice in one
   session.

3. **A device entry point with no gate.** A test that reaches the GPU must skip
   where there is no GPU, never pass.

This file is itself host-free by construction: it reads source, and runs the
same on every box.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

UNIT = Path(__file__).resolve().parent

#: Module names whose subject is another architecture's hardware.
_FOREIGN = re.compile(
    r"nvidia|cuda|sm_?1[02]0|sm_?90|ptx|nvvm|"
    r"rocm|gfx1\d{3}|hsaco|rocdl|mfma|wmma|"
    r"x86|amx|avx512"
)
#: Names that promise the device did something.
_EXEC_NAMED = re.compile(
    r"execut|native|device|numeric|parity|measur|timing|throughput|latency|"
    r"launch|e2e|proof|exact|tflop|benchmark"
)
#: Names that are plainly about IR, text or registries -- fine on any host.
_CONTRACT_NAMED = re.compile(
    r"emit|contract|registry|manifest|fixture|lit|pipeline|metadata|catalog|"
    r"selector|schedule|inventory|audit|plan|dialect|claim"
)
#: Names whose INVOCATION needs real hardware. Matched as call nodes, never as
#: text: a test that asserts ``"cuModuleLoadData" in source`` is inspecting
#: generated code and is perfectly host-free, while a regex over the file
#: cannot tell that from an actual call. Getting this wrong flagged nine
#: innocent modules on the first run of this gate.
#:
#: ``rt.launch`` is deliberately NOT here. It is fail-soft: on a host without
#: the device it returns a refusal ``reason`` instead of raising, so calling it
#: is not evidence that a test needs hardware. Listing it would flag three
#: host-free dispatch tests and encode a wrong rule to buy a green gate.
#:
#: It does leave a softer question this gate does not answer: a test that
#: launches and then asserts ``"STRICT_DISPATCH" not in reason`` passes
#: VACUOUSLY where the refusal is a different one. That is a weak assertion,
#: not an unbacked device claim, and it wants its own check rather than a
#: wrong entry here.
_DEVICE_CALLEES = frozenset({
    "hipModuleLoadData", "hipModuleLoad", "hipModuleLaunchKernel",
    "hipMalloc", "hipMemcpy", "hipFree", "hipDeviceSynchronize",
    "cuModuleLoadData", "cuModuleLoadDataEx", "cuLaunchKernel",
    "cuMemAlloc", "cuMemcpyHtoD", "cuMemcpyDtoH",
    "_load_hip_for_launch", "_load_cuda_for_launch",
})


def _calls_a_device_entry_point(path: Path) -> str | None:
    """The first device call this module makes, or None. Calls only."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (func.attr if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name) else None)
        if name in _DEVICE_CALLEES:
            return name
    return None
_GATED = re.compile(r"skipif|importorskip|pytest\.skip|_or_skip")
_DECLARED = re.compile(r"host[- ]free|no device|without a device|contract test", re.I)


def _modules() -> list[Path]:
    return sorted(p for p in UNIT.glob("test_*.py") if _FOREIGN.search(p.stem.lower()))


def _docstring(path: Path) -> str:
    try:
        return ast.get_docstring(ast.parse(path.read_text())) or ""
    except SyntaxError:  # a broken module is another test's problem
        return ""


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.stem)
def test_device_reaching_module_is_gated(path: Path) -> None:
    """A test that reaches the GPU must be able to SKIP where there is none.

    Without a gate it fails on every host that lacks the hardware, which buries
    real regressions in noise -- the mirror of a hollow green, and just as
    misleading.
    """
    src = path.read_text()
    callee = _calls_a_device_entry_point(path)
    if callee is None:
        return
    assert _GATED.search(src), (
        f"(reaches {callee}) "
        f"{path.name} reaches a device entry point but has no skip gate. On a "
        f"host without that hardware it will FAIL rather than skip. Gate it the "
        f"way tests/unit/test_x86_fft_compiled.py does."
    )


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.stem)
def test_execution_named_module_is_gated_or_declares_host_free(path: Path) -> None:
    """An execution-sounding name must be backed by a gate or disclaimed.

    Ratchet, not a wall: the known set is listed below and may only SHRINK. The
    point is that the class cannot grow silently while the backlog is worked
    down.
    """
    stem = path.stem.lower()
    if not _EXEC_NAMED.search(stem) or _CONTRACT_NAMED.search(stem):
        return
    src = path.read_text()
    if _GATED.search(src) or _DECLARED.search(_docstring(path)):
        return
    assert path.stem in _UNDECLARED_ON_2026_09_20, (
        f"{path.name}'s name promises execution but it neither gates on the "
        f"hardware nor declares itself host-free, so a green run here reads as "
        f"a device result on a host that may have no device. Add a gate, or add "
        f"a 'Host-free:' line to the module docstring saying why the name is "
        f"misleading. Do NOT add it to the ratchet list."
    )


def test_ratchet_only_shrinks() -> None:
    """Every name in the ratchet must still be an undeclared, ungated module.

    A stale entry is worse than none: it keeps a resolved name on a list that
    reads as outstanding debt, and it quietly re-permits the name if the file
    ever regresses.
    """
    stale = []
    for name in sorted(_UNDECLARED_ON_2026_09_20):
        path = UNIT / f"{name}.py"
        if not path.exists():
            stale.append(f"{name} (file gone)")
            continue
        src = path.read_text()
        if _GATED.search(src) or _DECLARED.search(_docstring(path)):
            stale.append(f"{name} (now gated or declared)")
    assert not stale, (
        "these are no longer undeclared and must be REMOVED from "
        f"_UNDECLARED_ON_2026_09_20: {stale}"
    )


#: Shrink-only. Measured 2026-09-20 from a full arm64 sweep: of 2217
#: foreign-target tests passing on a host with no CUDA/ROCm/x86 execution, 397
#: were execution-named and 257 of those were already gated or declared. What
#: remained is below. Nine modules were verified host-free and declared on the
#: same day, which is why this list is as short as it is. Entries come off as
#: each is checked; nothing goes on.
_UNDECLARED_ON_2026_09_20: frozenset[str] = frozenset({
    "test_nvidia_aot_benchmark",
    "test_nvidia_attention_backward_benchmark",
    "test_nvidia_e2e_paged_kv_baseline",
    "test_nvidia_e2e_spine_performance",
    "test_nvidia_replay_parity",
    "test_nvidia_shared_arena_proof",
    "test_nvidia_transport_parity",
    "test_rocm_attention_backward_program_benchmark",
    "test_rocm_attention_carrier_benchmark",
    "test_rocm_canonical_gemm_kloop_benchmark",
    "test_rocm_int4_terminal_benchmark",
    "test_rocm_lds_arena_occupancy_benchmark",
    "test_rocm_packed_consumers_benchmark",
    "test_rocm_timing_provider",
    "test_ssm_rocm_replay_benchmark",
    "test_x86_e2e_breadth_performance_gate",
    "test_x86_e2e_cohort2_performance_gate",
    "test_x86_e2e_cohort34_performance_gate",
    "test_x86_e2e_dtype_performance_gate",
    "test_x86_e2e_elementwise_performance_gate",
    "test_x86_e2e_flat_followon_performance_gate",
    "test_x86_e2e_typed_logic_performance_gate",
    "test_x86_layout_materialization_benchmark",
})
