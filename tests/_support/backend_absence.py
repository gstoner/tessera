"""Failures that mean "this host did not build / install that backend".

A Mac `tessera-opt` has no NVIDIA or ROCm backend, so a test that drives an
SM120 or gfx1201 lowering fails with the compiler's own refusal instead of
skipping. A sweep then reports ~160 red tests that no Mac can evaluate, and a
real defect (a `tessera-opt` segfault in the same sweep) hides among them.

`backend_absence_reason` converts such a failure to a skip, but only when BOTH
hold, so it can never turn a defect into a green result:

1. the failure text is one of a short list of "backend not present" messages;
2. a probe of THIS host confirms the named capability is really missing
   (dialect not registered in the driver, pass not registered, tool not found).

A host that has the backend never skips: the message there is a defect. This is
the claim-integrity rule in CLAUDE.md (a host that cannot evaluate a lane must
say so, never report a pass), applied in the other direction from a hollow green.
"""

from __future__ import annotations

import re

from tests._support import compiler_tool
from tests._support.environment import nvidia_cuda_tool

_NVIDIA_DIALECT_REFUSAL = "requires the registered NVIDIA Target IR dialect"
_NVIDIA_TOOLS_REFUSAL = "NVIDIA native packaging tools unavailable:"
_UNKNOWN_ROCM_PASS = re.compile(r"Unknown command line argument '--(lower-tile-to-rocm)")
_UNREGISTERED_PASS = re.compile(r"'(tessera-rocm-executable)' does not refer to a registered pass")
# Any backend-owned pass a driver does not register, as an option or inside a
# textual pipeline. Only names that carry a backend token qualify, so a missing
# *generic* pass -- a real regression -- can never be converted to a skip.
_UNKNOWN_PASS = re.compile(r"Unknown command line argument '--([A-Za-z0-9][\w-]*)")
_UNREGISTERED_PIPELINE_PASS = re.compile(r"'([A-Za-z0-9][\w-]*)' does not refer to a registered pass")
_BACKEND_TOKENS = (("NVIDIA", ("nvidia", "nvvm")), ("ROCm", ("rocm", "rocdl")))
# MLIR's parser names the dialect it could not load for a custom op.
_UNREGISTERED_DIALECT = re.compile(r"Dialect `(tessera_nvidia|tessera_rocm)' not found for custom op")
# A subprocess.CalledProcessError's text carries argv, not stderr: a backend pass
# in argv, together with a non-zero exit, is checked against the driver below.
_FAILED_COMMAND = re.compile(r"Command '\[.*\]' returned non-zero exit status")
_ARGV_PASS = re.compile(r"'--([A-Za-z0-9][\w-]*)(?:=[^']*)?'")
_ROCM_CLANG_REFUSAL = "ROCm native packaging requires AMD clang to identify OCML/OCKL/OCLC"
_HIPCC_GATE = re.compile(r"hipcc not on PATH")


# Test families that skip on the Mac for want of an NVIDIA/ROCm backend but
# exercise a target-independent capability Apple does not have yet. The skip
# reason names the tracked item so a gap never reads as "just platform-specific"
# (docs/audit/backend/apple/todo.md, "Needs To Be Implemented on Apple").
APPLE_NEEDS_IMPLEMENTATION: dict[str, str] = {
    **{name: "APPLE-NTBI-1" for name in (
        "test_portable_bounded_source_attention_reverse.py",
        "test_prepared_bounded_source_attention_reverse.py",
        "test_portable_source_attention_reverse.py",
        "test_portable_source_attention_reverse_auto.py",
        "test_portable_source_attention_reverse_rows.py",
        "test_native_bounded_source_attention_reverse.py",
        "test_native_source_attention_reverse_transport.py",
        "test_native_sm120_source_reverse.py",
        "test_sm120_bounded_attention_pullback.py",
        "test_sm120_bounded_source_rows.py",
    )},
    **{name: "APPLE-NTBI-2" for name in (
        "test_native_tensor_shape_plan.py",
        "test_native_semantic_dag_ownership.py",
        "test_native_tensor_shape_ownership.py",
        "test_native_tensor_bounded_dag.py",
        "test_native_tensor_program_pipeline.py",
        "test_native_tensor_pipeline_layout.py",
        "test_native_tensor_binary_producers.py",
        "test_native_tensor_bounded_producers.py",
        "test_native_tensor_mlp_program.py",
        "test_native_tensor_fp32_roots.py",
    )},
    "test_native_bounded_attention_jvp_runtime.py": "APPLE-NTBI-3",
    **{name: "APPLE-NTBI-4" for name in (
        "test_rocm_block_quant_package.py",
        "test_rocm_block_quant_frontend.py",
        "test_rocm_quantized_product_program.py",
        "test_rocm_quantized_nvfp4_product_program.py",
        "test_rocm_quantized_gate_up_program.py",
        "test_rocm_quantized_mlp_program.py",
    )},
}


def apple_gap_note(test_path: str) -> str:
    """On a Mac, the Needs-To-Be-Implemented tag for a backend-absent skip, else ''."""
    import platform
    from pathlib import Path

    if platform.system() != "Darwin":
        return ""
    item = APPLE_NEEDS_IMPLEMENTATION.get(Path(test_path).name)
    return f" [Apple: Needs To Be Implemented — {item}]" if item else ""


def _driver():
    return compiler_tool.tessera_opt_path()


def _missing_nvidia_tools(text: str) -> list[str]:
    match = re.search(re.escape(_NVIDIA_TOOLS_REFUSAL) + r"\s*\[([^\]]*)\]", text)
    if not match:
        return []
    return re.findall(r"'([^']+)'", match.group(1))


def _tool_is_absent(name: str) -> bool:
    if name == "tessera-nvidia-opt":
        from tests._support.environment import CompilerToolchain

        return CompilerToolchain.discover().nvidia_opt is None
    return nvidia_cuda_tool(name) is None


def _backend_of(pass_name: str) -> str | None:
    lowered = pass_name.lower()
    for backend, tokens in _BACKEND_TOKENS:
        if any(token in lowered for token in tokens):
            return backend
    return None


def _amd_clang_is_absent() -> bool:
    # The refusal's own discovery, so the probe cannot disagree with the raise.
    from tessera.compiler import rocm_native

    return rocm_native._rocm_clang(rocm_native._rocm_path()) is None


def _hipcc_is_absent() -> bool:
    import shutil
    from pathlib import Path

    roots = ("/opt/rocm", "/opt/rocm/core", "/opt/rocm/core-10.0")
    return shutil.which("hipcc") is None and not any(
        (Path(root) / "bin/hipcc").is_file() for root in roots)


def backend_absence_reason(text: str) -> str | None:
    """A skip reason when `text` is a backend-absent failure this host confirms."""
    driver = _driver()

    if _NVIDIA_DIALECT_REFUSAL in text:
        if driver is not None and "tessera_nvidia" not in compiler_tool.registered_dialects(driver):
            return ("this host's tessera-opt has no NVIDIA backend "
                    "(tessera_nvidia dialect not registered); evaluate on an NVIDIA box")
        return None

    tools = _missing_nvidia_tools(text)
    if tools:
        if all(_tool_is_absent(tool) for tool in tools):
            return f"NVIDIA native packaging tools are not installed on this host: {', '.join(tools)}"
        return None

    for pattern in (_UNKNOWN_ROCM_PASS, _UNREGISTERED_PASS):
        match = pattern.search(text)
        if match:
            name = match.group(1)
            if driver is not None and name not in compiler_tool.registered_passes(driver):
                return (f"this host's tessera-opt has no ROCm backend "
                        f"(pass '{name}' not registered); evaluate on a ROCm box")
            return None

    for pattern in (_UNKNOWN_PASS, _UNREGISTERED_PIPELINE_PASS):
        match = pattern.search(text)
        if match:
            name = match.group(1)
            backend = _backend_of(name)
            if (backend is not None and driver is not None
                    and name not in compiler_tool.registered_passes(driver)):
                return (f"this host's tessera-opt has no {backend} backend "
                        f"(pass '{name}' not registered); evaluate on a{'n' if backend == 'NVIDIA' else ''} "
                        f"{backend} box")
            return None

    match = _UNREGISTERED_DIALECT.search(text)
    if match:
        dialect = match.group(1)
        if driver is not None and dialect not in compiler_tool.registered_dialects(driver):
            backend = "NVIDIA" if dialect == "tessera_nvidia" else "ROCm"
            return (f"this host's tessera-opt has no {backend} backend "
                    f"({dialect} dialect not registered); evaluate on a{'n' if backend == 'NVIDIA' else ''} "
                    f"{backend} box")
        return None

    if _FAILED_COMMAND.search(text) and driver is not None:
        registered = compiler_tool.registered_passes(driver)
        for name in _ARGV_PASS.findall(text):
            backend = _backend_of(name)
            if backend is not None and name not in registered:
                return (f"this host's tessera-opt has no {backend} backend "
                        f"(pass '{name}' in the failed command is not registered); "
                        f"evaluate on a{'n' if backend == 'NVIDIA' else ''} {backend} box")
        return None

    if _ROCM_CLANG_REFUSAL in text:
        if _amd_clang_is_absent():
            return "this host has no ROCm toolkit (AMD clang not found); evaluate on a ROCm box"
        return None

    if _HIPCC_GATE.search(text):
        if _hipcc_is_absent():
            return "this host has no ROCm toolkit (hipcc not found); evaluate on a ROCm box"
        return None
    return None
