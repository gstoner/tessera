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
    return None
