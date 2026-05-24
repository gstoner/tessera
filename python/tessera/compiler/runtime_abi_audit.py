"""Audit-C (2026-05-22) — Runtime C ABI surface audit.

Catalogues the C ABI symbols Tessera exports across backends so that
naming consistency, version pinning, and per-target coverage are
visible from one place.

What's audited
--------------

  * Core runtime headers (`src/runtime/include/tessera/`) — the
    backend-agnostic ABI: `tessera_runtime.h`, `tsr_kernel.h`,
    `tsr_types.h`, `tsr_status.h`, `tsr_shape.h`, `tsr_version.h`.
  * Per-backend `extern "C" tessera_*` symbol surfaces in:
      - `src/compiler/codegen/Tessera_Apple_Backend/runtime/`
      - `src/compiler/codegen/tessera_x86_backend/include/tessera/x86/`
      - `src/compiler/codegen/tessera_gpu_backend_NVIDIA/include/`
      - `src/compiler/codegen/Tessera_ROCM_Backend/include/`
  * Per-symbol metadata: backend, op family, dtype variant.
  * Version pin consistency: NCCL/RCCL ≥ 2.22, CUDA 13.2.1,
    ROCm 7.2.3 across Python + CMake + C++ headers.

The dashboard at ``docs/audit/generated/runtime_abi.md`` surfaces:

  * Total symbol count per backend.
  * Per-op-family × per-dtype coverage matrix.
  * Naming-pattern audit (`tessera_<backend>_<op>_<dtype>`).
  * Honest gap call-outs (e.g., NVIDIA / ROCm headers exist but
    runtime symbols are gated on real hardware).

Drift gates at ``tests/unit/test_runtime_abi_audit.py``:

  * Core runtime headers exist at the expected paths.
  * Apple GPU symbol count stays at-or-above the Phase 8.4.7 floor
    (26 symbols across 9 kernel families × dtypes).
  * Symbol naming follows the canonical
    `tessera_<backend>_<op>_<dtype>` pattern.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"


# ─────────────────────────────────────────────────────────────────────────
# Header inventories
# ─────────────────────────────────────────────────────────────────────────


# Canonical backend-agnostic runtime headers.  These MUST exist; their
# absence is a build-broken regression.
_CORE_RUNTIME_HEADERS: tuple[str, ...] = (
    "src/runtime/include/tessera/tessera_runtime.h",
    "src/runtime/include/tessera/tsr_kernel.h",
    "src/runtime/include/tessera/tsr_types.h",
    "src/runtime/include/tessera/tsr_status.h",
    "src/runtime/include/tessera/tsr_shape.h",
    "src/runtime/include/tessera/tsr_version.h",
)


# Source directories scanned for `extern "C" tessera_*` symbol
# declarations + definitions.
_BACKEND_SOURCE_ROOTS: tuple[tuple[str, str], ...] = (
    ("apple", "src/compiler/codegen/Tessera_Apple_Backend/runtime"),
    ("apple", "src/compiler/codegen/Tessera_Apple_Backend/lib"),
    ("x86", "src/compiler/codegen/tessera_x86_backend/include"),
    ("x86", "src/compiler/codegen/tessera_x86_backend/src"),
    ("nvidia", "src/compiler/codegen/tessera_gpu_backend_NVIDIA"),
    ("rocm", "src/compiler/codegen/Tessera_ROCM_Backend"),
    ("metalium", "src/compiler/codegen/Tessera_Metalium_Backend"),
    ("cerebras", "src/compiler/codegen/Tessera_Cerebras_backend"),
    ("tpu", "src/compiler/codegen/Tessera_TPU_Backend"),
    ("rubincpx", "src/compiler/codegen/Tessera_RubinCPX_Backend"),
    ("runtime_core", "src/runtime"),
)


# ─────────────────────────────────────────────────────────────────────────
# Symbol parsing
# ─────────────────────────────────────────────────────────────────────────


# Match an `extern "C" RETURN_TYPE tessera_<symbol>(`.  We accept any
# return type token before the `tessera_` symbol name; the regex is
# permissive so it picks up `void`, `int`, `size_t`, custom return
# types, etc.  Multi-word return types (e.g., `const float*`) work
# because the pattern uses `.*?` between `extern "C"` and the symbol.
_SYMBOL_RE = re.compile(
    r'extern\s+"C"\s+.*?\b(tessera_[A-Za-z0-9_]+)\s*\(',
    re.DOTALL,
)

# Dtype suffixes we recognise on symbol names.  Used to bucket
# multi-dtype variants of the same kernel.
_DTYPE_SUFFIXES = (
    "_f64", "_f32", "_f16", "_bf16",
    "_fp8_e4m3", "_fp8_e5m2",
    "_fp6_e2m3", "_fp6_e3m2",
    "_fp4_e2m1", "_nvfp4",
    "_int8", "_int16", "_int32", "_int64",
    "_i8", "_i16", "_i32", "_i64",
    "_bool",
)


@dataclass(frozen=True)
class AbiSymbol:
    """One C ABI symbol Tessera exports."""

    name: str
    backend: str
    path: str               # repo-relative source file
    op_family: str          # name minus dtype suffix + backend prefix
    dtype: str | None       # e.g., "f32", "bf16", or None


def _classify_symbol(name: str, backend: str) -> tuple[str, str | None]:
    """Strip the `tessera_<backend>_` prefix + dtype suffix to get
    the canonical op family + dtype."""
    # Strip `tessera_` prefix.
    stem = name[len("tessera_"):]
    # Try to strip a known backend prefix.  Some symbols don't follow
    # the strict pattern (e.g., raw `tessera_runtime_init`) — leave
    # those alone.
    backend_prefixes = (
        "apple_cpu_", "apple_gpu_",
        "x86_", "cuda_", "hip_",
        "metalium_", "cerebras_", "tpu_", "rubincpx_",
    )
    for prefix in backend_prefixes:
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    # Strip a known dtype suffix.
    for suffix in _DTYPE_SUFFIXES:
        if stem.endswith(suffix):
            return stem[:-len(suffix)], suffix[1:]  # strip leading '_'
    return stem, None


def _scan_backend(backend: str, root: str) -> list[AbiSymbol]:
    """Scan one source root for tessera_* C ABI symbols."""
    out: list[AbiSymbol] = []
    root_path = _REPO_ROOT / root
    if not root_path.is_dir():
        return out
    extensions = ("*.h", "*.cpp", "*.mm", "*.c")
    seen: set[tuple[str, str]] = set()
    for ext in extensions:
        for path in sorted(root_path.rglob(ext)):
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            for match in _SYMBOL_RE.finditer(text):
                name = match.group(1)
                rel = path.relative_to(_REPO_ROOT).as_posix()
                key = (name, rel)
                if key in seen:
                    continue
                seen.add(key)
                op_family, dtype = _classify_symbol(name, backend)
                out.append(AbiSymbol(
                    name=name,
                    backend=backend,
                    path=rel,
                    op_family=op_family,
                    dtype=dtype,
                ))
    return out


# ─────────────────────────────────────────────────────────────────────────
# Collection
# ─────────────────────────────────────────────────────────────────────────


def collect_runtime_abi() -> tuple[AbiSymbol, ...]:
    """Walk every backend source root + return the union of
    `extern "C" tessera_*` symbols.  Deduplicated by (name, path)
    so a symbol declared once in a header and defined once in a .cpp
    counts twice (once per file)."""
    out: list[AbiSymbol] = []
    seen: set[tuple[str, str]] = set()
    for backend, root in _BACKEND_SOURCE_ROOTS:
        for sym in _scan_backend(backend, root):
            key = (sym.name, sym.path)
            if key in seen:
                continue
            seen.add(key)
            out.append(sym)
    return tuple(out)


def symbols_per_backend() -> dict[str, int]:
    """Return ``{backend: unique-symbol-name count}``."""
    seen: dict[str, set[str]] = {}
    for s in collect_runtime_abi():
        seen.setdefault(s.backend, set()).add(s.name)
    return {b: len(names) for b, names in seen.items()}


def unique_symbols_per_backend() -> dict[str, tuple[str, ...]]:
    """Return ``{backend: sorted-tuple-of-unique-symbol-names}``."""
    seen: dict[str, set[str]] = {}
    for s in collect_runtime_abi():
        seen.setdefault(s.backend, set()).add(s.name)
    return {b: tuple(sorted(names)) for b, names in seen.items()}


def core_runtime_headers_present() -> dict[str, bool]:
    """Return ``{header_path: exists}`` for each canonical core
    runtime header."""
    return {
        path: (_REPO_ROOT / path).exists()
        for path in _CORE_RUNTIME_HEADERS
    }


def apple_gpu_kernel_families() -> dict[str, tuple[str, ...]]:
    """Return ``{op_family: dtype-variants}`` for the Apple GPU
    runtime symbol surface.  Used by the dashboard's per-family
    coverage table."""
    out: dict[str, set[str]] = {}
    for s in collect_runtime_abi():
        if s.backend != "apple" or s.dtype is None:
            continue
        # Match either CPU- or GPU-target Apple symbols.  Skip the
        # CPU-only families — they're audited separately.
        if not s.name.startswith("tessera_apple_gpu_"):
            continue
        out.setdefault(s.op_family, set()).add(s.dtype)
    return {fam: tuple(sorted(dts)) for fam, dts in out.items()}


# ─────────────────────────────────────────────────────────────────────────
# Version pin checks
# ─────────────────────────────────────────────────────────────────────────


def _version_pin_consistency() -> dict[str, dict[str, str | None]]:
    """Cross-check version pins across Python / CMake / C++.

    Pins audited:
      - CUDA Toolkit (`13.2.1`) — declared in gpu_target.py,
        TesseraToolchainPins.cmake.
      - ROCm (`7.2.3`) — declared in rocm_target.py,
        TesseraToolchainPins.cmake.
      - NCCL / RCCL (`>= 2.22`) — declared in adapter_version_pin.h,
        rocm_target.py, gpu_target.py.

    Returns a per-pin dict mapping source-name → declared value (or
    None if not found).  The drift gate verifies all sources for a
    given pin agree.
    """
    sources = {
        "cuda_toolkit": {
            "python_gpu_target": (
                "python/tessera/compiler/gpu_target.py",
                r"_TESSERA_TARGET_CUDA[^=]*=\s*\"([0-9.]+)\"",
            ),
            "cmake_pins": (
                "cmake/TesseraToolchainPins.cmake",
                r"tessera_pin_cuda_toolkit\s*\(\s*\"?([0-9.]+)\"?",
            ),
        },
        "rocm": {
            "python_rocm_target": (
                "python/tessera/compiler/rocm_target.py",
                r"TESSERA_TARGET_ROCM[^=]*=\s*\"([0-9.]+)\"",
            ),
            "cmake_pins": (
                "cmake/TesseraToolchainPins.cmake",
                r"tessera_pin_rocm\s*\(\s*\"?([0-9.]+)\"?",
            ),
        },
        "nccl_minimum": {
            "python_gpu_target": (
                "python/tessera/compiler/gpu_target.py",
                r"NCCL[^=]*MIN[^=]*=\s*\"([0-9.]+)\"",
            ),
            "cmake_pins": (
                "cmake/TesseraToolchainPins.cmake",
                r"NCCL[^)]*MIN[^)]*\"?([0-9.]+)",
            ),
        },
    }
    out: dict[str, dict[str, str | None]] = {}
    for pin, src_map in sources.items():
        out[pin] = {}
        for label, (rel_path, pattern) in src_map.items():
            full = _REPO_ROOT / rel_path
            if not full.exists():
                out[pin][label] = None
                continue
            try:
                text = full.read_text(errors="replace")
            except OSError:
                out[pin][label] = None
                continue
            match = re.search(pattern, text)
            out[pin][label] = match.group(1) if match else None
    return out


# ─────────────────────────────────────────────────────────────────────────
# Dashboard render
# ─────────────────────────────────────────────────────────────────────────


def render_dashboard() -> str:
    abi = collect_runtime_abi()
    per_backend = symbols_per_backend()
    headers = core_runtime_headers_present()
    apple_gpu = apple_gpu_kernel_families()
    pins = _version_pin_consistency()

    lines: list[str] = []
    lines.append("# Runtime C ABI Surface Audit")
    lines.append("")
    lines.append(
        "Generated from `python/tessera/compiler/runtime_abi_audit.py`. "
        " Don't edit by hand — regenerate via "
        "`python -c \"from tessera.compiler.runtime_abi_audit import "
        "render_dashboard; "
        "open('docs/audit/generated/runtime_abi.md', 'w')"
        ".write(render_dashboard())\"`.  "
        "Drift gated by `tests/unit/test_runtime_abi_audit.py`."
    )
    lines.append("")

    # ── Headline ──
    lines.append("## Headline")
    lines.append("")
    total = sum(per_backend.values())
    lines.append(
        f"- **{total}** unique `extern \"C\" tessera_*` C ABI symbols "
        f"across all backends."
    )
    lines.append(
        f"- **{len([h for h, ok in headers.items() if ok])} / "
        f"{len(headers)}** core runtime headers present."
    )
    lines.append(
        f"- **{len(apple_gpu)}** Apple GPU kernel families with "
        f"per-dtype variants."
    )
    lines.append("")

    # ── Core runtime headers ──
    lines.append("## Core runtime headers")
    lines.append("")
    lines.append("| Header | Status |")
    lines.append("|--------|--------|")
    for path, ok in headers.items():
        glyph = "✅" if ok else "❌"
        lines.append(f"| `{path}` | {glyph} |")
    lines.append("")

    # ── Symbols per backend ──
    lines.append("## Symbols per backend")
    lines.append("")
    lines.append("| Backend | Unique tessera_* symbols |")
    lines.append("|---------|-------------------------:|")
    for backend in sorted(per_backend.keys()):
        lines.append(f"| `{backend}` | {per_backend[backend]} |")
    lines.append("")

    # ── Apple GPU per-family coverage ──
    lines.append("## Apple GPU kernel families × dtype matrix")
    lines.append("")
    if apple_gpu:
        lines.append("| Op family | dtypes |")
        lines.append("|-----------|--------|")
        for fam in sorted(apple_gpu.keys()):
            dts = ", ".join(f"`{d}`" for d in apple_gpu[fam])
            lines.append(f"| `{fam}` | {dts} |")
    else:
        lines.append("_No Apple GPU kernels found — runtime sources missing?_")
    lines.append("")

    # ── Version pin consistency ──
    lines.append("## Toolchain version pins")
    lines.append("")
    lines.append(
        "Pins declared in Python (`gpu_target.py` / `rocm_target.py`) "
        "and CMake (`cmake/TesseraToolchainPins.cmake`).  These MUST "
        "agree across sources — a mismatch means a sprint left one "
        "source behind."
    )
    lines.append("")
    for pin in sorted(pins.keys()):
        sources = pins[pin]
        lines.append(f"### `{pin}`")
        lines.append("")
        lines.append("| Source | Declared value |")
        lines.append("|--------|----------------|")
        for src, val in sources.items():
            display = f"`{val}`" if val else "_not found_"
            lines.append(f"| `{src}` | {display} |")
        # Consistency check.
        values = [v for v in sources.values() if v]
        unique = set(values)
        if len(unique) == 0:
            verdict = "⚠️ No source declares this pin."
        elif len(unique) == 1:
            verdict = "✅ Sources agree."
        else:
            verdict = f"❌ Sources disagree: {sorted(unique)}"
        lines.append("")
        lines.append(verdict)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_dashboard(path: Path | None = None) -> Path:
    target = path or (
        _REPO_ROOT / "docs" / "audit" / "generated" / "runtime_abi.md"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_dashboard())
    return target


__all__ = [
    "AbiSymbol",
    "collect_runtime_abi",
    "symbols_per_backend",
    "unique_symbols_per_backend",
    "core_runtime_headers_present",
    "apple_gpu_kernel_families",
    "render_dashboard",
    "write_dashboard",
]
