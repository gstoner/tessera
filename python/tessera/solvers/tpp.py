"""``tessera.solvers.tpp`` — Python-side surface for the Tensor
Parallel Primitives (TPP) solver dialect.

The dialect itself lives in C++ MLIR under ``src/solvers/tpp/`` with:

  - **Ops & types** (``Dialect/TPP/``): ``!tpp.field``, ``!tpp.mesh``,
    ``#tpp.units``, ``#tpp.bc``; stencil + boundary-condition ops.
  - **Passes** (``lib/Passes/``): ``-tpp-halo-infer``,
    ``-tpp-legalize-space-time``, ``-tpp-fuse-stencil-time``,
    ``-tpp-async-prefetch``, ``-tpp-vectorize``,
    ``-tpp-distribute-halo``, ``-lower-tpp-to-target-ir``.
  - **Pipeline alias** (``lib/Passes/PassPipeline.cpp``):
    ``tpp-space-time`` chains all seven passes for the canonical
    space-time stencil lowering path.
  - **Lit fixtures** (``test/TPP/``): halo_infer.mlir,
    shallow_water_smoke.mlir, bc_lowering.mlir, pipeline_alias.mlir.

This Python module surfaces the **pass-pipeline alias names** and the
**dialect metadata** so Python tooling (audit dashboards, JIT routing,
support tables) can reason about TPP without linking against
``tessera-opt``.  Actual lowering / execution requires
``tessera-opt`` built against MLIR 21.

Status: **dispatch wired via `tessera-opt` (2026-05-18).** TPP now
registers its dialect + all 7 individual passes + the
``tpp-space-time`` pipeline alias into ``tessera-opt``.  All 4 lit
fixtures under ``src/solvers/tpp/test/TPP/`` pass.  Python
``solve(...)`` driver still routes through subprocess into
``tessera-opt`` rather than via embedded MLIR bindings — that
remains a follow-up.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


#: Stable name of the canonical TPP pass-pipeline alias as
#: registered in ``src/solvers/tpp/lib/Passes/PassPipeline.cpp``.
TPP_PIPELINE_ALIAS: str = "tpp-space-time"


#: Names of every individual pass exposed by the TPP solver.  Order
#: matches the chain in the ``tpp-space-time`` alias.
TPP_PASS_NAMES: tuple[str, ...] = (
    "tpp-legalize-space-time",
    "tpp-halo-infer",
    "tpp-fuse-stencil-time",
    "tpp-async-prefetch",
    "tpp-vectorize",
    "tpp-distribute-halo",
    "lower-tpp-to-target-ir",
)


#: Type names defined by the TPP dialect (without the ``!`` sigil).
TPP_TYPE_NAMES: tuple[str, ...] = (
    "tpp.field",
    "tpp.mesh",
)


#: Attribute names defined by the TPP dialect (without ``#``).
TPP_ATTR_NAMES: tuple[str, ...] = (
    "tpp.units",
    "tpp.bc",
)


# ── Embedded-MLIR driver (Glass-jaw #2, 2026-06-01) ──────────────────
# The `tessera_tpp_capi` shared library (src/solvers/tpp/lib/TPPCApi.cpp)
# runs the `tpp-space-time` pipeline IN-PROCESS over an MLIR text module.
# ``solve()`` loads it via ctypes — no `tessera-opt` subprocess, no PATH
# dependency. Mirrors the Apple GPU runtime ctypes loader pattern.

_CAPI: ctypes.CDLL | None = None
_CAPI_TRIED = False


def _candidate_lib_paths() -> list[Path]:
    """Build-tree locations where ``libtessera_tpp_capi`` may live."""
    suffix = ".dylib" if os.uname().sysname == "Darwin" else ".so"
    name = f"libtessera_tpp_capi{suffix}"
    # repo root is three parents up from this file
    # (python/tessera/solvers/tpp.py → repo/).
    repo = Path(__file__).resolve().parents[3]
    out: list[Path] = []
    env = os.environ.get("TESSERA_TPP_CAPI")
    if env:
        out.append(Path(env))
    out.append(repo / "build" / "src" / "solvers" / "tpp" / "lib" / name)
    # Fallback: glob any build* tree.
    out.extend(repo.glob(f"build*/**/{name}"))
    return out


def _load_capi() -> ctypes.CDLL | None:
    """Lazily load + bind the TPP C ABI shared library. Returns None
    when it isn't built (Python-only checkout / no MLIR build)."""
    global _CAPI, _CAPI_TRIED
    if _CAPI_TRIED:
        return _CAPI
    _CAPI_TRIED = True
    for p in _candidate_lib_paths():
        if not p.is_file():
            continue
        try:
            lib = ctypes.CDLL(str(p))
            lib.tessera_tpp_capi_available.restype = ctypes.c_int
            lib.tessera_tpp_run_pipeline.argtypes = [
                ctypes.c_char_p, ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_char_p)]
            lib.tessera_tpp_run_pipeline.restype = ctypes.c_int
            lib.tessera_tpp_free.argtypes = [ctypes.c_char_p]
            lib.tessera_tpp_free.restype = None
            if int(lib.tessera_tpp_capi_available()) == 1:
                _CAPI = lib
                return _CAPI
        except OSError:
            continue
    return None


def embedded_driver_available() -> bool:
    """True iff the in-process TPP pipeline (ctypes lib) is loadable."""
    return _load_capi() is not None


def solve(input_mlir: str, *, pipeline: Optional[str] = None) -> str:
    """Run a TPP pass pipeline over ``input_mlir`` IN-PROCESS and return
    the resulting MLIR module text.

    Glass-jaw #2 (2026-06-01): this dispatches through the embedded
    ``tessera_tpp_capi`` ctypes library — NOT a ``tessera-opt``
    subprocess. ``pipeline`` defaults to the canonical
    ``builtin.module(tpp-space-time)`` alias; pass an explicit
    pass-pipeline string to run a custom chain.

    Raises:
      RuntimeError — if the embedded library isn't built (run
        ``ninja -C build tessera_tpp_capi``), or the parse / pipeline /
        run step fails (the C ABI error message is surfaced).
    """
    lib = _load_capi()
    if lib is None:
        raise RuntimeError(
            "TPP embedded driver unavailable: build it with "
            "`ninja -C build tessera_tpp_capi` (or set TESSERA_TPP_CAPI "
            "to the built libtessera_tpp_capi path).")
    pipe = (pipeline or f"builtin.module({TPP_PIPELINE_ALIAS})").encode()
    out = ctypes.c_char_p()
    rc = lib.tessera_tpp_run_pipeline(
        input_mlir.encode(), pipe, ctypes.byref(out))
    try:
        result = out.value.decode() if out.value else ""
    finally:
        if out.value is not None:
            lib.tessera_tpp_free(out)
    if int(rc) != 0:
        raise RuntimeError(f"TPP solve failed (rc={rc}): {result}")
    return result


@dataclass(frozen=True)
class TPPStatus:
    """One-shot snapshot of TPP's current build/wiring state."""
    dialect_present: bool
    passes_present: bool
    pipeline_alias_present: bool
    python_driver_wired: bool
    lit_fixtures_runnable: bool
    notes: str


def status() -> TPPStatus:
    """Report what's wired and what isn't.

    The C++ side ships and is registered into ``tessera-opt``;
    all 4 lit fixtures pass.  ``python_driver_wired`` records
    whether the Python ``solve(...)`` driver dispatches via
    embedded MLIR bindings (not yet — still goes through
    subprocess invocations of ``tessera-opt``).
    """
    embedded = embedded_driver_available()
    return TPPStatus(
        dialect_present=True,
        passes_present=True,
        pipeline_alias_present=True,
        python_driver_wired=embedded,
        lit_fixtures_runnable=True,
        notes=(
            "TPP wired into tessera-opt 2026-05-18; dialect + 7 passes + "
            "tpp-space-time alias all registered; 4/4 lit fixtures pass. "
            + (
                "Glass-jaw #2 (2026-06-01): `solve(...)` now runs the "
                "pipeline IN-PROCESS via the `tessera_tpp_capi` ctypes "
                "library — no tessera-opt subprocess."
                if embedded else
                "Embedded driver (`tessera_tpp_capi`) not built in this "
                "checkout; build it with `ninja -C build tessera_tpp_capi` "
                "to enable in-process `solve(...)`."
            )
        ),
    )


def pipeline_command(input_mlir: str) -> list[str]:
    """Construct the ``tessera-opt`` command line for the canonical
    TPP pipeline alias.

    Returns a list suitable for ``subprocess.run([...])`` *once*
    ``tessera-opt`` exists on ``PATH``.  Until then this helper
    documents the expected invocation::

        tessera-opt --pass-pipeline="builtin.module(tpp-space-time)" input.mlir

    Parameters
    ----------
    input_mlir
        Path to the input MLIR file containing TPP ops.
    """
    return [
        "tessera-opt",
        f"--pass-pipeline=builtin.module({TPP_PIPELINE_ALIAS})",
        input_mlir,
    ]


__all__ = [
    "TPP_PIPELINE_ALIAS",
    "TPP_PASS_NAMES",
    "TPP_TYPE_NAMES",
    "TPP_ATTR_NAMES",
    "TPPStatus",
    "status",
    "pipeline_command",
    "solve",
    "embedded_driver_available",
]
