"""The non-Darwin stub must keep pace with the real Apple GPU runtime.

`apple_gpu_runtime_stub.cpp` is the off-Darwin half of the Apple C ABI: same
symbols, host compute instead of Metal. It is what keeps a Linux build linkable
and the Python ctypes layer platform-agnostic.

It has drifted. 51 symbols defined in `apple_gpu_runtime.mm` have no stub
counterpart, because stubbing happens per feature and each sprint stubs what it
remembers. The consequence is bounded — `getattr(runtime, name, None)` returns
None for a missing symbol and the caller falls back to the Python reference, so
results stay correct — but off-Darwin those lanes silently lose the C++
reference the Darwin lane is matched against.

This is a ratchet, not a pass/fail on the current gap: `KNOWN_UNSTUBBED` is
allowed to shrink and never to grow. A new symbol added to the `.mm` without a
stub fails immediately; closing an existing one means deleting its line here.
Do not add to this list to make a test pass — write the stub.
"""

from __future__ import annotations

import re
from pathlib import Path


RUNTIME = (Path(__file__).resolve().parents[2]
           / "src/compiler/codegen/Tessera_Apple_Backend/runtime")
REAL = RUNTIME / "apple_gpu_runtime.mm"
STUB = RUNTIME / "apple_gpu_runtime_stub.cpp"

#: `extern "C" <ret> <name>(<args>) {` — a definition, not a declaration.
_DEFINITION = re.compile(
    r'^extern\s+"C"\s+[A-Za-z_][\w:<>*\s]*?\s+\**\s*(\w+)\s*\([^;{]*?\)\s*\{',
    re.M | re.S,
)

#: A `#define NAME(...)` and its backslash-continued body.
_MACRO_BLOCK = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(\w+)\((?:[^)]*)\)(.*?)(?<!\\)$",
                          re.M | re.S)

#: Symbols only a runtime C ABI would export; used to read macro arguments.
_ABI_NAME = re.compile(r"\b((?:tessera_apple|ts)_\w+)\b")

#: Real-runtime symbols with no stub as of 2026-07-28. Shrink only.
KNOWN_UNSTUBBED = {
    # Clifford cl30 geometric-algebra family.
    "tessera_apple_gpu_clifford_codiff_cl30_f32",
    "tessera_apple_gpu_clifford_conjugate_cl30_f32",
    "tessera_apple_gpu_clifford_exp_cl30_f32",
    "tessera_apple_gpu_clifford_ext_deriv_cl30_f32",
    "tessera_apple_gpu_clifford_geo_product_cl30_bf16",
    "tessera_apple_gpu_clifford_geo_product_cl30_f16",
    "tessera_apple_gpu_clifford_geo_product_cl30_f32",
    "tessera_apple_gpu_clifford_grade_involution_cl30_f32",
    "tessera_apple_gpu_clifford_grade_projection_cl30_f32",
    "tessera_apple_gpu_clifford_hodge_star_cl30_f32",
    "tessera_apple_gpu_clifford_inner_cl30_f32",
    "tessera_apple_gpu_clifford_integral_cl30_f32",
    "tessera_apple_gpu_clifford_left_contraction_cl30_f32",
    "tessera_apple_gpu_clifford_log_cl30_f32",
    "tessera_apple_gpu_clifford_norm_cl30_f32",
    "tessera_apple_gpu_clifford_norm_squared_cl30_f32",
    "tessera_apple_gpu_clifford_reverse_cl30_f32",
    "tessera_apple_gpu_clifford_rotor_sandwich_cl30_bf16",
    "tessera_apple_gpu_clifford_rotor_sandwich_cl30_f16",
    "tessera_apple_gpu_clifford_rotor_sandwich_cl30_f32",
    "tessera_apple_gpu_clifford_rotor_sandwich_norm_cl30_f32",
    "tessera_apple_gpu_clifford_vec_deriv_cl30_f32",
    "tessera_apple_gpu_clifford_wedge_cl30_f32",
    # Complex / conformal.
    "tessera_apple_gpu_complex_exp_f32",
    "tessera_apple_gpu_complex_mobius_f32",
    "tessera_apple_gpu_complex_mul_f32",
    "tessera_apple_gpu_complex_stereographic_f32",
    # Encode-session conv2d wrappers (one-command-buffer substrate).
    "tessera_apple_gpu_conv2d_dev_bf16_enc",
    "tessera_apple_gpu_conv2d_dev_f16_enc",
    "tessera_apple_gpu_conv2d_dev_f32_enc",
    # EBM value lane.
    "tessera_apple_gpu_ebm_decode_init_noise_apply_f32",
    "tessera_apple_gpu_ebm_ebt_tiny_refinement_argmin_f32",
    "tessera_apple_gpu_ebm_energy_quadratic_f32",
    "tessera_apple_gpu_ebm_inner_step_f32",
    "tessera_apple_gpu_ebm_langevin_step_f32",
    "tessera_apple_gpu_ebm_langevin_step_philox_f32",
    "tessera_apple_gpu_ebm_partition_exact_f32",
    "tessera_apple_gpu_ebm_refinement_f32",
    "tessera_apple_gpu_ebm_self_verify_hard_argmin_f32",
    "tessera_apple_gpu_ebm_sphere_langevin_step_f32",
    # MPSGraph cache instrumentation.
    "tessera_apple_gpu_mpsgraph_cache_capacity",
    "tessera_apple_gpu_mpsgraph_cache_evictions",
    # Philox RNG lane (the EBM Langevin Philox step is grouped above).
    "tessera_apple_gpu_philox_dropout_f32",
    "tessera_apple_gpu_philox_normal_f32",
    "tessera_apple_gpu_philox_uniform_f32",
    # Spiking conv2d.
    "tessera_apple_gpu_spike_conv2d_multi_tile_f16",
    "tessera_apple_gpu_spike_conv2d_single_tile_f16",
    # Sparse / scatter / optimizer entry points.
    "tessera_apple_gpu_optimizer_f32",
    "tessera_apple_gpu_scatter_f32",
    "tessera_apple_gpu_sddmm_f32",
    "tessera_apple_gpu_spmm_csr_f32",
}


def _definitions(path: Path) -> set[str]:
    """C ABI symbols this translation unit defines, literal or macro-generated.

    Both files define some entry points through a `#define` whose parameter is
    the symbol name, so a naive scan reports the *parameter* (`NAME`, `name`) as
    a symbol and misses the real ones. Macro bodies are removed and their
    invocations read instead — without that the guard both invents symbols and
    overlooks stubs that exist, which is worse than no guard at all.
    """
    text = path.read_text(encoding="utf-8")

    generators: set[str] = set()
    for match in _MACRO_BLOCK.finditer(text):
        if 'extern "C"' in match.group(2):
            generators.add(match.group(1))
    body_free = _MACRO_BLOCK.sub("", text)

    symbols = {m.group(1) for m in _DEFINITION.finditer(body_free)}
    for macro in generators:
        for invocation in re.finditer(rf"\b{macro}\s*\(([^;]*?)\)", body_free, re.S):
            symbols.update(_ABI_NAME.findall(invocation.group(1)))
    return symbols


def test_no_new_apple_runtime_symbol_lands_without_a_stub() -> None:
    real, stub = _definitions(REAL), _definitions(STUB)
    assert real, "parsed no definitions from apple_gpu_runtime.mm — check the regex"
    new_gaps = sorted(real - stub - KNOWN_UNSTUBBED)
    assert not new_gaps, (
        "these symbols are defined in apple_gpu_runtime.mm with no counterpart "
        "in apple_gpu_runtime_stub.cpp, so an off-Darwin build loses their C++ "
        "reference: " + ", ".join(new_gaps) + ". Write the stub — do not add "
        "them to KNOWN_UNSTUBBED."
    )


def test_the_unstubbed_allowlist_only_shrinks() -> None:
    """A closed gap must leave the list, or the ratchet stops ratcheting."""
    real, stub = _definitions(REAL), _definitions(STUB)
    now_stubbed = sorted(KNOWN_UNSTUBBED & stub)
    assert not now_stubbed, (
        "these now have stubs and must be removed from KNOWN_UNSTUBBED: "
        + ", ".join(now_stubbed)
    )
    gone = sorted(KNOWN_UNSTUBBED - real)
    assert not gone, (
        "these are in KNOWN_UNSTUBBED but no longer defined in the real "
        "runtime; drop them: " + ", ".join(gone)
    )


def test_stub_defines_nothing_the_real_runtime_does_not() -> None:
    """A stub-only symbol is a Darwin gap — the ABI would differ by platform."""
    real, stub = _definitions(REAL), _definitions(STUB)
    orphans = sorted(stub - real)
    assert not orphans, (
        "apple_gpu_runtime_stub.cpp defines symbols the real runtime does not, "
        "so the C ABI differs by platform: " + ", ".join(orphans)
    )
