"""The compiled ROCm pipeline must be able to lower a `tile.mma`.

`GenerateWMMAGemmKernel`'s `via-tile` option emits `tile.mma %a, %b, %acc` at
the Tile-IR seam instead of `tessera_rocm.wmma`, and its comment says the op
"flows through rocm-wave-lds-pipeline + lower-tile-to-rocm, which lowers it back
to tessera_rocm.wmma with the SAME (a, b, acc) operands".

Measured 2026-08-04: the runtime's compiled-matmul pipeline contained **no**
`lower-tile-to-rocm`. `tile.mma` therefore survived to LLVM translation and the
build died with

    cannot be converted to LLVM IR: missing `LLVMTranslationDialectInterface`
    registration for dialect for op: tile.mma

so via-tile was unreachable in production and W1.1's Tile-IR seam could not be
exercised by the lane that actually executes.

Two gates, because they fail for different reasons:

  * the STRUCTURAL one runs everywhere and pins the pipeline composition;
  * the NUMERIC one needs an AMD GPU and proves the accumulator survives the
    round trip rather than merely that the pass is present.
"""

from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tessera import runtime as rt


#: The pass literal as it appears in the SOURCE, where the f-string escapes its
#: braces. Matching the *rendered* form (`{arch=`) is what the first version of
#: this gate did, and it failed against its own target.
_PASS_LITERAL = "lower-tile-to-rocm{{arch={chip}}}"
_REPO = Path(__file__).resolve().parents[2]
_TESSERA_OPT = _REPO / "build" / "tools" / "tessera-opt" / "tessera-opt"
_GEMM_DIRECTIVE = (
    'module { "tessera_rocm.wmma_gemm"() {name = "gemm", '
    'm = 16 : i64, n = 16 : i64, k = 16 : i64, mt = 2 : i64, '
    'nt = 4 : i64, dtype = "f16"} : () -> () }\n'
)


def _run_opt(pipeline: str, source: str = _GEMM_DIRECTIVE):
    if not _TESSERA_OPT.is_file():
        pytest.skip("tessera-opt not built")
    return subprocess.run(
        [str(_TESSERA_OPT), "-", f"--pass-pipeline={pipeline}"],
        input=source,
        capture_output=True,
        text=True,
    )


def test_every_wmma_gemm_pipeline_can_lower_tile_mma():
    """Structural: the tile lowering follows the generator in every ROCm lane.

    Both the plain and the canonical pipeline were missing it. Counting rather
    than checking presence, so adding a third lane without the pass fails here
    instead of silently shipping a lane where `tile.mma` reaches LLVM
    translation.
    """
    src = inspect.getsource(rt)
    generators = src.count('"generate-wmma-gemm-kernel,"') + \
        src.count('f"generate-wmma-gemm-kernel{{canonical-staging={staging}}},"')
    lowerings = src.count(_PASS_LITERAL)
    assert lowerings == generators, (
        f"{generators} wmma-gemm pipeline(s) but {lowerings} tile lowering(s); "
        "a lane without it emits tile.mma that dies at LLVM translation"
    )
    assert lowerings >= 2, "expected the plain and canonical lanes"


def test_the_tile_lowering_always_states_its_arch():
    """`lower-tile-to-rocm` defaults to a CDNA part.

    Without `arch=`, it emits `llvm.amdgcn.mfma.contract` — an MFMA intrinsic
    that is wrong for RDNA 3.5 (gfx1151 has WMMA, no MFMA) and does not resolve,
    so the build fails at translation with an unreferenced symbol. The pipeline
    string would still be syntactically valid, which is why this is a gate and
    not a comment.
    """
    src = inspect.getsource(rt)
    for bad in ('"lower-tile-to-rocm,"', '"lower-tile-to-rocm)"'):
        assert bad not in src, f"tile lowering used without arch=: {bad}"


def test_gfx1151_typed_contract_materializes_the_direct_physical_body():
    """The Tile artifact is real, then its target body has one owner."""
    generated = _run_opt(
        "builtin.module(generate-wmma-gemm-kernel{via-tile=true})"
    )
    assert generated.returncode == 0, generated.stderr
    assert "tessera.rocm.typed_gfx11_gemm_contract" in generated.stdout
    assert generated.stdout.count("tile.fragment_pack") == 24
    assert generated.stdout.count("tile.mma ") == 32
    assert generated.stdout.count("tile.fragment_unpack") == 16
    assert generated.stdout.count("tile.store ") == 16

    for dtype in ("f16", "bf16"):
        source = _GEMM_DIRECTIVE.replace('dtype = "f16"', f'dtype = "{dtype}"')
        direct = _run_opt(
            "builtin.module(generate-wmma-gemm-kernel,"
            "lower-tile-to-rocm{arch=gfx1151})",
            source,
        )
        typed = _run_opt(
            "builtin.module(generate-wmma-gemm-kernel{via-tile=true},"
            "lower-tile-to-rocm{arch=gfx1151})",
            source,
        )
        assert direct.returncode == typed.returncode == 0
        assert typed.stdout == direct.stdout


def test_gfx1151_typed_physical_contract_fails_closed_when_tampered():
    generated = _run_opt(
        "builtin.module(generate-wmma-gemm-kernel{via-tile=true})"
    )
    assert generated.returncode == 0, generated.stderr
    tampered = generated.stdout.replace(
        "tessera.rocm.physical_panel_mt = 2 : i64",
        "tessera.rocm.physical_panel_mt = 3 : i64",
        1,
    )
    assert tampered != generated.stdout
    lowered = _run_opt(
        "builtin.module(lower-tile-to-rocm{arch=gfx1151})", tampered
    )
    assert lowered.returncode != 0
    assert "failed to materialize the shared gfx11 GEMM body" in lowered.stderr

    # Counts and types still agree after swapping one same-typed A fragment;
    # only exact body identity can detect this valid-looking semantic change.
    operand_tampered = generated.stdout.replace(
        "tile.mma %60, %67", "tile.mma %63, %67", 1
    )
    assert operand_tampered != generated.stdout
    lowered = _run_opt(
        "builtin.module(lower-tile-to-rocm{arch=gfx1151})",
        operand_tampered,
    )
    assert lowered.returncode != 0
    assert "Tile-body digest mismatch" in lowered.stderr


@pytest.mark.skipif(
    not rt._rocm_wmma_runtime_available(),
    reason="no AMD GPU / libtessera_rocm_gemm.so",
)
def test_via_tile_matches_the_production_lane_on_hardware(monkeypatch):
    """The claim the structural gate cannot make: the accumulator survives.

    `via-tile` routes the production 2x4 macro tile through the complete typed
    `tile.view -> fragment_pack -> tile.mma -> fragment_unpack -> tile.store`
    chain. If addressing, masking, accumulator threading, or stores diverge,
    bit-identical output fails; a lowering fixture checking emitted ops would
    not prove any of those properties.

    The control matters as much as the comparison: an earlier version of this
    experiment reported bit-identical output while the injection silently never
    applied, so the production lane simply ran twice.
    """
    monkeypatch.setenv("TESSERA_STRICT_DISPATCH", "1")
    real_run = subprocess.run

    # PR #508 review — without this the comparison proves nothing.
    #
    # `_execute_rocm_compiled_gemm` is `try: _rocm_compiled_gemm_impl(...)
    # except _RocmCompiledUnavailable: _execute_rocm_wmma_artifact(...)`, and a
    # module-load failure (e.g. the live chip differs from `_rocm_chip()`'s
    # default) raises exactly that. It is ENVELOPE-class by deliberate design,
    # so TESSERA_STRICT_DISPATCH does not reject it -- that classification is
    # what keeps strict runs working on CPU-only hosts.
    #
    # The consequence here is sharp: both launches below could execute the same
    # hand-written oracle and be TRIVIALLY bit-identical while no tile.mma
    # hsaco ever ran. That is the "ran the same lane twice" failure this file's
    # own control was written to catch, arriving through a second door.
    #
    # So make the fallback fatal for the duration of the test: if the compiled
    # lane degrades for any reason, the test says so instead of passing.
    def _no_fallback(*_args, **_kwargs):
        raise AssertionError(
            "the compiled ROCm lane degraded to the hand-written WMMA oracle; "
            "the tile.mma hsaco never ran, so a bit-identical comparison would "
            "be vacuous"
        )

    monkeypatch.setattr(rt, "_execute_rocm_wmma_artifact", _no_fallback)

    def launch(inject, a, b):
        rt._rocm_compiled_hsaco_cache.clear()
        hits = [0]

        def patched(cmd, *args, **kwargs):
            if isinstance(cmd, list) and inject:
                out = []
                for c in cmd:
                    if isinstance(c, str) and "generate-wmma-gemm-kernel," in c:
                        c = c.replace("generate-wmma-gemm-kernel,", inject)
                        hits[0] += 1
                    out.append(c)
                cmd = out
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr(subprocess, "run", patched)
        try:
            artifact = rt.RuntimeArtifact(metadata={
                "target": "rocm", "compiler_path": "rocm_compiled",
                "executable": True, "execution_kind": "native_gpu",
                "arg_names": ["a", "b"], "output_name": "c",
                "ops": [{"op_name": "tessera.matmul", "result": "c",
                         "operands": ["a", "b"], "kwargs": {}}],
            })
            return rt.launch(artifact, (a, b)), hits[0]
        finally:
            monkeypatch.setattr(subprocess, "run", real_run)

    zeros = np.zeros((64, 64), dtype=np.float16)
    control, _ = launch("generate-wmma-gemm-kernel{not-a-real-option=true},",
                        zeros, zeros)
    assert control.get("ok") is False, (
        "the harness cannot fail, so a match below would prove nothing"
    )

    rng = np.random.default_rng(21)
    # The second shape crosses every dynamic-address boundary: ragged M/N make
    # the edge stores and A/B bounds live, while ragged K executes the scalar
    # guarded tail pack. The aligned case remains the production 2x4 baseline.
    for m, n, k in ((64, 64, 64), (65, 67, 31)):
        a = (rng.standard_normal((m, k)) * 0.4).astype(np.float16)
        b = (rng.standard_normal((k, n)) * 0.4).astype(np.float16)
        base, _ = launch(None, a, b)
        tiled, hits = launch(
            "generate-wmma-gemm-kernel{via-tile=true},", a, b
        )

        assert hits == 1, "via-tile injection did not reach the pipeline"
        assert base.get("ok") is True, base.get("reason")
        assert tiled.get("ok") is True, tiled.get("reason")
        assert float(np.max(np.abs(base["output"] - tiled["output"]))) == 0.0, (
            f"via-tile diverged from production at ragged shape {m}x{n}x{k}"
        )
