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

import numpy as np
import pytest

from tessera import runtime as rt


#: The pass literal as it appears in the SOURCE, where the f-string escapes its
#: braces. Matching the *rendered* form (`{arch=`) is what the first version of
#: this gate did, and it failed against its own target.
_PASS_LITERAL = "lower-tile-to-rocm{{arch={chip}}}"


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


@pytest.mark.skipif(
    not rt._rocm_wmma_runtime_available(),
    reason="no AMD GPU / libtessera_rocm_gemm.so",
)
def test_via_tile_matches_the_production_lane_on_hardware(monkeypatch):
    """The claim the structural gate cannot make: the accumulator survives.

    `via-tile` routes the GEMM through `tile.mma %a, %b, %acc`. If the
    accumulator were dropped in the round trip — the defect found on the NVIDIA
    side, where the WGMMA lowering passes only (A, B) — a K-loop would return
    the last partial product. Bit-identical output is what rules that out; a
    lowering fixture checking emitted ops would not.

    The control matters as much as the comparison: an earlier version of this
    experiment reported bit-identical output while the injection silently never
    applied, so the production lane simply ran twice.
    """
    monkeypatch.setenv("TESSERA_STRICT_DISPATCH", "1")
    real_run = subprocess.run

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
    a = (rng.standard_normal((64, 64)) * 0.4).astype(np.float16)
    b = (rng.standard_normal((64, 64)) * 0.4).astype(np.float16)
    base, _ = launch(None, a, b)
    tiled, hits = launch("generate-wmma-gemm-kernel{via-tile=true},", a, b)

    assert hits == 1, "via-tile injection did not reach the pipeline"
    assert base.get("ok") is True, base.get("reason")
    assert tiled.get("ok") is True, tiled.get("reason")
    assert float(np.max(np.abs(base["output"] - tiled["output"]))) == 0.0, (
        "via-tile diverged from the production lane — the accumulator did not "
        "survive the tile.mma round trip"
    )
