"""Host-free proof that the ROCm compiled lane still produces a code object.

WHY THIS EXISTS (root-caused 2026-08-23, PR #619)
-------------------------------------------------
Every compiled ROCm lane failed at hsaco serialization with
``error: lld invocation failed`` and nothing noticed, because the only
automated ROCm coverage is ``check-tessera-rocm``, which CI does not run.

The outage did not need a GPU to detect. Producing an hsaco is *compile-time*
work: MLIR's ROCDL ``gpu-module-to-binary`` runs ``ld.lld`` on the host and
never touches a device. Measured on a HIP-less ``tessera-rocm-opt`` under
``env -i`` with no AMD toolkit installed, the whole pipeline succeeds against
nothing but a stock ``lld`` -- so this proof runs on an ordinary GPU-less
runner, and would have caught that regression on the first push.

What this does NOT prove: that the code object runs, or computes the right
answer. Execution evidence needs the real device (repo Decision #26) and stays
on the gfx1151 box.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

def _activation(kind: str) -> str:
    """Smallest directive that drives the full generator -> ROCDL -> hsaco path."""
    return (
        "module {\n"
        f'  "tessera_rocm.activation"() {{name = "a", kind = "{kind}", dtype = "f32"}}'
        " : () -> ()\n"
        "}\n"
    )


#: `relu` lowers to `arith.maximumf` -- no call into AMD's OCML device library,
#: so it links against a bare `ld.lld` with no ROCm installed. `gelu`/`silu`
#: reach `__ocml_*` and additionally need `<ROCM_PATH>/amdgcn/bitcode`, which a
#: stock runner does not have (measured 2026-08-23). Keeping the always-on proof
#: on the bitcode-free path is what lets this run on GitHub-hosted CI at all.
_DIRECTIVE = _activation("relu")

_PIPELINE = (
    "builtin.module(tessera-rocm-executable{family=scalar_activation "
    "input=directive output=binary arch=gfx1151 staging=register "
    "tile-q=64 tile-kv=64})"
)


def _rocm_opt() -> Path | None:
    """Locate ``tessera-rocm-opt`` (the standalone ROCm driver).

    Deliberately NOT ``tessera-opt``: on a HIP-less host
    ``TESSERA_BUILD_ROCM_BACKEND=ON`` forces the lean artifact driver
    (``tools/tessera-opt/CMakeLists.txt``), which drops core TesseraIR and the
    x86 Target IR. ``tessera-rocm-opt`` is a separate executable with no HIP
    dependency, so it carries the ROCm dialect on a runner that has neither a
    GPU nor ROCm installed."""
    env = os.environ.get("TESSERA_ROCM_OPT")
    if env and Path(env).is_file():
        return Path(env)
    rel = "src/compiler/codegen/Tessera_ROCM_Backend/tools/tessera-rocm-opt"
    for build in ("build-rocm-ci", "build"):
        cand = REPO_ROOT / build / rel
        if cand.is_file():
            return cand
    return None


def _host_lld() -> Path | None:
    """Any ``ld.lld`` on this host -- the ROCm toolkit's or a stock LLVM's.

    The serializer only needs *a* linker that understands the AMDGPU target;
    the stock apt.llvm.org ``lld-23`` is sufficient (measured)."""
    for cand in (
        "/opt/rocm/core/lib/llvm/bin/ld.lld",
        "/opt/rocm/core/llvm/bin/ld.lld",
        "/opt/rocm/llvm/bin/ld.lld",
        "/usr/lib/llvm-23/bin/ld.lld",
    ):
        if Path(cand).is_file():
            return Path(cand)
    found = shutil.which("ld.lld")
    return Path(found) if found else None


_OPT = _rocm_opt()
_LLD = _host_lld()

pytestmark = [
    pytest.mark.skipif(_OPT is None, reason="tessera-rocm-opt not built"),
    pytest.mark.skipif(_LLD is None, reason="no ld.lld on this host (install lld-23)"),
]


def _toolkit_shim(tmp_path: Path) -> Path:
    """A minimal ROCM_PATH: just ``<root>/llvm/bin/ld.lld``.

    Using a shim rather than the real toolkit is the point -- it proves the
    serializer needs a linker and not an installed ROCm or a device."""
    root = tmp_path / "rocm-shim"
    lld_dir = root / "llvm" / "bin"
    lld_dir.mkdir(parents=True, exist_ok=True)
    link = lld_dir / "ld.lld"
    if not link.exists():
        link.symlink_to(_LLD)
    return root


def _serialize(env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(_OPT), "-", f"--pass-pipeline={_PIPELINE}"],
        input=_DIRECTIVE,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )


def test_rocm_lane_serializes_an_hsaco_without_a_gpu(tmp_path: Path) -> None:
    """The compiled ROCm lane emits a real AMDGPU ELF on a GPU-less host."""
    env = dict(os.environ)
    env["ROCM_PATH"] = str(_toolkit_shim(tmp_path))
    proc = _serialize(env)

    assert proc.returncode == 0, f"serialization failed:\n{proc.stderr[:2000]}"
    assert "gpu.binary" in proc.stdout, "pipeline produced no gpu.binary"

    from tessera.runtime import _extract_hsaco_blob

    blob = _extract_hsaco_blob(proc.stdout)
    assert blob[:4] == b"\x7fELF", f"gpu.binary is not an ELF: {blob[:16]!r}"
    # e_machine 0xE0 == EM_AMDGPU: the object is for the GPU, not the host.
    assert blob[18] == 0xE0, f"ELF e_machine is not EM_AMDGPU: {blob[18]:#x}"


def test_missing_rocm_path_is_the_diagnosed_failure(tmp_path: Path) -> None:
    """Negative control: without ROCM_PATH the lane fails as PR #619 saw it.

    This is what makes the positive test meaningful -- it shows the check can
    actually observe the regression rather than passing for an unrelated
    reason. Note MLIR does NOT fall back to ``PATH`` for ``ld.lld``: measured,
    the serializer still fails with ld.lld on PATH but ROCM_PATH unset, which
    is exactly why a non-interactive shell broke every compiled lane."""
    env = {k: v for k, v in os.environ.items() if k not in ("ROCM_PATH", "HIP_PATH")}
    proc = _serialize(env)

    assert proc.returncode != 0, (
        "expected serialization to fail without ROCM_PATH; if this now passes, "
        "MLIR gained a fallback and the production env plumbing in "
        "runtime.py/_rocm_serializer_env may no longer be required"
    )
    assert "lld invocation failed" in proc.stderr


def _device_bitcode_root() -> Path | None:
    """A ROCm root that actually ships `amdgcn/bitcode` (OCML/OCKL), or None."""
    for cand in (Path("/opt/rocm/core"), Path("/opt/rocm")):
        if (cand / "amdgcn" / "bitcode").is_dir():
            return cand
    return None


@pytest.mark.skipif(
    _device_bitcode_root() is None,
    reason="no ROCm device bitcode (amdgcn/bitcode) on this host",
)
def test_ocml_calling_kernel_serializes_where_bitcode_exists() -> None:
    """The OCML path, proven only where the device libraries are installed.

    Deliberately separate from the always-on proof above and honestly gated:
    `gelu` lowers to `__ocml_tanh_f32`, so it needs AMD's bitcode and CANNOT be
    covered on a stock runner. Stating that as a skip keeps the CI lane from
    implying more coverage than it has (repo Decision #26)."""
    env = dict(os.environ)
    env["ROCM_PATH"] = str(_device_bitcode_root())
    proc = subprocess.run(
        [str(_OPT), "-", f"--pass-pipeline={_PIPELINE}"],
        input=_activation("gelu"),
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert proc.returncode == 0, f"OCML serialization failed:\n{proc.stderr[:2000]}"
    assert "gpu.binary" in proc.stdout
