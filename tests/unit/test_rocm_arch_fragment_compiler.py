"""ROCM-5 compiler/assembler ratchets for architecture-owned fragments."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from tests._support.environment import CompilerToolchain


pytestmark = pytest.mark.compiler_tool


REPO = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO / "src/compiler/codegen/Tessera_ROCM_Backend/test/rocm"
    / "architecture_tile_fragment_store.mlir"
)


def _lower(
    tools: CompilerToolchain,
    arch: str,
    source: str | None = None,
    *,
    generic: bool = False,
) -> str:
    tessera_opt = tools.require_tessera_opt()
    command = [
        str(tessera_opt), "-" if source is not None else str(FIXTURE),
        "--allow-unregistered-dialect",
    ]
    if generic:
        command.append("--mlir-print-op-generic")
    command.append(
            "--pass-pipeline=builtin.module("
            f"lower-tile-to-rocm{{arch={arch}}},lower-tessera-target-to-rocdl)",
    )
    result = subprocess.run(
        command,
        input=source, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _typed_source(dtype: str) -> str:
    source = FIXTURE.read_text()
    if dtype == "bf16":
        return source.replace("f16", "bf16")
    if dtype in ("e4m3", "e5m2"):
        mlir_dtype = "f8E4M3FN" if dtype == "e4m3" else "f8E5M2"
        return source.replace(
            "memref<256xf16>", f"memref<256x{mlir_dtype}>"
        ).replace(
            'a = "f16", b = "f16"', f'a = "{dtype}", b = "{dtype}"'
        )
    if dtype in ("int8", "int4"):
        source = source.replace("memref<256xf16>", "memref<256xi8>")
        source = source.replace("memref<256xf32>", "memref<256xi32>")
        source = source.replace(
            'a = "f16", b = "f16", acc = "f32"',
            f'a = "{dtype}", b = "{dtype}", acc = "i32"',
        )
        if dtype == "int4":
            source = source.replace("memref<256xi8>", "memref<512xi8>")
            source = source.replace("k = 16", "k = 32")
            old_layout = "shard = [16, 16] : [16, 1]"
            source = source.replace(
                old_layout, "shard = [16, 32] : [32, 1]", 1)
            source = source.replace(
                old_layout, "shard = [32, 16] : [16, 1]", 1)
            source = source.replace("leading_dim = 16", "leading_dim = 32", 2)
        return source
    raise AssertionError(dtype)


def _gfx125x_source(dtype: str) -> str:
    source = FIXTURE.read_text()
    if dtype == "bf16":
        source = source.replace("f16", "bf16")
    mlir_dtype = "bf16" if dtype == "bf16" else "f16"
    source = source.replace(
        f"memref<256x{mlir_dtype}>", f"memref<512x{mlir_dtype}>")
    source = source.replace("k = 16", "k = 32")
    old_layout = "shard = [16, 16] : [16, 1]"
    source = source.replace(
        old_layout, "shard = [16, 32] : [32, 1]", 1)
    source = source.replace(
        old_layout, "shard = [32, 16] : [16, 1]", 1)
    source = source.replace("leading_dim = 16", "leading_dim = 32", 2)
    return source


def _serialize(tools: CompilerToolchain, lowered: str, arch: str) -> str:
    mlir_opt = tools.require_mlir_opt()
    pipeline = (
        "builtin.module(gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
        "reconcile-unrealized-casts),"
        f"rocdl-attach-target{{chip={arch}}},gpu-module-to-binary)"
    )
    result = subprocess.run(
        [str(mlir_opt), f"--pass-pipeline={pipeline}"],
        input=lowered, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "gpu.binary @architecture_fragment_mod" in result.stdout
    return result.stdout


@pytest.mark.parametrize(
    ("arch", "input_type", "intrinsic", "wave_size"),
    [
        ("gfx1100", "vector<16xf16>", "rocdl.wmma.f32.16x16x16.f16", 32),
        ("gfx1151", "vector<16xf16>", "rocdl.wmma.f32.16x16x16.f16", 32),
        ("gfx1200", "vector<8xf16>", "rocdl.wmma.f32.16x16x16.f16", 32),
        ("gfx1201", "vector<8xf16>", "rocdl.wmma.f32.16x16x16.f16", 32),
        ("gfx90a", "vector<4xf16>", "rocdl.mfma.f32.16x16x16f16", 64),
        ("gfx942", "vector<4xf16>", "rocdl.mfma.f32.16x16x16f16", 64),
        ("gfx950", "vector<4xf16>", "rocdl.mfma.f32.16x16x16f16", 64),
    ],
)
def test_same_tile_fixture_selects_and_assembles_exact_family(
    compiler_toolchain, arch, input_type, intrinsic, wave_size,
):
    lowered = _lower(compiler_toolchain, arch)
    assert input_type in lowered
    assert intrinsic in lowered
    binary = _serialize(compiler_toolchain, lowered, arch)
    assert f"wavefront_size = {wave_size} : i64" in binary
    assert "vgpr_spill_count = 0 : i64" in binary
    assert "sgpr_spill_count = 0 : i64" in binary


def test_fragment_resource_ratchet_keeps_family_proofs_compact(compiler_toolchain):
    # gfx1151 needs one extra live value versus the historical one-wave-only
    # path because lane identity is now reduced modulo wave_size, making the
    # materializer correct inside multi-wave workgroups.
    limits = {
        "gfx1100": 28, "gfx1151": 28,
        "gfx1200": 24, "gfx1201": 24,
        "gfx90a": 16, "gfx942": 16, "gfx950": 16,
    }
    for arch, limit in limits.items():
        binary = _serialize(
            compiler_toolchain,
            _lower(compiler_toolchain, arch),
            arch,
        )
        match = re.search(r"vgpr_count = (\d+) : i64", binary)
        assert match, binary[:1000]
        assert int(match.group(1)) <= limit, (arch, match.group(1), limit)


@pytest.mark.parametrize(
    ("dtype", "intrinsic"),
    [
        ("bf16", "rocdl.wmma.f32.16x16x16.bf16"),
        ("e4m3", "rocdl.wmma.f32.16x16x16.fp8_fp8"),
        ("e5m2", "rocdl.wmma.f32.16x16x16.bf8_bf8"),
        ("int8", "rocdl.wmma.i32.16x16x16.iu8"),
        ("int4", "rocdl.wmma.i32.16x16x32.iu4"),
    ],
)
def test_rdna4_dtype_fragment_forms_lower_and_assemble(
    compiler_toolchain, dtype, intrinsic,
):
    lowered = _lower(compiler_toolchain, "gfx1201", _typed_source(dtype))
    assert intrinsic in lowered
    binary = _serialize(compiler_toolchain, lowered, "gfx1201")
    assert "vgpr_spill_count = 0 : i64" in binary
    vgpr = re.search(r"vgpr_count = (\d+) : i64", binary)
    assert vgpr
    assert int(vgpr.group(1)) <= (36 if dtype == "int4" else 24)


@pytest.mark.parametrize("arch", ["gfx90a", "gfx942", "gfx950"])
def test_cdna_bf16_fragment_form_lowers_and_assembles(compiler_toolchain, arch):
    lowered = _lower(compiler_toolchain, arch, _typed_source("bf16"))
    assert "vector<4xbf16>" in lowered
    assert "rocdl.mfma.f32.16x16x16bf16.1k" in lowered
    binary = _serialize(compiler_toolchain, lowered, arch)
    assert "vgpr_spill_count = 0 : i64" in binary
    vgpr = re.search(r"vgpr_count = (\d+) : i64", binary)
    assert vgpr and int(vgpr.group(1)) <= 16


@pytest.mark.parametrize("arch", ["gfx1250", "gfx1251"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_gfx125x_wmma_v2_lowers_modifiers_and_assembles(
    compiler_toolchain, arch, dtype,
):
    source = _gfx125x_source("bf16" if dtype == "bf16" else "fp16")
    lowered = _lower(compiler_toolchain, arch, source, generic=True)
    assert f"rocdl.wmma.f32.16x16x32.{dtype.replace('fp', 'f')}" in lowered
    assert "signA = false" in lowered
    assert "signB = false" in lowered
    assert "modC = 0" in lowered
    assert "reuseA = false" in lowered
    assert "reuseB = false" in lowered
    binary = _serialize(compiler_toolchain, lowered, arch)
    assert "wavefront_size = 32 : i64" in binary
    assert "vgpr_spill_count = 0 : i64" in binary
    assert "sgpr_spill_count = 0 : i64" in binary
    vgpr = re.search(r"vgpr_count = (\d+) : i64", binary)
    assert vgpr and int(vgpr.group(1)) <= 30


def test_gfx940_descriptor_path_lowers_to_real_mfma(compiler_toolchain):
    # Debian LLVM 22 does not recognize gfx940 as a serialization processor;
    # retain the real lowering proof while gfx942 covers the same CDNA3 ISA
    # family through object generation.
    lowered = _lower(compiler_toolchain, "gfx940")
    assert "vector<4xf16>" in lowered
    assert "rocdl.mfma.f32.16x16x16f16" in lowered


def test_family_mismatch_is_a_named_error(compiler_toolchain):
    tessera_opt = compiler_toolchain.require_tessera_opt()
    source = FIXTURE.read_text().replace('family = "auto"', 'family = "mfma"')
    result = subprocess.run(
        [
            str(tessera_opt), "-", "--allow-unregistered-dialect",
            "--pass-pipeline=builtin.module(lower-tile-to-rocm{arch=gfx1201})",
        ],
        input=source, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "ROCM_FRAGMENT_ILLEGAL_ARCH_DESCRIPTOR" in result.stderr
