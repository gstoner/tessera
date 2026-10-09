"""Native program ownership for scaled producer result permutations."""
import base64
import json
import os
import re
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def opt():
    tool = os.environ.get("TESSERA_OPT")
    if tool is None:
        pytest.skip("matching native compiler required")
    return tool


def source(shape=(17, 19, 64)):
    fixture = Path(__file__).resolve().parents[1] / "tessera-ir/phase2/e2e_mxfp8_scale_schedule_unsigned.mlir"
    m, n, k = shape
    groups = (k + 31) // 32
    replacements = {
        "17x64xf8E4M3FN": f"{m}x{k}xf8E4M3FN",
        "64x19xf8E4M3FN": f"{k}x{n}xf8E4M3FN",
        "17x2xui8": f"{m}x{groups}xui8",
        "2x19xui8": f"{groups}x{n}xui8",
        "17x19xf32": f"{m}x{n}xf32",
    }
    text = re.sub("|".join(re.escape(key) for key in replacements),
                  lambda match: replacements[match.group(0)], fixture.read_text())
    text = text.replace(f") -> tensor<{m}x{n}xf32> {{", f") -> tensor<{n}x{m}xf32> {{")
    return text.replace(
        f"return %0 : tensor<{m}x{n}xf32>",
        '%permuted = "tessera.transpose"(%0) {permutation = array<i64: 1, 0>} '
        f': (tensor<{m}x{n}xf32>) -> tensor<{n}x{m}xf32>\n'
        f'    return %permuted : tensor<{n}x{m}xf32>')


def export(opt, text, selection=None):
    options = "export-scaled-primal=true"
    if selection is not None:
        options += f" select-scaled-member={selection}"
    return subprocess.run(
        [opt, "--pass-pipeline=builtin.module(tessera-autodiff-forward{" + options + "})"],
        input=text, text=True, capture_output=True, timeout=90)


def manifest(text):
    encoded = re.search(r'tessera.autodiff.scaled_program_json = "([^"]+)"', text)
    assert encoded is not None, text
    return json.loads(base64.b64decode(encoded.group(1), validate=True))


def test_native_result_permutation_retains_computed_buffer_lifetimes(opt):
    result = export(opt, source())
    assert result.returncode == 0, result.stderr
    program = manifest(result.stdout)
    assert program["kind"] == "primal"
    assert [step["operation"] for step in program["steps"]] == [
        "tessera.scaled_matmul", "tessera.transpose"]
    assert program["steps"][1]["inputs"] == [4]
    assert program["steps"][1]["permutation"] == [1, 0]
    assert program["outputs"] == [5]
    product, output = program["buffers"][4:]
    assert product["shape"] == [17, 19] and output["shape"] == [19, 17]
    assert product["bytes"] == output["bytes"] == 17 * 19 * 4
    assert (product["ownership"], product["first_write"], product["last_read"]) == (1, 0, 1)
    assert (output["ownership"], output["first_write"], output["last_read"]) == (2, 1, 2)


def test_native_result_member_projection_carries_one_source_binding(opt):
    result = export(opt, source(), 1)
    assert result.returncode == 0, result.stderr
    assert "tessera.autodiff.scaled_program_witness" in result.stdout
    assert 'tessera.launch_bindings = ["member_input_0", "member_output"]' in result.stdout
    assert "permutation = array<i64: 1, 0>" in result.stdout
    assert "tessera.transpose" in result.stdout


def test_invalid_result_axes_cannot_export_an_owned_program(opt):
    result = export(opt, source().replace("array<i64: 1, 0>", "array<i64: 0, 1>"))
    assert result.returncode != 0
    assert "declared input axes" in result.stderr
    assert "scaled_program_json" not in result.stdout


def compile_member(opt, text, stages):
    prefix = "tessera-autodiff-forward{export-scaled-primal=true select-scaled-member=1}"
    return subprocess.run(
        [opt, "--pass-pipeline=builtin.module(" + prefix + "," + stages + ")"],
        input=text, text=True, capture_output=True, timeout=90)


def test_result_permutation_carries_native_schedule_tile_and_target(opt):
    schedule = compile_member(opt, source(), "tessera-graph-to-schedule")
    assert schedule.returncode == 0, schedule.stderr
    assert 'shape_key = "family=result_permutation"' in schedule.stdout
    assert "source_shape = array<i64: 17, 19>" in schedule.stdout
    tile = compile_member(opt, source(), "tessera-graph-to-schedule,tessera-schedule-to-tile")
    assert tile.returncode == 0, tile.stderr
    assert "tile.transpose_kernel" in tile.stdout
    assert "source_shape = array<i64: 17, 19>" in tile.stdout
    target = compile_member(opt, source(),
        "tessera-graph-to-schedule,tessera-schedule-to-tile,"
        "tessera-rocm-executable{family=scalar_unary input=tile output=target arch=gfx1201}")
    assert target.returncode == 0, target.stderr
    assert "gpu.func @tessera_tile_result_permutation_" in target.stdout
    assert "memref.load" in target.stdout and "memref.store" in target.stdout
    assert "tessera.rocm.program_member_json" in target.stdout
    assert "tile.transpose_kernel" not in target.stdout


def test_schedule_replay_rejects_changed_output_axes(opt):
    schedule = compile_member(opt, source(), "tessera-graph-to-schedule")
    assert schedule.returncode == 0, schedule.stderr
    corrupted = schedule.stdout.replace("output_shape = array<i64: 19, 17>",
                                       "output_shape = array<i64: 18, 17>")
    assert corrupted != schedule.stdout
    result = subprocess.run([opt, "--tessera-schedule-to-tile"], input=corrupted,
                            text=True, capture_output=True, timeout=90)
    assert result.returncode != 0
    assert "Schedule contract was altered" in result.stderr


def test_target_replay_rejects_changed_tile_permutation(opt):
    tile = compile_member(opt, source(), "tessera-graph-to-schedule,tessera-schedule-to-tile")
    assert tile.returncode == 0, tile.stderr
    lines = tile.stdout.splitlines()
    matches = [i for i, line in enumerate(lines) if "tile.transpose_kernel" in line]
    assert len(matches) == 1
    index = matches[0]
    lines[index] = lines[index].replace("permutation = array<i64: 1, 0>",
                                      "permutation = array<i64: 0, 1>")
    corrupted = "\n".join(lines)
    result = subprocess.run(
        [opt, "--pass-pipeline=builtin.module(tessera-rocm-executable{family=scalar_unary input=tile output=target arch=gfx1201})"],
        input=corrupted, text=True, capture_output=True, timeout=90)
    assert result.returncode != 0
    assert "Tile differs from its Schedule" in result.stderr
