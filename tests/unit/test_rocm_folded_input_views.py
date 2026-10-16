"""Typed semantic input views must retain their physical roots and row pitches."""
from pathlib import Path
import subprocess

import pytest

from tessera.compiler.scheduled_matmul import find_tessera_opt

PITCHED = "rocm_mxfp4_w4a8_folded_row_pitched_v1"


def pitched_graph(a_pitch=80, b_pitch=96, m=256, n=64, k=64):
    return f"""module attributes {{tessera.target = "rocm", tessera.arch = "gfx1201"}} {{
  func.func @folded_input_views(%a_storage: tensor<{m}x{a_pitch}xui8>,
      %b_storage: tensor<{n}x{b_pitch}xui8>, %sa: tensor<{m}xf32>,
      %ref: tensor<{n}xui8>) -> tensor<{m}x{n}xbf16> {{
    %a = tensor.extract_slice %a_storage[0, 0] [{m}, {k}] [1, 1]
      : tensor<{m}x{a_pitch}xui8> to tensor<{m}x{k}xui8>
    %b = tensor.extract_slice %b_storage[0, 0] [{n}, {k}] [1, 1]
      : tensor<{n}x{b_pitch}xui8> to tensor<{n}x{k}xui8>
    %out = tessera.scaled_matmul %a, %b scales(%sa, %ref) {{
      physical_contract = "{PITCHED}",
      numeric_policy = {{accum = "fp32", execution_mode = "folded_row_reference_explicit_approximate"}},
      scale_layout = {{granularity = "output_column", block = [1, {k}], format = "e8m0_row_reference"}}
    }} : (tensor<{m}x{k}xui8>, tensor<{n}x{k}xui8>, tensor<{m}xf32>,
          tensor<{n}xui8>) -> tensor<{m}x{n}xbf16>
    return %out : tensor<{m}x{n}xbf16>
  }}
}}
"""


@pytest.fixture(scope="module")
def compiler():
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("requires matching native compiler")
    return Path(tool)


@pytest.mark.parametrize("a_pitch,b_pitch", [(64, 64), (80, 64), (64, 96), (80, 96)])
def test_semantic_input_views_reach_native_target_with_explicit_pitches(compiler, a_pitch, b_pitch):
    result = subprocess.run([str(compiler), "--tessera-graph-to-schedule",
        "--tessera-schedule-to-tile", "--lower-tile-to-rocm=arch=gfx1201"],
        input=pitched_graph(a_pitch, b_pitch), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'physical_contract = "' + PITCHED + '"' in result.stdout
    assert f"a_row_stride = {a_pitch} : i64" in result.stdout
    assert f"b_row_stride = {b_pitch} : i64" in result.stdout
    assert 'abi = "a_bfold_sa_rowref_d_m_n_k_lda_ldb"' in result.stdout


@pytest.mark.parametrize("mutation", ["column_offset", "column_step", "partial_rows", "unaligned_pitch"])
def test_unsupported_views_do_not_lose_their_storage_contract(compiler, mutation):
    graph = pitched_graph(a_pitch=160)
    if mutation == "column_offset":
        graph = graph.replace("%a_storage[0, 0]", "%a_storage[0, 16]")
    elif mutation == "column_step":
        graph = graph.replace("[256, 64] [1, 1]", "[256, 64] [1, 2]")
    elif mutation == "partial_rows":
        graph = graph.replace("tensor<256x160xui8>", "tensor<257x160xui8>")
    else:
        graph = graph.replace("tensor<256x160xui8>", "tensor<256x161xui8>")
    result = subprocess.run([str(compiler), "--tessera-graph-to-schedule"],
                            input=graph, text=True, capture_output=True)
    assert result.returncode != 0, result.stdout


def test_runtime_pitch_projection_has_one_native_identity(compiler):
    identities = []
    for a_pitch, b_pitch in [(64, 64), (80, 64), (64, 96), (80, 96)]:
        target = subprocess.run([str(compiler), "--tessera-graph-to-schedule",
            "--tessera-schedule-to-tile", "--lower-tile-to-rocm=arch=gfx1201"],
            input=pitched_graph(a_pitch, b_pitch), text=True, capture_output=True, check=True).stdout
        projected = subprocess.run([str(compiler),
            "--tessera-rocm-project-kernel-identity=family=folded_matmul runtime-k=true"],
            input=target, text=True, capture_output=True)
        assert projected.returncode == 0, projected.stderr
        assert "runtime_row_strides" in projected.stdout
        assert "a_row_stride = 0 : i64" in projected.stdout
        generated = subprocess.run([str(compiler), "--generate-wmma-gemm-kernel=via-tile=true"],
            input=projected.stdout, text=True, capture_output=True)
        assert generated.returncode == 0, generated.stderr
        assert "tile.fragment_folded_scale" in generated.stdout
        identities.append(projected.stdout)
    assert len(set(identities)) == 1


@pytest.mark.parametrize("pitches", [(63, 64), (65, 64), (64, True), (64, 1 << 63), [64, 64]])
def test_frontend_rejects_invalid_storage_spans(pitches):
    from tessera.compiler.rocm_mxfp4_folded_frontend import author_folded_scaled_matmul_graph
    with pytest.raises(ValueError, match="row pitches"):
        author_folded_scaled_matmul_graph(256, 64, 64, input_row_strides=pitches)


@pytest.mark.parametrize("old,new", [
    ('granularity = "output_column"', 'granularity = "per_tensor"'),
    ('block = [1, 64]', 'block = [2, 64]'),
    ('accum = "fp32"', 'accum = "fp16"'),
])
def test_pitched_graph_preserves_the_requested_numeric_contract(compiler, old, new):
    graph = pitched_graph()
    assert old in graph
    result = subprocess.run([str(compiler), "--tessera-graph-to-schedule"],
        input=graph.replace(old, new), text=True, capture_output=True)
    assert result.returncode != 0, result.stdout


def test_tile_cannot_substitute_a_different_storage_pitch(compiler):
    tile = subprocess.run([str(compiler), "--tessera-graph-to-schedule",
        "--tessera-schedule-to-tile"], input=pitched_graph(),
        text=True, capture_output=True, check=True).stdout
    assert "a_row_stride = 80 : i64" in tile
    result = subprocess.run([str(compiler)], text=True, capture_output=True,
        input=tile.replace("a_row_stride = 80 : i64", "a_row_stride = 96 : i64"))
    assert result.returncode != 0
    assert "leading dimensions" in result.stderr


def test_schedule_cannot_substitute_a_different_storage_pitch(compiler):
    schedule = subprocess.run([str(compiler), "--tessera-graph-to-schedule"],
        input=pitched_graph(), text=True, capture_output=True, check=True).stdout
    assert "a_row_stride = 80 : i64" in schedule
    result = subprocess.run([str(compiler), "--tessera-schedule-to-tile"],
        text=True, capture_output=True,
        input=schedule.replace("a_row_stride = 80 : i64", "a_row_stride = 96 : i64"))
    assert result.returncode != 0
    assert "altered after hashing" in result.stderr


@pytest.mark.parametrize("pitches", [(64, 64), (80, 64), (64, 96), (80, 96)])
def test_canonicalization_preserves_dense_and_pitched_input_views(compiler, pitches):
    result = subprocess.run([str(compiler), "--canonicalize", "--tessera-graph-to-schedule",
        "--tessera-schedule-to-tile", "--lower-tile-to-rocm=arch=gfx1201",
        "--tessera-rocm-project-kernel-identity=family=folded_matmul runtime-k=true"],
        input=pitched_graph(*pitches), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "runtime_row_strides" in result.stdout


@pytest.mark.parametrize("encoded_type", [
    "tensor<256x80xui8>", "tensor<64x96xui8>", "tensor<256x64xui8>",
    "tensor<64x64xui8>", "tensor<256xf32>", "tensor<64xui8>", "tensor<256x64xbf16>",
])
def test_dense_row_pitch_abi_does_not_discard_tensor_encodings(compiler, encoded_type):
    graph = pitched_graph()
    encoded = encoded_type[:-1] + ', "col_major">'
    result = subprocess.run([str(compiler), "--tessera-graph-to-schedule",
        "--tessera-schedule-to-tile", "--lower-tile-to-rocm=arch=gfx1201"],
        input=graph.replace(encoded_type, encoded), text=True, capture_output=True)
    assert result.returncode != 0, result.stdout
    assert "unencoded" in result.stderr
