"""Full verified Graph passes and native semantic members share compact layout contracts."""
import pytest
import tessera as ts

from tests.unit.test_native_tensor_binary_producers import binary_source
from tests.unit.test_native_tensor_mlp_program import mlp_source
from tests.unit.test_native_tensor_program_pipeline import (
    _run, norm_product, relu_product, traced_source,
)

pytestmark = [pytest.mark.compiler_route, pytest.mark.usefixtures("production_compiler")]


def softmax_product(a, b):
    return ts.ops.matmul(ts.ops.softmax(a, axis=-1), b, output_dtype="fp32")


def annotated_source(kind, storage, key, value):
    source = (binary_source("lhs", storage) if kind == "add" else
              traced_source({"relu": relu_product, "rmsnorm": norm_product,
                             "softmax": softmax_product}[kind], storage))
    lines = source.splitlines()
    index = next(i for i, line in enumerate(lines) if "= tessera." + kind + " " in line)
    assert "{" in lines[index]
    lines[index] = lines[index].replace("{", "{" + key + " = " + value + ", ", 1)
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("kind", ["add", "relu", "rmsnorm", "softmax"])
@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("key", ["layout", "tessera.layout"])
def test_explicit_compact_layout_keeps_native_admission_and_full_pipeline(kind, storage, key):
    source = annotated_source(kind, storage, key, '\"row_major\"')
    query = _run(source, "--tessera-graph-to-schedule=tensor-program-query")
    assert query.returncode == 0, query.stderr
    assert "tessera.native.tensor_program_admission = true" in query.stdout
    assert key + ' = \"row_major\"' in query.stdout
    compiled = _run(source, "--tessera-nvidia-pipeline-sm120", "--tessera-lower-to-nvidia-sm120")
    assert compiled.returncode == 0, compiled.stderr
    assert compiled.stdout.count("tessera_nvidia.program_invoke ") == 2
    assert "tessera.native.tensor_program_frame_json" in compiled.stdout
    assert "tessera_nvidia.mma_sync " not in compiled.stdout
    assert "nvvm.mma.sync" in compiled.stdout
    replay = _run(compiled.stdout, "--tessera-graph-to-schedule=tensor-program-replay")
    assert replay.returncode == 0, replay.stderr
    assert replay.stdout == compiled.stdout


@pytest.mark.parametrize("kind", ["add", "relu", "rmsnorm", "softmax"])
@pytest.mark.parametrize("key", ["layout", "tessera.layout"])
@pytest.mark.parametrize("value", ['\"col_major\"', "false", "12 : i64"])
def test_conflicting_or_malformed_layout_never_becomes_native_admission(kind, key, value):
    source = annotated_source(kind, "f16", key, value)
    query = _run(source, "--tessera-graph-to-schedule=tensor-program-query")
    assert query.returncode == 0, query.stderr
    assert "tessera.native.tensor_program_admission = false" in query.stdout
    assert "tensor_stage_" not in query.stdout
    assert key + " = " + value in query.stdout


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("reordered", [False, True])
@pytest.mark.parametrize("intermediate", ["f16", "f32"])
def test_full_pipeline_keeps_multi_matmul_program_after_layout_inference(storage, reordered, intermediate):
    source = mlp_source(storage, reordered=reordered, intermediate=intermediate)
    compiled = _run(source, "--tessera-nvidia-pipeline-sm120", "--tessera-lower-to-nvidia-sm120")
    assert compiled.returncode == 0, compiled.stderr
    assert compiled.stdout.count("tessera_nvidia.program_invoke ") == 3
    assert "tessera.native.tensor_program_frame_json" in compiled.stdout
    assert "tessera_nvidia.mma_sync " not in compiled.stdout
    assert "nvvm.mma.sync" in compiled.stdout
    replay = _run(compiled.stdout, "--tessera-graph-to-schedule=tensor-program-replay")
    assert replay.returncode == 0, replay.stderr
    assert replay.stdout == compiled.stdout
