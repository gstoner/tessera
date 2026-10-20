"""Original FP32 roots retain storage through native producer-to-matmul programs."""
from dataclasses import replace
import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera.compiler.native_tensor_jit import input_signature
from tessera.compiler.native_tensor_program import _frame
from tests.unit.test_native_tensor_program_pipeline import _run

pytestmark = [pytest.mark.compiler_route, pytest.mark.usefixtures("production_compiler")]


def source(x, weight):
    return ts.ops.matmul(ts.ops.relu(x), weight, output_dtype="fp32")

def reversed_source(weight, x):
    return source(x, weight)

def inputs(storage, reordered):
    dtype = {"f16": np.float16, "bf16": ml_dtypes.bfloat16, "f32": np.float32}[storage]
    x = np.arange(17*32, dtype=np.float32).reshape(17,32)/128 - 2
    rhs = np.ones((32,19), dtype=dtype)
    return (rhs,x) if reordered else (x,rhs)

@pytest.mark.parametrize("storage", ["f16", "bf16", "f32"])
@pytest.mark.parametrize("reordered", [False, True])
def test_fp32_root_program_keeps_original_storage_and_binding(storage, reordered):
    function = reversed_source if reordered else source
    fn = ts.jit(target="nvidia_sm120")(function)
    graph = fn._traced_autodiff_module(inputs(storage,reordered), {})
    graph = replace(graph, module_attrs={**graph.module_attrs,
        "tessera.target": '"nvidia_sm120"', "tessera.arch": '"sm_120"'})
    result = _run(graph.to_mlir(target="nvidia_sm120", canonical=True),
        "--tessera-graph-to-schedule", "--tessera-schedule-to-tile",
        "--tessera-lower-to-nvidia-sm120")
    assert result.returncode == 0, result.stderr
    frame = _frame(result.stdout)
    assert frame["roots"] == [0,1]
    assert [b["storage"] for b in frame["buffers"]] == (
        [storage,"f32","f32","f32"] if reordered else ["f32",storage,"f32","f32"])
    assert frame["stages"][0]["inputs"] == ([1] if reordered else [0])
    assert frame["stages"][1]["inputs"] == ([2,0] if reordered else [2,1])
    assert "scalar_fma_kernel" in result.stdout
    assert len(frame["stages"]) == 2 and frame["output"] == 3

@pytest.mark.parametrize("storage", ["f16", "bf16", "f32"])
@pytest.mark.parametrize("reordered", [False, True])
def test_fp32_root_frontend_signature_retains_width(storage,reordered):
    values = inputs(storage,reordered)
    signature = input_signature(values)
    assert signature is not None
    assert signature[0] is False
    assert [row[0] for row in signature[1]] == [str(v.dtype) for v in values]
    assert [row[2] for row in signature[1]] == [v.strides for v in values]


class ResidentMetadata:
    def __init__(self, shape, dtype, *, stream=11):
        self.dtype = np.dtype(dtype)
        self.interface = {"version": 3, "shape": shape,
            "typestr": self.dtype.str, "strides": None,
            "data": (4096, True), "stream": stream}

    @property
    def __cuda_array_interface__(self):
        return self.interface

    def __array__(self, *args, **kwargs):
        raise AssertionError("root metadata projection must not coerce resident data")


@pytest.mark.parametrize("storage", ["f16", "bf16", "f32"])
def test_fp32_resident_signature_preserves_primitive_width(storage):
    dtype = {"f16": np.float16, "bf16": ml_dtypes.bfloat16, "f32": np.float32}[storage]
    roots = (ResidentMetadata((17,32), np.float32),
             ResidentMetadata((32,19), dtype, stream=13))
    signature = input_signature(roots)
    assert signature is not None and signature[0] is True
    assert signature[1][0] == ("float32", (17,32), (128,4))
    assert signature[1][1][0] == str(np.dtype(dtype))


@pytest.mark.parametrize("field,value", [
    ("version", 2), ("data", (0, True)), ("stream", 0),
    ("shape", (17, -32)), ("strides", (32, 1)),
])
def test_fp32_resident_signature_rejects_invalid_physical_metadata(field,value):
    root = ResidentMetadata((17,32), np.float32)
    root.interface[field] = value
    other = ResidentMetadata((32,19), np.float16)
    try:
        signature = input_signature((root, other))
    except ValueError:
        return
    assert signature is None


def test_fp32_root_signature_declines_pitched_and_unsupported_storage():
    a,b = inputs("f16",False)
    assert input_signature((a[:, ::2], b)) is None
    assert input_signature((a.astype(np.float64), b)) is None
