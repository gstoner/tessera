"""Native producer DAGs preserve multiple matmuls and three physical roots."""
from dataclasses import replace
import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera.compiler.native_tensor_program import _frame
from tests.unit.test_native_tensor_program_pipeline import _run

pytestmark = [pytest.mark.compiler_route, pytest.mark.usefixtures("production_compiler")]


def mlp(x, up, down):
    hidden = ts.ops.matmul(x, up, output_dtype="fp16")
    return ts.ops.matmul(ts.ops.relu(hidden), down, output_dtype="fp32")


def default_mlp(x, up, down):
    hidden = ts.ops.matmul(x, up, output_dtype="fp32")
    return ts.ops.matmul(ts.ops.relu(hidden), down, output_dtype="fp32")


def reordered_default_mlp(down, up, x):
    return default_mlp(x, up, down)


def reordered_mlp(down, up, x):
    return mlp(x, up, down)


def mlp_roots(storage, *, reordered=False, shape=(17, 35, 29, 19), seed=120321, intermediate="f16"):
    m, k, h, n = shape
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .2, (m, k)).astype(dtype)
    up = rng.normal(0, .2, (k, h)).astype(dtype)
    # The source explicitly stores the first product as FP16, so this
    # consumer weight has FP16 storage even for BF16 first-stage operands.
    down = rng.normal(0, .2, (h, n)).astype(np.float16 if intermediate == "f16" else dtype)
    return (down, up, x) if reordered else (x, up, down)


def mlp_source(storage, reordered=False, intermediate="f16"):
    function = ((reordered_default_mlp if reordered else default_mlp)
                if intermediate == "f32" else (reordered_mlp if reordered else mlp))
    fn = ts.jit(target="nvidia_sm120")(function)
    graph = fn._traced_autodiff_module(
        mlp_roots(storage, reordered=reordered, intermediate=intermediate), {})
    graph = replace(graph, module_attrs={**graph.module_attrs,
        "tessera.target": '"nvidia_sm120"', "tessera.arch": '"sm_120"'})
    return graph.to_mlir(target="nvidia_sm120", canonical=True)


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("reordered", [False, True])
def test_native_query_admits_three_root_two_matmul_source(storage, reordered):
    source = mlp_source(storage, reordered)
    verified = _run(source, "--verify-each")
    assert verified.returncode == 0, verified.stderr
    queried = _run(source, "--tessera-graph-to-schedule=tensor-program-query")
    assert queried.returncode == 0, queried.stderr
    assert "tessera.native.tensor_program_admission = true" in queried.stdout


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("reordered", [False, True])
def test_native_mlp_keeps_each_producer_consumer_and_root_binding(storage, reordered):
    source = mlp_source(storage, reordered)
    result = _run(source, "--tessera-graph-to-schedule",
                  "--tessera-schedule-to-tile", "--tessera-lower-to-nvidia-sm120")
    assert result.returncode == 0, result.stderr
    frame = _frame(result.stdout)
    assert frame["roots"] == [0, 1, 2]
    assert len(frame["buffers"]) == 6 and frame["output"] == 5
    assert [stage["output"] for stage in frame["stages"]] == [3, 4, 5]
    assert [stage["inputs"] for stage in frame["stages"]] == (
        [[2, 1], [3], [4, 0]] if reordered else [[0, 1], [3], [4, 2]])
    assert [spec["storage"] for spec in frame["buffers"][3:]] == ["f16", "f16", "f32"]
    assert result.stdout.count("tessera_nvidia.program_invoke ") == 3
    assert "tile.program_invoke " not in result.stdout



@pytest.mark.parametrize("matrix_index", [0, 2])
@pytest.mark.parametrize("attribute,value", [
    ("activation", "relu"), ("rhs_storage_order", "col_major"),
    ("numeric_policy", {"accum": "fp16"}),
])
def test_each_matmul_declines_unimplemented_physical_policy(matrix_index, attribute, value):
    fn = ts.jit(target="nvidia_sm120")(mlp)
    graph = fn._traced_autodiff_module(mlp_roots("f16"), {})
    graph.functions[0].body[matrix_index].kwargs[attribute] = value
    graph = replace(graph, module_attrs={**graph.module_attrs,
        "tessera.target": '"nvidia_sm120"', "tessera.arch": '"sm_120"'})
    source = graph.to_mlir(target="nvidia_sm120", canonical=True)
    result = _run(source, "--tessera-graph-to-schedule=tensor-program-query")
    assert result.returncode != 0 or "tessera.native.tensor_program_admission = false" in result.stdout


def test_each_matmul_checks_its_operand_storage_pair():
    source = mlp_source("f16").replace("tensor<35x29xf16>", "tensor<35x29xbf16>")
    result = _run(source, "--tessera-graph-to-schedule=tensor-program-query")
    assert result.returncode != 0 or "tessera.native.tensor_program_admission = false" in result.stdout


def test_variadic_frontend_signature_preserves_all_root_metadata():
    from tessera.compiler.native_tensor_jit import input_signature
    first = mlp_roots("bf16", reordered=True)
    changed = mlp_roots("bf16", reordered=True, seed=120322)
    signature = input_signature(first)
    assert signature is not None and len(signature[1]) == 3
    assert signature == input_signature(changed)
    assert [spec[0] for spec in signature[1]] == ["float16", "bfloat16", "bfloat16"]
    assert input_signature((first[0], first[1], first[2][:, ::2])) is None



@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("reordered", [False, True])
def test_default_fp32_mlp_preserves_precision_through_native_program(storage, reordered):
    source = mlp_source(storage, reordered, intermediate="f32")
    verified = _run(source, "--verify-each")
    assert verified.returncode == 0, verified.stderr
    queried = _run(source, "--tessera-graph-to-schedule=tensor-program-query")
    assert queried.returncode == 0, queried.stderr
    assert "tessera.native.tensor_program_admission = true" in queried.stdout
    tiled = _run(source, "--tessera-graph-to-schedule", "--tessera-schedule-to-tile")
    assert tiled.returncode == 0, tiled.stderr
    assert "tile.scalar_matmul_kernel" in tiled.stdout
    assert "tile.activation_kernel" in tiled.stdout
    frame = _frame(tiled.stdout)
    assert [spec["storage"] for spec in frame["buffers"][3:]] == ["f32", "f32", "f32"]
    assert [stage["inputs"] for stage in frame["stages"]] == (
        [[2, 1], [3], [4, 0]] if reordered else [[0, 1], [3], [4, 2]])
    target = _run(tiled.stdout, "--tessera-lower-to-nvidia-sm120")
    assert target.returncode == 0, target.stderr
    assert target.stdout.count("tessera_nvidia.program_invoke ") == 3
    assert "tile.scalar_matmul_kernel" not in target.stdout
    assert "llvm.fma" in target.stdout or "math.fma" in target.stdout


def f32_relu_tile(schedule="flat_128", zero_mode="preserve"):
    return f"""module attributes {{tessera.arch = "sm_120", tessera.target = "nvidia_sm120"}} {{
      llvm.func @f32_relu(%x: !llvm.ptr, %y: !llvm.ptr, %m: i64, %n: i64)
          attributes {{nvvm.kernel}} {{
        "tile.activation_kernel"(%x, %y, %m, %n) {{
          storage = "f32", accum = "f32", kind = "relu", schedule = "{schedule}",
          nan_mode = "propagate", zero_mode = "{zero_mode}",
          axis = -1 : i64, tessera.workgroup_size = 128 : i64
        }} : (!llvm.ptr, !llvm.ptr, i64, i64) -> ()
        llvm.return
      }}
    }}"""


@pytest.mark.parametrize("schedule", ["serial", "flat_128"])
def test_f32_relu_native_leaf_preserves_storage_and_zero_policy(schedule):
    lowered = _run(f32_relu_tile(schedule), "--tessera-lower-to-nvidia-sm120")
    assert lowered.returncode == 0, lowered.stderr
    assert "tile.activation_kernel" not in lowered.stdout
    assert "fpext" not in lowered.stdout and "fptrunc" not in lowered.stdout
    assert "olt" in lowered.stdout
    assert "alignment = 4" in lowered.stdout


def test_f32_relu_refuses_precision_changing_zero_policy():
    result = _run(f32_relu_tile(zero_mode="positive"), "--verify-each")
    assert result.returncode != 0
    assert "propagated NaNs" in result.stderr


def mlp_numeric_interval(args, reordered, intermediate):
    """Independent finite-input enclosure of declared stores and FP32 FMAs.

    gamma_k bounds accumulation, not a measured tolerance. Rounding interval
    endpoints through the declared store isolates half-midpoint ambiguity.
    The final linear interval includes RHS signs before FP32 accumulation.
    """
    down, up, x = args if reordered else (args[2], args[1], args[0])
    x, up, down = (value.astype(np.float64) for value in (x, up, down))
    u = 2.0 ** -24
    def gamma(n):
        if n * u >= 1:
            raise ValueError("FP32 accumulation enclosure requires n*u < 1")
        return n * u / (1 - n * u)
    if not all(np.isfinite(value).all() for value in (x, up, down)):
        raise ValueError("MLP interval certifies finite inputs only")
    product = x @ up
    error = gamma(x.shape[1]) * (np.abs(x) @ np.abs(up))
    # Include gradual-underflow absolute error for each FP32 accumulation.
    error += x.shape[1] * 2.0 ** -150
    dtype = np.float16 if intermediate == "f16" else np.float32
    low = np.maximum(0, (product - error).astype(dtype).astype(np.float64))
    high = np.maximum(0, (product + error).astype(dtype).astype(np.float64))
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("MLP interval requires nonoverflowing intermediate stores")
    positive, negative = np.maximum(down, 0), np.minimum(down, 0)
    lo = low @ positive + high @ negative
    hi = high @ positive + low @ negative
    last_error = gamma(down.shape[0]) * (high @ np.abs(down))
    last_error += down.shape[0] * 2.0 ** -150
    return lo - last_error, hi + last_error


def assert_mlp_numeric(actual, args, reordered, intermediate):
    low, high = mlp_numeric_interval(args, reordered, intermediate)
    assert actual.dtype == np.float32 and actual.shape == low.shape
    assert np.isfinite(actual).all()
    outside = np.maximum(low - actual, actual - high)
    assert np.all(outside <= 0), (
        f"MLP exceeds independent store/FMA enclosure by {outside.max():.9g}")


def test_mlp_oracle_rejects_implicit_half_intermediate():
    value = np.array([[1 + 2.0 ** -10]], dtype=np.float16)
    args = (value, value, np.ones((1, 1), dtype=np.float16))
    exact = (value.astype(np.float64) @ value.astype(np.float64)).astype(np.float32)
    assert_mlp_numeric(exact, args, False, "f32")
    with pytest.raises(AssertionError, match="enclosure"):
        assert_mlp_numeric(exact.astype(np.float16).astype(np.float32), args, False, "f32")


def test_mlp_oracle_rejects_changed_arithmetic():
    args = mlp_roots("f16", intermediate="f16")
    x, up, down = (value.astype(np.float64) for value in args)
    exact = (np.maximum(0, (x @ up).astype(np.float16)).astype(np.float64) @ down).astype(np.float32)
    assert_mlp_numeric(exact, args, False, "f16")
    with pytest.raises(AssertionError, match="enclosure"):
        assert_mlp_numeric(exact + np.float32(.01), args, False, "f16")
