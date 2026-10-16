"""Native precision-policy AD regression; no GPU execution claim.

Uses the existing NUMPOL carrier LLVM object toolchain and independent FP64
oracles. Wider computation is explicit in Graph types; outputs keep storage
dtype. All native outputs are compared through exact bit encoding.
"""

from pathlib import Path
import subprocess
import pytest
from tests.unit.test_numeric_policy_carrier_execution import (
    _BUFFERIZE,
    _tessera_opt,
    _toolchain_or_skip,
)
import numpy as np
import ml_dtypes

ROOT = Path(".")
CANDIDATE = ""
FLAGS = ["--empty-tensor-to-alloc-tensor", *_BUFFERIZE]


def call(tool, source, flag):
    return subprocess.check_output([tool, flag], input=source, text=True)


def dense(a):
    if a.ndim == 0:
        return format(float(a), ".17e")
    return "[" + ", ".join(dense(v) for v in a) + "]"


def run_case(kind, storage, accum, affine, dynamic, mode, mismatch=False, permuted=False,
             profile="overflow", epsilon=1e-5):
    dtype = {"f16": np.float16, "bf16": ml_dtypes.bfloat16, "f32": np.float32}[storage]
    m, n = 2, 32
    value = 1e20 if storage == "f32" else 64.0
    x = np.full((m, n), value, dtype=dtype)
    x[1, 1::2] = -value
    gamma = np.tile(np.array([1.5, 0.5], dtype=dtype), n // 2)
    beta = np.full(n, 0.25, dtype=dtype)
    seeds = [np.tile(np.array([1, 1, -1, -1], dtype=dtype), (m, n // 4))]
    roots = [x]
    if affine >= 1:
        roots.append(gamma)
        seeds.append(np.full(n, 0.125, dtype=dtype))
    if affine == 2:
        roots.append(beta)
        seeds.append(np.full(n, 0.0625, dtype=dtype))
    dy = np.tile(np.array([1, -1, -1, 1], dtype=dtype), (m, n // 4))
    if profile != "overflow":
        assert profile in {"mixed", "offset", "radial", "tiny_zero", "tiny_radial"}
        rng = np.random.default_rng(619)
        x[:] = (rng.normal(0, 2, x.shape) + (64 if profile == "offset" else 0)).astype(dtype)
        gamma[:] = rng.uniform(-1.5, 1.5, n).astype(dtype)
        beta[:] = rng.normal(0, 0.25, n).astype(dtype)
        for seed in seeds:
            seed[:] = rng.normal(0, 0.5, seed.shape).astype(dtype)
        dy[:] = rng.normal(0, 0.5, dy.shape).astype(dtype)
        if profile == "radial":
            # A genuinely smooth epsilon derivative survives even when the
            # large projection terms cancel. Do not relax the one-encoding gate.
            gamma[:] = 2
            seeds[0][:] = x
            for seed in seeds[1:]:
                seed.fill(0)
            dy[:] = (x.astype(np.float64) / (2 if affine else 1)).astype(dtype)
        if profile in {"tiny_zero", "tiny_radial"}:
            x.fill(0 if profile == "tiny_zero" else 1e-30)
            gamma.fill(2)
            for seed in seeds:
                seed.fill(0)
            if profile == "tiny_radial":
                seeds[0][:] = x
            dy[:] = x.astype(np.float64) / (2 if affine else 1)
    types = [f"tensor<{'?' if dynamic else m}x{n}x{storage}>"] + [f"tensor<{n}x{storage}>"] * affine
    order = [1, 0, *range(2, len(types))] if permuted else list(range(len(types)))
    args = ", ".join(f"%a{i}: {types[i]}" for i in order)
    names = ", ".join(f"%a{i}" for i in range(len(types)))
    policy = (
        ""
        if accum is None
        else ', numeric_policy = {storage = "'
        + {"f16": "fp16", "bf16": "bf16", "f32": "fp32"}[storage]
        + '", accum = "'
        + accum
        + '"}'
    )
    attr = ", ".join(str(i) + " : i64" for i in range(len(types)))
    source = f'''module {{
 func.func @norm({args}) -> {types[0]} attributes {{tessera.autodiff = "{"forward" if mode == "jvp" else "reverse"}", tessera.autodiff.wrt_indices = [{attr}]}} {{
  %y = "tessera.{kind}"({names}) {{eps = {epsilon:.17e} : f64{policy}}} : ({", ".join(types)}) -> {types[0]}
  return %y : {types[0]}
 }}
}}
'''
    flag = "--tessera-autodiff-forward" if mode == "jvp" else "--tessera-autodiff-paired"
    ir = call(CANDIDATE, source, flag)
    label = f"{kind}-{storage}-{accum or 'default'}-a{affine}-d{int(dynamic)}-{mode}" + (
        "-mismatched-seed" if mismatch else ""
    )
    label += "-permuted" if permuted else ""
    label += "-" + profile
    (ROOT / (label + "-graph.mlir")).write_text(ir)
    if accum is None:
        assert "tessera.cast" not in ir, label + " widened an unspecified accumulator"
        assert "tensor<2x32xf32>" not in ir and "tensor<2x32xf64>" not in ir
        return {"case": label, "status": "default_storage_types_preserved"}
    wide = x.astype(np.float64)
    center = np.mean(wide, axis=-1, keepdims=True) if kind == "layer_norm" else 0.0
    inv = 1 / np.sqrt(np.mean((wide - center) ** 2, axis=-1, keepdims=True) + epsilon)
    z = (wide - center) * inv
    g = gamma.astype(np.float64) if affine else 1.0
    primal = z * g + (beta.astype(np.float64) if affine == 2 else 0)
    if mode == "jvp":
        dx = seeds[0].astype(np.float64)
        tangent = (
            (
                dx
                - (np.mean(dx, axis=-1, keepdims=True) if kind == "layer_norm" else 0)
                - z * np.mean(dx * z, axis=-1, keepdims=True)
            )
            * inv
            * g
        )
        if affine:
            tangent += z * seeds[1].astype(np.float64)
        if affine == 2:
            tangent += seeds[2].astype(np.float64)
        expected = [primal, tangent]
        launch = [roots[i] for i in order] + [seeds[i] for i in order]
        launch_types = [types[i] for i in order] * 2
        results = [types[0], types[0]]
        symbol = "norm__jvp"
    else:
        projected = dy.astype(np.float64) * g
        dx = (
            projected
            - (np.mean(projected, axis=-1, keepdims=True) if kind == "layer_norm" else 0)
            - z * np.mean(projected * z, axis=-1, keepdims=True)
        ) * inv
        expected = [dx]
        if affine:
            expected.append(np.sum(dy.astype(np.float64) * z, axis=0))
        if affine == 2:
            expected.append(np.sum(dy.astype(np.float64), axis=0))
        expected = [expected[i] for i in order]
        launch = [roots[i] for i in order] + [dy]
        launch_types = [types[i] for i in order] + [types[0]]
        results = [types[i] for i in order]
        symbol = "norm__bwd"
    expected = [v.astype(dtype).astype(np.float64) for v in expected]
    if mismatch:
        index = len(roots) if mode == "jvp" else len(launch) - 1
        launch[index] = np.concatenate((launch[index], launch[index][:1]), axis=0)
    lines = [
        "func.func private @printI64(i64)",
        "func.func private @printNewline()",
        "func.func @probe() {",
    ]
    values = []
    for i, (value, target) in enumerate(zip(launch, launch_types, strict=True)):
        static = "tensor<" + "x".join(str(v) for v in value.shape) + "x" + storage + ">"
        lines.append(f"%v{i} = arith.constant dense<{dense(value)}> : {static}")
        name = f"%v{i}"
        if static != target:
            lines.append(f"%b{i} = tensor.cast {name} : {static} to {target}")
            name = f"%b{i}"
        values.append(name)
    ids = ", ".join(f"%r{i}" for i in range(len(results)))
    result_spec = results[0] if len(results) == 1 else "(" + ", ".join(results) + ")"
    lines.append(f"{ids} = func.call @{symbol}({', '.join(values)}) : ({', '.join(launch_types)}) -> {result_spec}")
    for i, v in enumerate(expected):
        for j, index in enumerate(np.ndindex(v.shape)):
            idx = []
            for axis, k in enumerate(index):
                name = f"%i{i}_{j}_{axis}"
                lines.append(f"{name} = arith.constant {k} : index")
                idx.append(name)
            lines.append(f"%h{i}_{j} = tensor.extract %r{i}[{', '.join(idx)}] : {results[i]}")
            lines.append(f"%f{i}_{j} = arith.extf %h{i}_{j} : {storage} to f64")
            lines.append(f"%bits{i}_{j} = arith.bitcast %f{i}_{j} : f64 to i64")
            lines.append(f"func.call @printI64(%bits{i}_{j}) : (i64) -> ()")
            lines.append("func.call @printNewline() : () -> ()")
    lines += ["return", "}"]
    pos = ir.rfind("}")
    module = ir[:pos] + "\n" + "\n".join(lines) + "\n" + ir[pos:]
    linear = call(CANDIDATE, module, "--tessera-to-linalg")
    lowered = subprocess.check_output(["mlir-opt", *FLAGS], input=linear, text=True)
    path = ROOT / (label + "-llvm.mlir")
    path.write_text(lowered)
    ll = ROOT / (label + ".ll")
    obj = ROOT / (label + ".o")
    executable = ROOT / (label + ".exe")
    subprocess.run(
        ["mlir-translate", "--mlir-to-llvmir", str(path), "-o", str(ll)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "llc",
            "-filetype=obj",
            "-relocation-model=pic",
            "-O2",
            str(ll),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["clang", str(obj), str(ROOT / "driver.c"), "-o", str(executable), "-lm"],
        check=True,
        capture_output=True,
    )
    process = subprocess.run([str(executable)], text=True, capture_output=True, cwd=ROOT)
    if mismatch:
        assert (
            process.returncode == -6
            and process.stdout.strip() == "normalization derivative seed shape must match primal"
        ), (label, process.returncode, process.stdout, process.stderr)
        return {
            "case": label,
            "status": "native_shape_guard_rejected",
            "returncode": process.returncode,
        }
    process.check_returncode()
    actual = np.asarray([int(v) for v in process.stdout.split()], dtype=np.int64).view(np.float64)
    reference = np.concatenate([v.reshape(-1) for v in expected])
    assert actual.shape == reference.shape, (label, actual.shape, reference.shape)
    if profile == "overflow":
        np.testing.assert_allclose(
            actual,
            reference,
            rtol=3e-6,
            atol=1e-25 if storage == "f32" else 3e-7,
            err_msg=label,
        )
    else:
        # Independent FP64 algebra is rounded once to the output storage type.
        # FP32 intermediate reduction order can straddle a storage rounding
        # midpoint; permit at most one adjacent output encoding in either
        # direction. This bound is per element, not a global relative tolerance.
        stored = reference.astype(dtype)
        upper = np.nextafter(stored, np.full(stored.shape, np.inf, dtype=dtype))
        lower = np.nextafter(stored, np.full(stored.shape, -np.inf, dtype=dtype))
        ulp = np.maximum(
            upper.astype(np.float64) - reference,
            reference - lower.astype(np.float64),
        )
        assert np.all(np.isfinite(actual)), label
        assert np.all(np.abs(actual - reference) <= ulp), (
            label, float(np.max(np.abs(actual - reference) / ulp))
        )
    return {
        "case": label,
        "status": "native_llvm_oracle_passed",
        "max_abs_error": float(np.max(np.abs(actual - reference))),
    }


_DRIVER = r"""
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <sys/resource.h>
extern void probe(void);
void printI64(int64_t value) { printf("%" PRId64, value); }
void printNewline(void) { putchar('\n'); }
int main(void) {
  struct rlimit core_limit = {0, 0};
  if (setrlimit(RLIMIT_CORE, &core_limit) != 0) return 80;
  setvbuf(stdout, NULL, _IONBF, 0);
  probe();
  return 0;
}
"""


@pytest.fixture(scope="module", autouse=True)
def _native_environment(tmp_path_factory):
    global ROOT, CANDIDATE
    _toolchain_or_skip()
    CANDIDATE = _tessera_opt()
    ROOT = tmp_path_factory.mktemp("normalization_policy")
    (ROOT / "driver.c").write_text(_DRIVER)


_CASES = []
for _mode in ("jvp", "vjp"):
    for _kind in ("rmsnorm", "layer_norm"):
        for _storage in ("f16", "bf16"):
            for _affine in range(2 if _kind == "rmsnorm" else 3):
                for _dynamic in (False, True):
                    _CASES.append((_kind, _storage, "fp32", _affine, _dynamic, _mode))
                _CASES.append((_kind, _storage, None, _affine, False, _mode))
        _CASES.append((_kind, "f32", "fp64", 0, False, _mode))


@pytest.mark.compiler_route
@pytest.mark.parametrize("case", _CASES)
def test_normalization_ad_preserves_requested_accumulator(case):
    run_case(*case)


@pytest.mark.compiler_route
@pytest.mark.parametrize("kind", ("rmsnorm", "layer_norm"))
@pytest.mark.parametrize("mode", ("jvp", "vjp"))
def test_normalization_ad_rejects_mismatched_symbolic_seed(kind, mode):
    run_case(kind, "f16", "fp32", 0, True, mode, mismatch=True)


@pytest.mark.compiler_route
@pytest.mark.parametrize("kind", ("rmsnorm", "layer_norm"))
@pytest.mark.parametrize("mode", ("jvp", "vjp"))
def test_normalization_ad_does_not_substitute_a_different_accumulator(kind, mode):
    flag = "--tessera-autodiff-forward" if mode == "jvp" else "--tessera-autodiff-paired"
    source = f"""
module {{
  func.func @unsupported(%x: tensor<2x32xbf16>) -> tensor<2x32xbf16>
    attributes {{tessera.autodiff = "{"forward" if mode == "jvp" else "reverse"}",
                tessera.autodiff.wrt_indices = [0 : i64]}} {{
    %y = "tessera.{kind}"(%x)
      {{eps = 1.0e-5 : f64,
        numeric_policy = {{storage = "bf16", accum = "fp16"}}}}
      : (tensor<2x32xbf16>) -> tensor<2x32xbf16>
    return %y : tensor<2x32xbf16>
  }}
}}
"""
    result = subprocess.run([CANDIDATE, flag], input=source, text=True, capture_output=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert kind in result.stderr and "fp16" in result.stderr and "bf16" in result.stderr


@pytest.mark.compiler_route
def test_symbolic_normalization_admission_does_not_admit_unproved_channel_extent():
    source = """
module {
  func.func @unknown_channels(%x: tensor<2x?xf16>) -> tensor<2x?xf16>
    attributes {tessera.autodiff = "forward", tessera.autodiff.wrt_indices = [0 : i64]} {
    %y = "tessera.rmsnorm"(%x) {eps = 1.0e-5 : f64}
      : (tensor<2x?xf16>) -> tensor<2x?xf16>
    return %y : tensor<2x?xf16>
  }
}
"""
    result = subprocess.run(
        [CANDIDATE, "--tessera-autodiff-forward"],
        input=source,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "requires static ranked numeric" in result.stderr


@pytest.mark.compiler_route
@pytest.mark.parametrize("kind", ("rmsnorm", "layer_norm"))
@pytest.mark.parametrize("storage", ("f16", "bf16"))
@pytest.mark.parametrize("mode", ("jvp", "vjp"))
def test_symbolic_normalization_keeps_permuted_affine_argument_lineage(kind, storage, mode):
    run_case(kind, storage, "fp32", 1 if kind == "rmsnorm" else 2, True, mode, permuted=True)


@pytest.mark.compiler_route
@pytest.mark.parametrize("kind", ("rmsnorm", "layer_norm"))
@pytest.mark.parametrize("storage", ("f16", "bf16"))
@pytest.mark.parametrize("mode", ("jvp", "vjp"))
@pytest.mark.parametrize("profile", ("mixed", "offset"))
def test_native_normalization_affine_nontrivial_and_cancellation(kind, storage, mode, profile):
    run_case(kind, storage, "fp32", 1 if kind == "rmsnorm" else 2, True, mode,
             permuted=True, profile=profile)


@pytest.mark.compiler_route
@pytest.mark.parametrize("storage,accum", [("f16", "fp32"), ("bf16", "fp32"), ("f32", "fp64")])
@pytest.mark.parametrize("mode", ["jvp", "vjp"])
@pytest.mark.parametrize("affine", [0, 1])
@pytest.mark.parametrize("dynamic", [False, True])
def test_native_rms_radial_derivative_retains_epsilon(storage, accum, mode, affine, dynamic):
    run_case("rmsnorm", storage, accum, affine, dynamic, mode,
             permuted=bool(affine), profile="radial")


@pytest.mark.compiler_route
@pytest.mark.parametrize("storage,accum,epsilon", [
    ("f16", "fp32", 1e-40), ("bf16", "fp32", 1e-40), ("f32", "fp64", 1e-310)])
@pytest.mark.parametrize("mode", ["jvp", "vjp"])
@pytest.mark.parametrize("affine", [0, 1])
@pytest.mark.parametrize("dynamic", [False, True])
def test_native_rms_tiny_epsilon_zero_seed_avoids_inverse_square_overflow(
    storage, accum, epsilon, mode, affine, dynamic,
):
    run_case("rmsnorm", storage, accum, affine, dynamic, mode,
             permuted=bool(affine), profile="tiny_zero", epsilon=epsilon)


@pytest.mark.compiler_route
@pytest.mark.parametrize("mode", ["jvp", "vjp"])
@pytest.mark.parametrize("affine", [0, 1])
@pytest.mark.parametrize("dynamic", [False, True])
def test_native_rms_tiny_radial_seed_avoids_inverse_square_overflow(mode, affine, dynamic):
    run_case("rmsnorm", "bf16", "fp32", affine, dynamic, mode,
             permuted=bool(affine), profile="tiny_radial", epsilon=1e-40)
