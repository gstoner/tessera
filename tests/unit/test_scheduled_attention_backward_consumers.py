from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from benchmarks.rocm.benchmark_rocm_attention_backward_program import _module
from tessera.compiler.attention_contract import (
    reference_attention_backward_split_reduced,
)
from tessera.compiler.rocm_native import package_scheduled_attention_backward as package_rocm
from tessera.compiler.scheduled_attention_backward import (
    lower_scheduled_attention_backward,
    supports_scheduled_attention_backward,
)
from tessera.compiler.x86_native import package_scheduled_attention_backward as package_x86
from tessera.runtime import (
    _rocm_wmma_runtime_available,
    _submit_rocm_gfx1151_attention_backward_program,
    _submit_x86_native,
)
from tessera.compiler.scheduled_matmul import find_tessera_opt

# These lower through the production `tessera-opt`, which the CI unit lane does
# not build. The library correctly RAISES rather than silently degrading, so
# the tests must skip there rather than fail.
_needs_opt = pytest.mark.skipif(
    find_tessera_opt() is None, reason="tessera-opt not built"
)


def _x86_module(hq: int, hkv: int, sq: int, sk: int, *, d: int = 16):
    module = _module(1, hq, hkv, sq, sk, d, dtype="fp32", lse_checkpoint="auto")
    module.functions[0].body[0].kwargs["window"] = 3
    module.functions[0].body[0].kwargs["softcap"] = 4.0
    return module


def _x86_reference(do, q, key, value, bias, *, scale, window, softcap):
    b, hq, sq, d = q.shape
    _, hkv, sk, _ = key.shape
    dq, dk, dv = np.zeros_like(q), np.zeros_like(key), np.zeros_like(value)
    row_lse = np.empty((b, hq, sq), np.float32)
    offset = max(sk - sq, 0)
    for batch in range(b):
        for head in range(hq):
            kv_head = head * hkv // hq
            for row in range(sq):
                begin = max(0, row + offset - window + 1)
                end = min(sk - 1, row + offset)
                keys = key[batch, kv_head, begin : end + 1]
                scores = scale * (keys @ q[batch, head, row])
                derivative = np.ones_like(scores)
                if softcap > 0:
                    tanh = np.tanh(scores / softcap)
                    scores = softcap * tanh
                    derivative = 1 - tanh * tanh
                scores += bias[batch, head, row, begin : end + 1]
                row_lse[batch, head, row] = np.logaddexp.reduce(scores)
                probabilities = np.exp(scores - row_lse[batch, head, row])
                values = value[batch, kv_head, begin : end + 1]
                dp = values @ do[batch, head, row]
                ds = probabilities * (dp - np.sum(probabilities * dp)) * derivative
                dq[batch, head, row] = scale * (ds @ keys)
                dk[batch, kv_head, begin : end + 1] += (
                    scale * ds[:, None] * q[batch, head, row]
                )
                dv[batch, kv_head, begin : end + 1] += (
                    probabilities[:, None] * do[batch, head, row]
                )
    return (dq, dk, dv), row_lse


def _apple_module(
    *, b: int = 1, hq: int = 4, hkv: int = 2, sq: int = 8, sk: int = 8,
    d: int = 16, dtype: str = "fp32", causal: bool = False,
    window: tuple[int, int] = (-1, -1), dropout_p: float = 0.0,
    lse_checkpoint: str = "auto", scale: float = 0.25,
):
    """Rank-4 module inside the Apple MSL VJP envelope.

    Deliberately separate from the shared `_module` helper: that one always
    supplies a bias operand, while the Apple envelope must be exercised both
    with and without one.
    """
    from tessera.compiler.graph_ir import (
        GraphIRFunction, GraphIRModule, IRArg, IROp, IRType,
    )

    element = {"fp16": "f16", "bf16": "bf16", "fp32": "f32"}[dtype]

    def tensor(shape, spelling=element, kind=dtype):
        return IRType(f"tensor<{'x'.join(map(str, shape))}x{spelling}>",
                      tuple(map(str, shape)), kind)

    q, key = tensor((b, hq, sq, d)), tensor((b, hkv, sk, d))
    value, do = tensor((b, hkv, sk, d)), tensor((b, hq, sq, d))
    dq = tensor((b, hq, sq, d), "f32", "fp32")
    dk = tensor((b, hkv, sk, d), "f32", "fp32")
    dv = tensor((b, hkv, sk, d), "f32", "fp32")
    return GraphIRModule(functions=[GraphIRFunction(
        name="apple_attention_backward_program",
        args=[IRArg("do", do), IRArg("q", q), IRArg("k", key), IRArg("v", value)],
        result_types=[dq, dk, dv],
        body=[IROp(
            result="dq,dk,dv", op_name="tessera.flash_attn_bwd",
            operands=["%do", "%q", "%k", "%v"],
            operand_types=[str(do), str(q), str(key), str(value)],
            result_type=f"({dq}, {dk}, {dv})",
            kwargs={"scale": scale, "causal": causal,
                    "window_left": window[0], "window_right": window[1],
                    "softcap": 0.0, "dropout_p": dropout_p, "dropout_seed": 0,
                    "query_block": 16, "key_block": 16, "split_count": 2,
                    "lse_checkpoint": lse_checkpoint},
        )],
        return_values=["%dq", "%dk", "%dv"],
    )])


def _apple_backward_reference(do, q, key, value, *, scale, causal):
    """Analytic VJP in float64, derived from the softmax Jacobian.

    Independent of the kernel under test: comparing a kernel to itself proves
    nothing.  GQA gradients are reduced over the query-head group, which is the
    property that catches a kernel accumulating into the wrong KV head.
    """
    b, hq, sq, d = q.shape
    _, hkv, sk, _ = key.shape
    group = hq // hkv
    k_x = np.repeat(key, group, axis=1).astype(np.float64)
    v_x = np.repeat(value, group, axis=1).astype(np.float64)
    q_d, do_d = q.astype(np.float64), do.astype(np.float64)
    scores = np.einsum("bhqd,bhkd->bhqk", q_d, k_x) * scale
    if causal:
        offset = max(sk - sq, 0)
        rows = np.arange(sq)[:, None] + offset
        cols = np.arange(sk)[None, :]
        scores = np.where(cols <= rows, scores, -np.inf)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    d_v = np.einsum("bhqk,bhqd->bhkd", probs, do_d)
    d_p = np.einsum("bhqd,bhkd->bhqk", do_d, v_x)
    d_s = probs * (d_p - (d_p * probs).sum(axis=-1, keepdims=True))
    d_q = np.einsum("bhqk,bhkd->bhqd", d_s, k_x) * scale
    d_k = np.einsum("bhqk,bhqd->bhkd", d_s, q_d) * scale

    def reduce_group(value_):
        return value_.reshape(b, hkv, group, *value_.shape[2:]).sum(axis=2)

    return d_q, reduce_group(d_k), reduce_group(d_v)


def test_apple_gpu_attention_backward_envelope_fails_closed() -> None:
    """Requests outside the Apple MSL VJP envelope must not be narrowed."""
    supports = supports_scheduled_attention_backward
    assert supports(_apple_module(), target="apple_gpu")
    assert supports(_apple_module(causal=True), target="apple_gpu")
    assert supports(_apple_module(dtype="fp16"), target="apple_gpu")
    assert supports(_apple_module(dtype="bf16"), target="apple_gpu")
    # A live window: the MSL non-causal window is a symmetric half-window, which
    # is not the shared window_left/window_right semantics.
    assert not supports(_apple_module(window=(4, 4)), target="apple_gpu")
    assert not supports(_apple_module(window=(4, 0)), target="apple_gpu")
    # The ABI carries no dropout.
    assert not supports(_apple_module(dropout_p=0.1), target="apple_gpu")
    # D > 256 exceeds the kernel bound; a non-whole GQA group is unmappable.
    assert not supports(_apple_module(d=512), target="apple_gpu")
    assert not supports(_apple_module(hq=3, hkv=2), target="apple_gpu")
    # Apple cannot save an LSE checkpoint: asking for one must fail closed
    # rather than silently recomputing under a `saved` label.
    assert not supports(_apple_module(lse_checkpoint="saved"), target="apple_gpu")


@_needs_opt
def test_apple_gpu_attention_backward_declares_its_own_lse_policy() -> None:
    artifact = lower_scheduled_attention_backward(
        _apple_module(), target="apple_gpu"
    )
    assert artifact.target == "apple_gpu" and artifact.architecture == "apple7"
    # Apple must not inherit x86's save_lse or the gfx1151 threshold.
    assert artifact.lse_checkpoint_policy == "apple7_recompute"
    assert artifact.lse_checkpoint_selection == "recompute"
    assert artifact.tile_ir.count("tile.attention_backward_kernel") == 1
    assert "tessera.flash_attn" not in artifact.tile_ir
    assert "schedule." not in artifact.tile_ir


@_needs_opt
def test_apple_gpu_packages_exact_backward_tile(tmp_path, monkeypatch) -> None:
    from tessera.compiler import apple_native

    dylib = tmp_path / "libTesseraAppleRuntime.dylib"
    dylib.write_bytes(b"apple-runtime-image")
    monkeypatch.setattr(apple_native, "_runtime_library_path", lambda: dylib)
    artifact = lower_scheduled_attention_backward(
        _apple_module(), target="apple_gpu"
    )
    package = apple_native.package_scheduled_attention_backward(
        artifact, pipeline_name="tessera-lower-to-apple_gpu"
    )
    assert package.tile_ir == artifact.tile_ir
    assert package.descriptor.entry_symbol == (
        "tessera_apple_gpu_flash_attn_bwd_variant_f32_status"
    )
    assert package.descriptor.provenance["work_item"] == "E2E-REAL-5B"
    # The shared contract declares a two-way split with ascending fixed-order
    # reduction, so the deterministic split route is the only faithful mapping;
    # the atomic route (1) is explicitly nondeterministic.
    assert package.descriptor.provenance["msl_route"] == 2
    assert package.descriptor.provenance["deterministic"] is True
    assert package.descriptor.provenance["reduction_order"] == [0, 1]
    assert package.descriptor.provenance["lse_checkpoint_policy"] == "apple7_recompute"
    assert package.descriptor.provenance["schedule_digest"] == artifact.schedule_digest


@_needs_opt
@pytest.mark.parametrize(
    "field,value",
    [
        ("target", "x86"),
        ("architecture", "gfx1151"),
        ("lse_checkpoint_policy", "save_lse"),
        ("lse_checkpoint_selection", "saved"),
        ("window_left", 4),
    ],
)
def test_apple_gpu_backward_rejects_foreign_contract(
    tmp_path, monkeypatch, field, value
) -> None:
    """Each guarded field is load-bearing, not decoration.

    Mutating exactly one field of an otherwise valid Apple artifact must be
    refused — otherwise a sibling's schedule or policy could be packaged under
    Apple's ABI.  Two layers refuse, and both are fail-closed: the LSE policy
    and selection are *digest-bound into the Tile text*, so tampering with them
    trips the artifact's own content-identity check before the Apple guard is
    even reached; target/architecture/window are caught by the guard itself.
    """
    from tessera.compiler import apple_native

    dylib = tmp_path / "libTesseraAppleRuntime.dylib"
    dylib.write_bytes(b"apple-runtime-image")
    monkeypatch.setattr(apple_native, "_runtime_library_path", lambda: dylib)
    artifact = lower_scheduled_attention_backward(
        _apple_module(), target="apple_gpu"
    )
    foreign = dataclasses.replace(artifact, **{field: value})
    digest_bound = field in {"lse_checkpoint_policy", "lse_checkpoint_selection"}
    expected = "stale" if digest_bound else "apple7"
    with pytest.raises(ValueError, match=expected):
        apple_native.package_scheduled_attention_backward(
            foreign, pipeline_name="tessera-lower-to-apple_gpu"
        )


@pytest.mark.hardware_apple_gpu
@_needs_opt
@pytest.mark.parametrize(
    "hq,hkv,sq,sk,causal",
    [
        (4, 2, 8, 8, False),    # GQA
        (4, 4, 16, 16, False),  # MHA
        (4, 1, 8, 8, False),    # MQA
        (2, 2, 8, 8, True),     # causal
    ],
)
def test_apple_gpu_backward_executes_and_matches_analytic_vjp(
    hq, hkv, sq, sk, causal
) -> None:
    """All three gradients must match an independently derived analytic VJP."""
    from tessera.compiler import apple_native
    from tessera.runtime import RuntimeArtifact, launch

    if not apple_native.tools_available():
        pytest.skip("Apple GPU runtime dylib unavailable")
    b, d, scale = 1, 16, 0.25
    artifact = lower_scheduled_attention_backward(
        _apple_module(b=b, hq=hq, hkv=hkv, sq=sq, sk=sk, d=d, causal=causal,
                      scale=scale),
        target="apple_gpu",
    )
    package = apple_native.package_scheduled_attention_backward(
        artifact, pipeline_name="tessera-lower-to-apple_gpu"
    )
    runtime_artifact = RuntimeArtifact(
        metadata={"target": "apple_gpu"}, native_image=package.image,
        launch_descriptor=package.descriptor, tile_ir=package.tile_ir,
        target_ir=package.target_ir,
    )
    rng = np.random.default_rng(5811)
    q = np.ascontiguousarray(rng.standard_normal((b, hq, sq, d)), dtype=np.float32)
    key = np.ascontiguousarray(rng.standard_normal((b, hkv, sk, d)), dtype=np.float32)
    value = np.ascontiguousarray(rng.standard_normal((b, hkv, sk, d)), dtype=np.float32)
    do = np.ascontiguousarray(rng.standard_normal((b, hq, sq, d)), dtype=np.float32)
    d_q = np.zeros_like(q)
    d_k = np.zeros_like(key)
    d_v = np.zeros_like(value)

    result = launch(runtime_artifact, {
        "do": do, "q": q, "k": key, "v": value,
        "dq": d_q, "dk": d_k, "dv": d_v,
    })

    assert result["ok"] is True, result.get("reason")
    assert result.get("execution_kind") == "native_gpu", result
    expected_q, expected_k, expected_v = _apple_backward_reference(
        do, q, key, value, scale=scale, causal=causal
    )
    np.testing.assert_allclose(d_q, expected_q, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(d_k, expected_k, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(d_v, expected_v, rtol=2e-4, atol=2e-5)


@pytest.mark.hardware_apple_gpu
@_needs_opt
def test_apple_gpu_backward_split_route_is_bit_deterministic() -> None:
    """The declared ascending fixed-order reduction implies bit-identical repeats."""
    from tessera.compiler import apple_native
    from tessera.runtime import RuntimeArtifact, launch

    if not apple_native.tools_available():
        pytest.skip("Apple GPU runtime dylib unavailable")
    artifact = lower_scheduled_attention_backward(
        _apple_module(hq=4, hkv=2, sq=16, sk=16), target="apple_gpu"
    )
    package = apple_native.package_scheduled_attention_backward(
        artifact, pipeline_name="tessera-lower-to-apple_gpu"
    )
    runtime_artifact = RuntimeArtifact(
        metadata={"target": "apple_gpu"}, native_image=package.image,
        launch_descriptor=package.descriptor, tile_ir=package.tile_ir,
        target_ir=package.target_ir,
    )
    rng = np.random.default_rng(991)
    q = np.ascontiguousarray(rng.standard_normal((1, 4, 16, 16)), dtype=np.float32)
    key = np.ascontiguousarray(rng.standard_normal((1, 2, 16, 16)), dtype=np.float32)
    value = np.ascontiguousarray(rng.standard_normal((1, 2, 16, 16)), dtype=np.float32)
    do = np.ascontiguousarray(rng.standard_normal((1, 4, 16, 16)), dtype=np.float32)
    runs = []
    for _ in range(3):
        grads = (np.zeros_like(q), np.zeros_like(key), np.zeros_like(value))
        result = launch(runtime_artifact, {
            "do": do, "q": q, "k": key, "v": value,
            "dq": grads[0], "dk": grads[1], "dv": grads[2],
        })
        assert result["ok"] is True, result.get("reason")
        runs.append(tuple(g.copy() for g in grads))
    for later in runs[1:]:
        for first_grad, later_grad in zip(runs[0], later):
            # Bit-identical, not merely close: atomics would fail this.
            assert np.array_equal(first_grad, later_grad)


@_needs_opt
@pytest.mark.parametrize("target", ["x86", "rocm_gfx1151"])
def test_attention_backward_schedule_is_one_content_addressed_three_result_program(target: str) -> None:
    module = _x86_module(4, 2, 17, 19) if target == "x86" else _module(1, 4, 2, 17, 19, 16)
    artifact = lower_scheduled_attention_backward(module, target=target)
    assert artifact.schedule_ir.count("schedule.attention_backward") == 1
    assert artifact.tile_ir.count("tile.attention_backward_kernel") == 1
    assert "tessera_attn.backward" not in artifact.tile_ir
    assert artifact.output_names == ("dq", "dk", "dv")
    assert artifact.reduction_order == (0, 1)
    assert artifact.workspace_bytes > 0
    artifact.validate()


def test_attention_backward_fails_closed_for_unmeasured_rdna4_profiles() -> None:
    module = _module(1, 4, 2, 17, 19, 16)
    assert not supports_scheduled_attention_backward(module, target="rocm_gfx1200")
    assert not supports_scheduled_attention_backward(module, target="rocm_gfx1250")
    with pytest.raises(ValueError, match="architecture-owned profiles"):
        lower_scheduled_attention_backward(module, target="rocm_gfx1200")


@_needs_opt
def test_attention_backward_rejects_stale_multi_output_identity() -> None:
    artifact = lower_scheduled_attention_backward(_x86_module(2, 2, 16, 16), target="x86")
    with pytest.raises(ValueError, match="content identity"):
        dataclasses.replace(artifact, schedule_digest="0" * 64).validate()


@pytest.mark.parametrize(
    ("hq", "hkv", "sq", "sk"),
    [(2, 2, 16, 16), (4, 2, 17, 19), (4, 1, 15, 21)],
)
@pytest.mark.hardware_avx512
@_needs_opt
def test_zen5_scheduled_attention_backward_exact_mha_gqa_mqa(
    hq: int, hkv: int, sq: int, sk: int
) -> None:
    # `x86_64` is the ISA FAMILY, not the feature. This gate used to accept any
    # x86_64 host, so on The-Super-Bear -- a Zen 2 Threadripper with AVX2 and
    # no AVX-512 -- it ran, packaged an `x86_64_avx512` image, and died inside
    # the runtime's (correct) fail-closed admission check with "not executable
    # on this host". Absent hardware must skip, not fail: the raise is the
    # runtime doing its job, and reading it as a test failure is the same
    # mistake as gating on `hardware_amx` to mean "x86 hardware".
    artifact = lower_scheduled_attention_backward(_x86_module(hq, hkv, sq, sk), target="x86")
    package = package_x86(artifact, pipeline_name="tessera-lower-to-x86")
    rng = np.random.default_rng(20260805 + hkv + sq)
    d = artifact.dims[-1]
    q = rng.normal(0, 0.2, (1, hq, sq, d)).astype(np.float32)
    key = rng.normal(0, 0.2, (1, hkv, sk, d)).astype(np.float32)
    value = rng.normal(0, 0.2, key.shape).astype(np.float32)
    do = rng.normal(0, 0.2, q.shape).astype(np.float32)
    bias = rng.normal(0, 0.05, (1, hq, sq, sk)).astype(np.float32)
    expected, row_lse = _x86_reference(
        do, q, key, value, bias, scale=artifact.scale, window=3, softcap=4.0
    )
    outputs = (np.empty_like(q), np.empty_like(key), np.empty_like(value))
    actual = _submit_x86_native(
        package.image, package.descriptor,
        {"do": do, "q": q, "key": key, "v": value, "bias": bias,
         "row_lse": row_lse, "dq": outputs[0], "dk": outputs[1], "dv": outputs[2]},
        {"B": 1, "Hq": hq, "Hkv": hkv, "Sq": sq, "Sk": sk, "D": d, "Dv": d},
        None,
    )
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, reference, rtol=2e-4, atol=3e-5)


@_needs_opt
@pytest.mark.compiler_rocm
def test_gfx1151_scheduled_attention_backward_packages_exact_tile_program(monkeypatch) -> None:
    import tessera.compiler.rocm_native as rocm_native

    # `native_packaging_available` exists for exactly this, and says so in its
    # own docstring: checking tools without also checking for AMD clang yields
    # a RuntimeError that "reads as a broken test on any host without ROCm
    # rather than an absent toolchain". This test did not call it, so on the
    # Mac it failed with `ROCm native packaging requires AMD clang ...` instead
    # of skipping.
    if not rocm_native.native_packaging_available():
        pytest.skip("gfx1151 native packaging requires AMD clang (OCML/OCKL/OCLC)")

    artifact = lower_scheduled_attention_backward(
        _module(1, 4, 2, 17, 19, 16), target="rocm_gfx1151"
    )
    def reject_graph_reentry(_module):
        raise AssertionError("scheduled Tile packaging re-entered Graph classification")

    monkeypatch.setattr(rocm_native, "_attention_backward_contract", reject_graph_reentry)
    program = package_rocm(artifact, pipeline_name="tessera-lower-to-rocm")
    assert len(program.descriptors) == 5
    assert program.descriptors[0].provenance["tile_ir_digest"] == artifact.tile_digest
    assert program.descriptors[0].provenance["lse_checkpoint"] == "recompute"


@pytest.mark.compiler_rocm
@pytest.mark.hardware_rocm
@pytest.mark.parametrize(
    ("hq", "hkv", "sq", "sk"),
    [(2, 2, 16, 16), (4, 2, 17, 19), (4, 1, 15, 21)],
)
@_needs_opt
def test_gfx1151_scheduled_attention_backward_exact_mha_gqa_mqa(
    hq: int, hkv: int, sq: int, sk: int
) -> None:
    if not _rocm_wmma_runtime_available():
        pytest.skip("a WSL-visible gfx1151 device is unavailable")
    artifact = lower_scheduled_attention_backward(
        _module(1, hq, hkv, sq, sk, 16), target="rocm_gfx1151"
    )
    program = package_rocm(artifact, pipeline_name="tessera-lower-to-rocm")
    rng = np.random.default_rng(20260805 + hkv + sk)
    q = rng.normal(0, 0.2, (1, hq, sq, 16)).astype(np.float16)
    key = rng.normal(0, 0.2, (1, hkv, sk, 16)).astype(np.float16)
    value = rng.normal(0, 0.2, key.shape).astype(np.float16)
    do = rng.normal(0, 0.2, q.shape).astype(np.float16)
    bias = rng.normal(0, 0.05, (1, hq, sq, sk)).astype(np.float32)
    outputs = (np.empty(q.shape, np.float32), np.empty(key.shape, np.float32), np.empty(value.shape, np.float32))
    result = _submit_rocm_gfx1151_attention_backward_program(
        program,
        {"do": do, "q": q, "key": key, "v": value, "bias": bias,
         "dq": outputs[0], "dk": outputs[1], "dv": outputs[2]},
        warmup=1, iterations=1,
    )
    expected = reference_attention_backward_split_reduced(
        do, q, key, value, split_count=2, scale=math.sqrt(1 / 16), bias=bias,
        causal=True, window_left=8, window_right=0, softcap=8.0,
        dropout_seed=37,
    )
    for observed, reference in zip(result["outputs"], expected, strict=True):
        np.testing.assert_allclose(observed, reference, rtol=0, atol=2e-2)


@_needs_opt
def test_backward_schedule_identity_distinguishes_adjacent_f32_policy():
    module = _x86_module(2, 1, 3, 4)
    module.functions[0].body[0].kwargs["scale"] = 0.5
    first = lower_scheduled_attention_backward(module, target="x86")
    module.functions[0].body[0].kwargs["scale"] = 0.5 + 2 ** -24
    second = lower_scheduled_attention_backward(module, target="x86")
    assert first.schedule_digest != second.schedule_digest
