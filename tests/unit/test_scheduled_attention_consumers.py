from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import rocm_native, scheduled_attention, x86_native
from tessera.compiler.driver import compile_graph_module
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.scheduled_attention import ScheduledAttentionArtifact
from tessera.compiler.scheduled_matmul import find_tessera_opt


def _module(*, target: str, bias: bool = False, query_rows: int = 17) -> GraphIRModule:
    rocm = target == "rocm"
    dtype, element = ("fp16", "f16") if rocm else ("fp32", "f32")
    b, hq, hkv, sk, d, dv = (1, 4, 2, 19, 64, 64) if rocm else (1, 2, 2, 7, 4, 3)
    q = IRType(f"tensor<{b}x{hq}x{query_rows}x{d}x{element}>", tuple(map(str, (b, hq, query_rows, d))), dtype)
    k = IRType(f"tensor<{b}x{hkv}x{sk}x{d}x{element}>", tuple(map(str, (b, hkv, sk, d))), dtype)
    v = IRType(f"tensor<{b}x{hkv}x{sk}x{dv}x{element}>", tuple(map(str, (b, hkv, sk, dv))), dtype)
    o = IRType(f"tensor<{b}x{hq}x{query_rows}x{dv}xf32>", tuple(map(str, (b, hq, query_rows, dv))), "fp32")
    args = [IRArg("q", q), IRArg("k", k), IRArg("v", v)]
    operands, operand_types = ["%q", "%k", "%v"], [str(q), str(k), str(v)]
    if bias:
        bias_type = IRType(f"tensor<{b}x{hq}x{query_rows}x{sk}xf32>", tuple(map(str, (b, hq, query_rows, sk))), "fp32")
        args.append(IRArg("bias", bias_type))
        operands.append("%bias")
        operand_types.append(str(bias_type))
    kwargs = {"scale": 0.125 if rocm else 0.5, "causal": rocm}
    if rocm:
        kwargs["window"] = (64, 0)
    elif bias:
        kwargs.update({"window": 3, "softcap": 4.0})
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name=f"{target}_scheduled_attention",
                args=args,
                result_types=[o],
                body=[
                    IROp(
                        result="o",
                        op_name="tessera.flash_attn",
                        operands=operands,
                        operand_types=operand_types,
                        result_type=str(o),
                        kwargs=kwargs,
                    )
                ],
                return_values=["%o"],
            )
        ]
    )


def _apple_module(
    *,
    dims: tuple[int, int, int, int, int] = (1, 4, 2, 8, 8),
    head_dim: int = 16,
    causal: bool = False,
    window: int = -1,
    value_dim: int | None = None,
) -> GraphIRModule:
    """Rank-4 f32 module inside the Apple GQA MSL ABI envelope.

    Kept separate from ``_module``: the x86 shape uses head_dim != value_dim,
    which the Apple ABI does not accept (it shares one head/value dim).
    """
    b, hq, hkv, sq, sk = dims
    dv = head_dim if value_dim is None else value_dim
    def _t(shape: tuple[int, ...], element: str = "f32", dtype: str = "fp32") -> IRType:
        text = "tensor<" + "x".join(map(str, shape)) + f"x{element}>"
        return IRType(text, tuple(map(str, shape)), dtype)
    q = _t((b, hq, sq, head_dim))
    k = _t((b, hkv, sk, head_dim))
    v = _t((b, hkv, sk, dv))
    o = _t((b, hq, sq, dv))
    return GraphIRModule(functions=[GraphIRFunction(
        name="apple_scheduled_attention",
        args=[IRArg("q", q), IRArg("k", k), IRArg("v", v)],
        result_types=[o],
        body=[IROp(
            result="o", op_name="tessera.flash_attn",
            operands=["%q", "%k", "%v"],
            operand_types=[str(q), str(k), str(v)],
            result_type=str(o),
            kwargs={"scale": 0.25, "causal": causal, "window": window,
                    "softcap": 0.0, "dropout_p": 0.0, "dropout_seed": 0},
        )],
        return_values=["%o"],
    )])


def _artifact(*, target: str) -> ScheduledAttentionArtifact:
    rocm = target == "rocm"
    apple = target == "apple_gpu"
    digest = hashlib.sha256(f"attention-{target}".encode()).hexdigest()
    policy = (
        "apple7_recompute" if apple else "gfx1151_auto_128" if rocm else "save_lse"
    )
    selection = "saved" if (not rocm and not apple) else "recompute"
    storage, dtype = ("f16", "fp16") if rocm else ("f32", "fp32")
    dims = (
        (1, 4, 2, 8, 8, 16, 16) if apple
        else (1, 4, 2, 17, 19, 64, 64) if rocm
        else (1, 2, 2, 5, 7, 4, 3)
    )
    workgroup = 256 if rocm else 1
    tile_ir = f'''module {{
  llvm.func @{target}_scheduled_attention() {{
    tile.attention_kernel {{tessera.schedule_hash = "{digest}",
      tessera.tile_q = {dims[3]} : i64, tessera.tile_kv = 16 : i64,
      tessera.workgroup_size = {workgroup} : i64,
      head_dim = {dims[5]} : i64, value_dim = {dims[6]} : i64,
      tessera.attention_recurrence = "rank4_batch_query_head_kv_online_softmax_v1",
      tessera.backward_lse_policy = "{policy}",
      tessera.backward_lse_selection = "{selection}"}}
    llvm.return
  }}
}}'''
    return ScheduledAttentionArtifact(
        graph_ir=f'module attributes {{tessera.target = "{target}"}} {{}}',
        schedule_ir=f'''module {{
  %graph = "tessera.flash_attn"() {{schedule.artifact_hash = "{digest}"}} : () -> tensor<1xf32>
  %scheduled = schedule.attention %graph {{artifact_hash = "{digest}"}} : tensor<1xf32> -> tensor<1xf32>
  schedule.artifact {{hash = "{digest}"}}
}}''',
        semantic_ir="""module { scf.for { "tessera.streaming_attention"() {tessera.attention_distribution = "batch"} : () -> ()
          "tessera_attn.streaming_update"() {tessera.attention_distribution = "query_head"} : () -> ()
          "tessera_attn.boundary_mask"() : () -> () } }""",
        tile_ir=tile_ir,
        target=target,
        architecture="apple7" if apple else "gfx1151" if rocm else "zen5-avx512",
        function_name=f"{target}_scheduled_attention",
        q_name="q",
        k_name="k",
        v_name="v",
        bias_name=None,
        output_name="o",
        dtype=dtype,
        storage=storage,
        accum="f32",
        dims=dims,
        scale=0.125 if rocm else 0.5,
        causal=rocm,
        window_left=64 if rocm else -1,
        window_right=0 if rocm else -1,
        softcap=0.0,
        dropout_p=0.0,
        dropout_seed=0,
        tile_q=dims[3],
        tile_kv=16,
        workgroup_size=workgroup,
        recurrence="rank4_batch_query_head_kv_online_softmax_v1",
        backward_lse_policy=policy,
        backward_lse_selection=selection,
        schedule_digest=digest,
    )


def test_flash_attention_graph_serializes_operand_segments() -> None:
    text = _module(target="x86").to_mlir(canonical=True)
    assert "operandSegmentSizes = array<i32: 1, 1, 1, 0>" in text
    biased = _module(target="x86", bias=True).to_mlir(canonical=True)
    assert "operandSegmentSizes = array<i32: 1, 1, 1, 1>" in biased


@pytest.mark.skipif(find_tessera_opt() is None, reason="production tessera-opt unavailable")
@pytest.mark.parametrize("target", ["x86", "rocm_gfx1151"])
def test_attention_lowers_through_one_content_addressed_tile_artifact(target: str) -> None:
    artifact = scheduled_attention.lower_scheduled_attention(
        _module(target="x86" if target == "x86" else "rocm"), target=target
    )
    assert artifact.schedule_ir.count("schedule.attention") == 1
    assert artifact.tile_ir.count("tile.attention_kernel") == 1
    assert "tessera.flash_attn" not in artifact.tile_ir
    assert "schedule." not in artifact.tile_ir
    assert artifact.recurrence in artifact.tile_ir


@pytest.mark.skipif(find_tessera_opt() is None, reason="production tessera-opt unavailable")
def test_attention_modifiers_survive_the_shared_recurrence() -> None:
    artifact = scheduled_attention.lower_scheduled_attention(_module(target="x86", bias=True), target="x86")
    assert "tessera_attn.score_bias" in artifact.semantic_ir
    assert "tessera_attn.softcap" in artifact.semantic_ir
    assert "tessera_attn.boundary_mask" in artifact.semantic_ir
    assert "bias = true" in artifact.tile_ir
    assert "softcap = 4.000000e+00 : f32" in artifact.tile_ir


@pytest.mark.skipif(find_tessera_opt() is None, reason="production tessera-opt unavailable")
def test_attention_schedule_policy_tampering_fails_closed() -> None:
    artifact = scheduled_attention.lower_scheduled_attention(_module(target="x86"), target="x86")
    tampered = artifact.schedule_ir.replace(
        'backward_lse_selection = "saved"',
        'backward_lse_selection = "recompute"',
    )
    with pytest.raises(RuntimeError, match="altered after hashing"):
        scheduled_attention.run_tessera_opt(find_tessera_opt(), tampered, "--tessera-schedule-to-tile")


@pytest.mark.parametrize("target", ["rocm_gfx1200", "rocm_gfx1250"])
def test_rdna4_attention_fails_closed_without_profile(target: str) -> None:
    assert not scheduled_attention.supports_scheduled_attention(_module(target="rocm"), target=target)


def test_x86_packages_exact_attention_tile(monkeypatch) -> None:
    artifact = _artifact(target="x86")
    monkeypatch.setattr(
        x86_native,
        "_lower",
        lambda tile, symbol, family: (
            ("target", b"image", "compiler", "toolchain")
            if tile == artifact.tile_ir
            else pytest.fail("Tile artifact was resynthesized")
        ),
    )
    package = x86_native.package_scheduled_attention(artifact, pipeline_name="tessera-lower-to-x86")
    assert package.tile_ir == artifact.tile_ir
    assert package.descriptor.provenance["work_item"] == "E2E-REAL-5A"
    assert package.descriptor.provenance["backward_lse_policy"] == "save_lse"


def test_rocm_packages_exact_attention_tile(monkeypatch) -> None:
    artifact = _artifact(target="rocm")
    monkeypatch.setattr(
        rocm_native,
        "_compile_scheduled_attention_tile_ir",
        lambda tile: (
            ("target", "backend", b"hsaco", "compiler", "toolchain", (), "cold")
            if tile == artifact.tile_ir
            else pytest.fail("Tile artifact was resynthesized")
        ),
    )
    package = rocm_native.package_scheduled_attention(artifact, pipeline_name="tessera-lower-to-rocm")
    assert package.tile_ir == artifact.tile_ir
    assert package.descriptor.provenance["work_item"] == "E2E-REAL-5A"
    assert package.descriptor.provenance["backward_lse_selection"] == "recompute"


def test_apple_gpu_packages_exact_attention_tile(monkeypatch, tmp_path) -> None:
    from tessera.compiler import apple_native

    artifact = _artifact(target="apple_gpu")
    fake_dylib = tmp_path / "libTesseraAppleRuntime.dylib"
    fake_dylib.write_bytes(b"apple-runtime-image")
    monkeypatch.setattr(apple_native, "_runtime_library_path", lambda: fake_dylib)

    package = apple_native.package_scheduled_attention(
        artifact, pipeline_name="tessera-lower-to-apple_gpu"
    )
    # Consumes the shared launch tile verbatim — no Graph re-entry.
    assert package.tile_ir == artifact.tile_ir
    assert package.descriptor.entry_symbol == (
        "tessera_apple_gpu_flash_attn_variant_f32_status"
    )
    assert package.descriptor.provenance["work_item"] == "E2E-REAL-5A"
    assert package.descriptor.provenance["route"] == "apple_gpu_flash_attn_variant_f32"
    # Apple owns its recompute policy; it must not inherit x86's save_lse.
    assert package.descriptor.provenance["backward_lse_policy"] == "apple7_recompute"
    assert package.descriptor.provenance["backward_lse_selection"] == "recompute"
    assert package.descriptor.provenance["schedule_digest"] == artifact.schedule_digest
    assert package.descriptor.provenance["tile_ir_digest"] == artifact.tile_digest


def test_apple_gpu_attention_rejects_foreign_lse_policy(monkeypatch, tmp_path) -> None:
    from tessera.compiler import apple_native

    fake_dylib = tmp_path / "libTesseraAppleRuntime.dylib"
    fake_dylib.write_bytes(b"apple-runtime-image")
    monkeypatch.setattr(apple_native, "_runtime_library_path", lambda: fake_dylib)
    # An x86 artifact is structurally valid but carries save_lse/saved.
    with pytest.raises(ValueError, match="apple7 f32 recompute policy"):
        apple_native.package_scheduled_attention(
            _artifact(target="x86"), pipeline_name="tessera-lower-to-apple_gpu"
        )


@pytest.mark.skipif(find_tessera_opt() is None, reason="production tessera-opt unavailable")
def test_apple_gpu_attention_envelope_fails_closed() -> None:
    """Requests outside the Apple GQA ABI envelope must not be silently narrowed."""
    supports = scheduled_attention.supports_scheduled_attention
    # In-envelope: GQA, shared head/value dim, no window, no dropout.
    assert supports(_apple_module(), target="apple_gpu")
    assert supports(_apple_module(causal=True), target="apple_gpu")
    # head_dim != value_dim — the ABI shares one dim.
    assert not supports(_apple_module(value_dim=8), target="apple_gpu")
    # A live window: the MSL non-causal window is a symmetric half-window, which
    # is not the shared window_left/window_right semantics.
    assert not supports(_apple_module(window=4), target="apple_gpu")
    # q_heads must be a whole multiple of kv_heads.
    assert not supports(_apple_module(dims=(1, 3, 2, 8, 8)), target="apple_gpu")
    # D > 256 exceeds the kernel's TESSERA_GQA_MAX_D bound.
    assert not supports(_apple_module(head_dim=512), target="apple_gpu")


@pytest.mark.skipif(find_tessera_opt() is None, reason="production tessera-opt unavailable")
def test_apple_attention_lowers_through_one_content_addressed_tile_artifact() -> None:
    artifact = scheduled_attention.lower_scheduled_attention(
        _apple_module(), target="apple_gpu"
    )
    assert artifact.schedule_ir.count("schedule.attention") == 1
    assert artifact.tile_ir.count("tile.attention_kernel") == 1
    assert "tessera.flash_attn" not in artifact.tile_ir
    assert "schedule." not in artifact.tile_ir
    assert artifact.backward_lse_policy == "apple7_recompute"
    assert artifact.architecture == "apple7"


@pytest.mark.hardware_apple_gpu
@pytest.mark.skipif(find_tessera_opt() is None, reason="production tessera-opt unavailable")
@pytest.mark.parametrize(
    "dims,causal",
    [
        ((1, 4, 2, 8, 8), False),    # GQA
        ((2, 4, 4, 16, 16), False),  # MHA, batch 2
        ((1, 4, 1, 8, 8), False),    # MQA
        ((1, 2, 2, 8, 8), True),     # causal
    ],
)
def test_apple_gpu_scheduled_attention_executes_exact_artifact(dims, causal) -> None:
    from tessera.compiler import apple_native

    if not apple_native.tools_available():
        pytest.skip("Apple GPU runtime dylib unavailable")
    b, hq, hkv, sq, sk = dims
    head_dim = 16
    bundle = compile_graph_module(
        _apple_module(dims=dims, head_dim=head_dim, causal=causal),
        source_origin="e2e-real-5a-exact-device", target="apple_gpu",
        options={"package_native": True}, enable_tool_validation=False,
    )
    assert bundle.native_image is not None and bundle.launch_descriptor is not None
    assert bundle.tile is not None and bundle.target_ir is not None
    assert bundle.lineage_complete
    assert bundle.launch_descriptor.provenance["route"] == "apple_gpu_flash_attn_variant_f32"
    runtime_artifact = rt.RuntimeArtifact(
        metadata={"target": "apple_gpu"}, native_image=bundle.native_image,
        launch_descriptor=bundle.launch_descriptor, tile_ir=bundle.tile.text,
        target_ir=bundle.target_ir.text,
    )
    rng = np.random.default_rng(5901)
    q = np.ascontiguousarray(rng.standard_normal((b, hq, sq, head_dim)), dtype=np.float32)
    k = np.ascontiguousarray(rng.standard_normal((b, hkv, sk, head_dim)), dtype=np.float32)
    v = np.ascontiguousarray(rng.standard_normal((b, hkv, sk, head_dim)), dtype=np.float32)
    out = np.zeros((b, hq, sq, head_dim), dtype=np.float32)

    result = rt.launch(runtime_artifact, {"q": q, "k": k, "v": v, "o": out})

    assert result["ok"] is True, result.get("reason")
    # The status-returning ABI proves placement; a CPU reference cannot pass.
    assert result.get("execution_kind") == "native_gpu", result
    group = hq // hkv
    k_expanded, v_expanded = np.repeat(k, group, axis=1), np.repeat(v, group, axis=1)
    scores = np.einsum("bhqd,bhkd->bhqk", q, k_expanded) * 0.25
    if causal:
        offset = max(sk - sq, 0)
        rows = np.arange(sq)[:, None] + offset
        cols = np.arange(sk)[None, :]
        scores = np.where(cols <= rows, scores, -np.inf)
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    expected = np.einsum("bhqk,bhkd->bhqd", probs, v_expanded)
    np.testing.assert_allclose(out, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("target", ["x86", "rocm_gfx1151"])
def test_driver_records_adjacent_attention_lineage(monkeypatch, target: str) -> None:
    artifact = _artifact(target="x86" if target == "x86" else "rocm")
    monkeypatch.setattr(scheduled_attention, "lower_scheduled_attention", lambda module, *, target: artifact)
    if target == "x86":
        monkeypatch.setattr(x86_native, "_lower", lambda tile, symbol, family: ("target", b"image", "compiler", "toolchain"))
    else:
        monkeypatch.setattr(
            rocm_native,
            "_compile_scheduled_attention_tile_ir",
            lambda tile: ("target", "backend", b"hsaco", "compiler", "toolchain", (), "cold"),
        )
    bundle = compile_graph_module(
        _module(target="x86" if target == "x86" else "rocm"),
        source_origin="test",
        target=target,
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.request.graph_ir == artifact.graph_ir
    assert bundle.schedule is not None and bundle.schedule.text == artifact.schedule_ir
    assert bundle.tile is not None and bundle.tile.text == artifact.tile_ir
    assert bundle.lineage_complete
    assert bundle.launch_descriptor.provenance["work_item"] == "E2E-REAL-5A"


@pytest.mark.skipif(
    not x86_native.tools_available() or find_tessera_opt() is None,
    reason="x86 compiler/image unavailable",
)
def test_x86_scheduled_attention_executes_exact_artifact() -> None:
    module = _module(target="x86", query_rows=5)
    bundle = compile_graph_module(
        module,
        source_origin="e2e-real-5a-exact-device",
        target="x86",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    runtime_artifact = rt.RuntimeArtifact(
        metadata={"target": "x86"},
        native_image=bundle.native_image,
        launch_descriptor=bundle.launch_descriptor,
        tile_ir=bundle.tile.text,
        target_ir=bundle.target_ir.text,
    )
    rng = np.random.default_rng(5151)
    q = np.ascontiguousarray(rng.standard_normal((1, 2, 5, 4)), dtype=np.float32)
    k = np.ascontiguousarray(rng.standard_normal((1, 2, 7, 4)), dtype=np.float32)
    v = np.ascontiguousarray(rng.standard_normal((1, 2, 7, 3)), dtype=np.float32)
    output = np.zeros((1, 2, 5, 3), dtype=np.float32)
    result = rt.launch(
        runtime_artifact,
        {
            "q": q,
            "k": k,
            "v": v,
            "o": output,
            "B": 1,
            "Hq": 2,
            "Hkv": 2,
            "Sq": 5,
            "Sk": 7,
            "D": 4,
            "Dv": 3,
        },
    )
    scores = np.einsum("bhqd,bhkd->bhqk", q, k) * 0.5
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights /= np.sum(weights, axis=-1, keepdims=True)
    expected = np.einsum("bhqk,bhkd->bhqd", weights, v)
    assert result["ok"] is True, result.get("reason")
    np.testing.assert_allclose(output, expected, rtol=3e-5, atol=3e-5)


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_ROCM_E2E_DEVICE_TEST") != "1",
    reason="set TESSERA_ROCM_E2E_DEVICE_TEST=1 on the exact gfx1151 host",
)
def test_gfx1151_scheduled_attention_compiles_exact_artifact() -> None:
    bundle = compile_graph_module(
        _module(target="rocm"),
        source_origin="e2e-real-5a-exact-device",
        target="rocm_gfx1151",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.native_image is not None and bundle.launch_descriptor is not None
    assert bundle.tile.text.count("tile.attention_kernel") == 1
    assert bundle.launch_descriptor.provenance["work_item"] == "E2E-REAL-5A"
    from tessera.compiler.attention_contract import reference_streaming_attention

    rng = np.random.default_rng(5152)
    q = (rng.standard_normal((1, 4, 17, 64)) * 0.2).astype(np.float16)
    k = (rng.standard_normal((1, 2, 19, 64)) * 0.2).astype(np.float16)
    v = (rng.standard_normal((1, 2, 19, 64)) * 0.2).astype(np.float16)
    output = np.zeros((1, 4, 17, 64), dtype=np.float32)
    rt._submit_rocm_gfx1151_native(
        bundle.native_image,
        bundle.launch_descriptor,
        {"q": q, "k": k, "v": v, "o": output},
        {"Sq": 17, "Sk": 19, "Scale": 0.125, "Causal": 1, "Hq": 4, "KvRatio": 2, "Window": 64},
        None,
    )
    expected = reference_streaming_attention(
        q,
        k,
        v,
        block_size=16,
        scale=0.125,
        causal=True,
        window_left=64,
        window_right=0,
    )
    np.testing.assert_allclose(output, expected, rtol=3e-2, atol=3e-2)
