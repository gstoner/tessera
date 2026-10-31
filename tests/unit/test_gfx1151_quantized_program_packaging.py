"""gfx1151 packaging and frontend of the native quantized MLP (host-free).

GFX1151-QUANTIZED-MLP-20261030, slices 3-4. The program's architecture is the
one its source Graph names: the C++ export, every packager, the image checks,
the LDS variant boxes, the portable record and the public frontend take it from
there. Slice 4 promoted the quantizer and SiLU members on gfx1151 in both
promotion mirrors (Python ``promoted_families`` and the C++ pipeline contract)
on Princess-Luna device proof. These tests stop at gfx1151 HSACOs and
refusals; the execution proof is the owning-device fixtures under
``tests/device/rocm/`` (``test_native_quantized_mlp_program.py`` and the bounded
``test_public_quantized_mlp_*_bound_device.py``), run on gfx1151.
"""
import re

import pytest

from tessera.compiler.scheduled_matmul import find_tessera_opt

pytestmark = pytest.mark.compiler_route

_BOUNDS = {
    "v4": "",
    "v5": "tessera.native.quantized_m_bound = 64 : i64, ",
    "v6": "tessera.native.quantized_m_bound = 64 : i64, tessera.native.quantized_k_bound = 512 : i64, ",
    "v7": "tessera.native.quantized_m_bound = 64 : i64, tessera.native.quantized_n_bound = 96 : i64, ",
    "v8": "tessera.native.quantized_m_bound = 64 : i64, tessera.native.quantized_h_bound = 256 : i64, ",
}


def _gfx1151(text):
    assert text.count('"rocm_gfx1201"') == 1 and text.count('tessera.arch = "gfx1201"') == 1
    return text.replace('"rocm_gfx1201"', '"rocm_gfx1151"').replace(
        'tessera.arch = "gfx1201"', 'tessera.arch = "gfx1151"')


def _mlp(schema="v4", arch="gfx1151"):
    from tests.unit.test_native_quantized_mlp_program import source
    text = source(37, 128, 256, 64)
    if arch == "gfx1151":
        text = _gfx1151(text)
    return text.replace("tessera.native.quantized_product", _BOUNDS[schema] + "tessera.native.quantized_product", 1)


def _plan(schema="v4", arch="gfx1151"):
    from tessera.compiler.rocm_quantized_product_program import _export
    return _export(find_tessera_opt(), _mlp(schema, arch))


@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("schema", sorted(_BOUNDS))
def test_gfx1151_mlp_exports_every_member_on_its_own_architecture(schema):
    from tessera.compiler.rocm_fp8_blockscale import module_arch
    from tessera.compiler.rocm_quantized_mlp_program import _components
    plan = _plan(schema)
    assert plan["schema"] == f"tessera.native.quantized_product.{schema}"
    for component in _components(plan):
        assert module_arch(component["source_graph_ir"]) == "gfx1151"
        quantizer, expansion, product = component["member_graphs"][:3]
        # Owned members keep the exact target; the product is generic ROCm.
        assert 'tessera.target = "rocm_gfx1151"' in quantizer and 'tessera.target = "rocm_gfx1151"' in expansion
        assert module_arch(product) == "gfx1151" and "rocm_gfx1201" not in product
        assert "gfx1201" not in "".join(component["member_graphs"])


@pytest.mark.usefixtures("production_compiler")
def test_gfx1201_mlp_export_is_unchanged_by_the_architecture_threading():
    from tessera.compiler.rocm_quantized_mlp_program import _components
    for component in _components(_plan(arch="gfx1201")):
        assert "gfx1151" not in "".join(component["member_graphs"])
        assert 'tessera.target = "rocm_gfx1201"' in component["member_graphs"][0]


@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("target,arch", [("rocm_gfx1151", "gfx1201"), ("rocm_gfx1201", "gfx1151"),
                                         ("rocm_gfx1200", "gfx1200")])
def test_export_refuses_any_other_target_architecture_pair(target, arch):
    from tessera.compiler.rocm_quantized_product_program import _export
    text = _mlp().replace('"rocm_gfx1151"', f'"{target}"').replace(
        'tessera.arch = "gfx1151"', f'tessera.arch = "{arch}"')
    with pytest.raises(RuntimeError, match="resident quantized product requires"):
        _export(find_tessera_opt(), text)


@pytest.mark.usefixtures("production_compiler")
def test_standalone_nvfp4_program_stays_gfx1201_only():
    """Only the quantized gate/up program, composing its own tail, admits gfx1151."""
    from tessera.compiler.native_nvfp4_program import export_native_nvfp4_program
    from tests.unit.test_native_nvfp4_expansion_program import source_graph
    with pytest.raises(RuntimeError, match="requires one gfx1201 Graph root"):
        export_native_nvfp4_program(_gfx1151(source_graph()))


@pytest.mark.parametrize("schema", sorted(_BOUNDS))
def test_gfx1151_product_images_are_register_panel_f16_wmma(rocm_image_toolchain, schema):
    """Every product image (static, and each bounded frame's shape-free image)
    compiles to a gfx1151 HSACO holding only F16 WMMA, with no LDS variants."""
    from tessera.compiler import rocm_quantized_product_program as qp
    from tessera.compiler.rocm_fp8_blockscale import admitted_rows_graph
    from tessera.compiler.rocm_quantized_mlp_program import _components
    from tests._support.rocm_isa import disassemble
    plan = _plan(schema)
    bounded = schema != "v4"
    for component, runtime_k, runtime_n in zip(_components(plan), ("k_frame" in plan, "h_frame" in plan),
                                               ("h_frame" in plan, "n_frame" in plan)):
        _, shape = qp._frame_shape(component)
        member = component["member_graphs"][2]
        child = admitted_rows_graph(member, global_staging=runtime_k or runtime_n) if bounded else member
        _, product, checked, _ = qp._compile_product(find_tessera_opt(), shape, child, arbitrary_m=bounded)
        panel = qp._check_product_image(shape, product, checked, bounded)
        image = product.image
        assert (image.target, image.architecture) == ("rocm_gfx1151", "gfx1151")
        assert (panel["staging"], panel["warps"]) == ("global", 1)
        assert product.descriptor.provenance["physical_route"].startswith("gfx1151_register_wmma_blockscale_")
        assert product.descriptor.provenance["runtime_shape_image"] is bounded
        text = disassemble(image.payload, chip="gfx1151")
        assert set(re.findall(r"\bv_wmma_\w+", text)) == {"v_wmma_f32_16x16x16_f16"}
        assert not re.search(r"\bv_\w*(fp8|bf8)\w*", text)
        vary = qp._vary(shape, runtime_k, False, None) if not runtime_n else None
        assert qp._variant_boxes(shape, vary, "gfx1151") == ()


@pytest.mark.parametrize("runtime_k,runtime_n", [(False, False), (True, False), (False, True)])
def test_gfx1151_expansion_images_package_for_every_frame(rocm_image_toolchain, runtime_k, runtime_n):
    from tessera.compiler.rocm_nvfp4_expand_native import package_nvfp4_expansion
    from tessera.compiler.rocm_quantized_mlp_program import _components
    gate, _ = _components(_plan("v5"))
    package = package_nvfp4_expansion(gate["member_graphs"][1], runtime_k=runtime_k, runtime_n=runtime_n)
    package.validate()
    assert (package.native.image.target, package.native.image.architecture) == ("rocm_gfx1151", "gfx1151")
    assert package.runtime_k is runtime_k and package.runtime_n is runtime_n


@pytest.mark.parametrize("schema", ["v4", "v5"])
def test_gfx1151_program_packages_every_member(rocm_image_toolchain, schema):
    from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph
    package = package_quantized_product_graph(_mlp(schema))
    assert package.arch == "gfx1151"
    assert package.gate.product.image.target == "rocm_gfx1151"


def _unpromote(monkeypatch):
    """Withdraw the gfx1151 member promotion, as for a target without device proof."""
    from tessera.compiler import rocm_pipeline
    promoted = rocm_pipeline.promoted_families
    monkeypatch.setattr(rocm_pipeline, "promoted_families",
                        lambda arch: promoted(arch) - ({"block_quant", "parent_view"} if arch == "gfx1151" else set()))


def test_architecture_owned_families_are_promoted_on_both_owning_arches():
    """Slice 4: promoted on Princess-Luna device proof, in both mirrors at once.
    They stay architecture-owned, so no other arch and no generic lane inherits them."""
    from tessera.compiler.rocm_pipeline import ARCHITECTURE_OWNED_FAMILIES, generic_lane_families, promoted_families
    assert {"block_quant", "parent_view"} <= ARCHITECTURE_OWNED_FAMILIES
    assert not {"block_quant", "parent_view"} & generic_lane_families()
    for arch in ("gfx1151", "gfx1201"):
        assert {"block_quant", "parent_view", "matmul", "quant_fp"} <= promoted_families(arch)
    for arch in ("gfx1200", "gfx1250"):
        assert not {"block_quant", "parent_view"} & promoted_families(arch)


def test_typed_directive_is_a_directive_level_matmul_request():
    from tessera.compiler.rocm_pipeline import ROCMExecutablePipeline, ROCMInputLevel
    typed = ROCMExecutablePipeline(family="matmul", arch="gfx1151",
                                   input_level=ROCMInputLevel.DIRECTIVE, typed_directive=True)
    assert "typed-directive=true" in typed.pass_pipeline()
    plain = ROCMExecutablePipeline(family="matmul", arch="gfx1151", input_level=ROCMInputLevel.DIRECTIVE)
    # Carried only when set: every existing pipeline string and cache key is unchanged.
    assert "typed-directive" not in plain.pass_pipeline()
    assert typed.cache_key() != plain.cache_key()
    for kwargs in ({"input_level": ROCMInputLevel.TILE}, {"family": "softmax", "input_level": ROCMInputLevel.DIRECTIVE},
                   {"typed_directive": 1, "input_level": ROCMInputLevel.DIRECTIVE}):
        options = {"family": "matmul", "arch": "gfx1151", "typed_directive": True, **kwargs}
        with pytest.raises(ValueError, match="typed_directive"):
            ROCMExecutablePipeline(**options)


def test_module_arch_is_a_semantic_key():
    from tessera.compiler.rocm_fp8_blockscale import module_arch
    assert module_arch('module attributes {tessera.arch = "gfx1151", tessera.target = "rocm"} {\n}') == "gfx1151"
    for text in ('module attributes {tessera.target = "rocm"} {\n}',
                 'module attributes {tessera.arch = "gfx1200"} {\n}', ""):
        with pytest.raises(ValueError, match="no admitted architecture"):
            module_arch(text)


def test_gfx1151_admission_is_scoped_to_the_quantized_program(monkeypatch):
    """The registry rows are unchanged; only the authored program admits its
    ops on gfx1151, and the generic ``rocm`` name inherits nothing."""
    # Generic ``rocm`` resolves to TESSERA_ROCM_CHIP, else gfx1151's registry rows.
    monkeypatch.delenv("TESSERA_ROCM_CHIP", raising=False)
    import tessera as ts
    from tessera.compiler.capabilities import supports_op
    from benchmarks.rocm.benchmark_public_quantized_mlp import make_mlp
    for op, dtype in (("tessera.scaled_matmul", "fp8_e4m3"), ("tessera.reshape", "fp8_e4m3"),
                      ("tessera.cast", "fp32"), ("tessera.nvfp4_expand", "uint8")):
        assert not supports_op("rocm_gfx1151", op, dtype=dtype).supported, (op, dtype)
    assert supports_op("rocm_gfx1201", "tessera.nvfp4_expand", dtype="uint8").runtime_status == "ready"
    with pytest.raises(Exception, match="LEGALITY_TARGET_CAPABILITY"):
        ts.jit(target="rocm")(make_mlp(3, 32, 48, 19))


def test_gfx1151_standalone_members_stay_refused_at_decoration():
    import tessera as ts
    from tests.unit.test_rocm_nvfp4_expansion_jit import expand
    with pytest.raises(Exception, match="LEGALITY_TARGET_CAPABILITY"):
        ts.jit(expand._fn, target="rocm_gfx1151")


def _public(target="rocm_gfx1151", **jit):
    import tessera as ts
    from benchmarks.rocm.benchmark_public_quantized_mlp import inputs, make_mlp
    return ts.jit(target=target, **jit)(make_mlp(3, 32, 48, 19)), inputs(3, 32, 48, 19)


def test_gfx1151_public_mlp_decorates():
    fn, _ = _public()
    from tessera.compiler.rocm_quantized_product_frontend import requests_quantized_product
    assert requests_quantized_product(fn.graph_ir)


def test_unpromoted_member_public_call_refuses_with_its_reason(monkeypatch):
    """No fallback computes the program (it has no eager form for its BF16
    roots), so a target missing a member promotion refuses by name instead of
    falling through."""
    from tessera.compiler.fallback import FallbackReason
    _unpromote(monkeypatch)
    fn, roots = _public()
    with pytest.raises(ValueError, match=r"members \['block_quant', 'parent_view'\] have no "
                                         r"exact-device promotion on rocm_gfx1151"):
        fn(*roots)
    assert fn.last_fallback_reason is FallbackReason.CAPABILITY_NOT_READY
    assert getattr(fn, "_rocm_quantized_product_last_program", None) is None


def test_unpromoted_member_native_required_call_raises_the_promotion_gap(monkeypatch):
    from tessera.compiler.fallback import TesseraNativeRequiredError
    _unpromote(monkeypatch)
    fn, roots = _public(native_required=True)
    with pytest.raises(TesseraNativeRequiredError, match=r"\['block_quant', 'parent_view'\]"):
        fn(*roots)


@pytest.mark.parametrize("bound", [None, 8])
def test_unpromoted_member_explicit_compile_reports_the_refusal(rocm_image_toolchain, bound, monkeypatch):
    _unpromote(monkeypatch)
    fn, roots = _public()
    with pytest.raises(ValueError, match="no exact-device promotion on gfx1151"):
        fn.compile_native_quantized_program(*roots, m_bound=bound)
    assert getattr(fn, "_rocm_quantized_product_bounded_program", None) is None


def test_frontend_source_names_the_requested_target():
    from tessera.compiler.rocm_quantized_product_frontend import _source
    fn, roots = _public()
    # The call path packages the traced capture, as `_try_rocm_quantized_product_call` does.
    module, _ = fn._trace_frontend_capture(tuple(roots), {})
    text = _source(module, "rocm_gfx1151")
    assert 'tessera.target = "rocm_gfx1151"' in text and 'tessera.arch = "gfx1151"' in text
    with pytest.raises(ValueError, match="native quantized product requires one of"):
        _source(module, "rocm_gfx1200")
