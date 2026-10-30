"""Per-active-M product images of a bounded native quantized MLP.

GFX1201-BOUNDED-PER-M-PANEL-20261030. Packaging only: the owning-device
replay lives in tests/device/rocm/test_quantized_mlp_per_m_panels_device.py.
"""
import json

import pytest

import tessera as ts
from benchmarks.rocm.benchmark_public_quantized_mlp import inputs, make_mlp

TRACED = (256, 2048, 4096, 256)


def _program():
    fn = ts.jit(target="rocm_gfx1201")(make_mlp(*TRACED))
    return fn, fn.compile_native_quantized_program(*inputs(*TRACED), m_bound=256)


def test_m_only_bounded_product_carries_the_static_rules_lds_ranges(rocm_image_toolchain):
    _, program = _program()
    gate, down = program.native.gate, program.native.down
    # Base: the register panel for M >= 1; variants: the LDS panels the static
    # rule takes on 97..128 (128x64) and 129..256 (128x128) at gate N = 4096.
    assert gate.product.descriptor.provenance["macro_tile"] == [16, 32]
    assert [(v.min_m, v.max_m) for v in gate.variants] == [(97, 128), (129, 256)]
    assert [v.native.descriptor.provenance["macro_tile"] for v in gate.variants] == [[128, 64], [128, 128]]
    for v in gate.variants:
        assert v.native.descriptor.provenance["arbitrary_m"] is True
        assert f"tessera.rocm.blockscale_min_m = {v.min_m} : i64" in v.recipe.graph_ir
    # The down product (N = 256) never reaches an LDS panel.
    assert down.variants == ()


def test_runtime_n_product_variants_mask_their_column_edge(rocm_image_toolchain):
    """An H-bounded gate product keeps its global base image and takes the same
    LDS ranges, each projected for an N that varies at launch (whole_n false)."""
    fn = ts.jit(target="rocm_gfx1201")(make_mlp(16, 512, 4096, 256))
    native = fn.compile_native_quantized_program(*inputs(16, 512, 4096, 256), m_bound=256, h_bound=2048).native
    gate = native.gate
    assert gate.product.descriptor.provenance["runtime_shape_image"] is True
    assert [(v.min_m, v.max_m) for v in gate.variants] == [(97, 128), (129, 256)]
    for v in gate.variants:
        provenance = v.native.descriptor.provenance
        assert provenance["arbitrary_n"] is True and provenance["lds_whole_n"] is False
        assert "whole_n = false" in v.native.target_ir
    assert native.down.variants == ()


def test_arbitrary_n_requires_the_lds_family():
    from tessera.compiler.rocm_native import _shape_free_target_ir
    with pytest.raises(ValueError, match="arbitrary_n projection requires the LDS W8A8 family"):
        _shape_free_target_ir("", family="scaled_matmul", directive="tessera_rocm.scaled_wmma_gemm",
                              arbitrary_m=True, arbitrary_n=True)


def test_variant_record_round_trips_and_tampering_is_refused(rocm_image_toolchain):
    import copy
    from tessera.compiler.rocm_quantized_product_program import NativeQuantizedProductProgram
    _, program = _program()
    record = json.loads(json.dumps(program.manifest()["native_program"]))
    assert [(v["min_m"], v["max_m"]) for v in record["gate"]["variants"]] == [(97, 128), (129, 256)]
    replay = NativeQuantizedProductProgram.from_manifest(copy.deepcopy(record))
    assert replay.binding_digest == program.native.binding_digest
    assert [(v.min_m, v.max_m) for v in replay.gate.variants] == [(97, 128), (129, 256)]
    edits = {
        "dropped": (lambda r: r["gate"].pop("variants"), "variant ranges differ|member binding differs"),
        "range": (lambda r: r["gate"]["variants"][0].update(max_m=120), "variant ranges differ|member binding differs"),
        "swapped": (lambda r: r["gate"]["variants"].reverse(), "variant ranges differ|member binding differs"),
        "empty": (lambda r: r["gate"].update(variants=[]), "per-M variants differ"),
        "extra key": (lambda r: r["gate"]["variants"][0].update(tile="x"), "per-M variants differ"),
    }
    for name, (edit, message) in edits.items():
        tampered = copy.deepcopy(record)
        edit(tampered)
        with pytest.raises(ValueError, match=message):
            NativeQuantizedProductProgram.from_manifest(tampered)


def test_rehashed_variant_panel_forgery_is_refused(rocm_image_toolchain):
    """A forger who rehashes still meets the re-derived ranges and panels."""
    import dataclasses
    _, program = _program()
    gate = program.native.gate
    forged = dataclasses.replace(gate, variants=gate.variants[:1])
    forged = dataclasses.replace(forged, binding_digest=forged._identity())
    with pytest.raises(ValueError, match="variant ranges differ"):
        forged.validate()
    swapped = (dataclasses.replace(gate.variants[0], native=gate.variants[1].native,
                                   checked_target_ir=gate.variants[1].checked_target_ir),
               gate.variants[1])
    forged = dataclasses.replace(gate, variants=swapped)
    forged = dataclasses.replace(forged, binding_digest=forged._identity())
    with pytest.raises(ValueError, match="variant Graph differs|variant panel differs|differs"):
        forged.validate()
