"""Bounded-row export of the native seven-member quantized MLP (schema v5).

GFX1201-NVFP4-BOUNDED-MLP-20261027. The compiler retypes the activation's
row-local forward slice to capacity and declares the row frame; the static v4
program is unchanged.
"""
import pytest
from tessera.compiler.rocm_quantized_mlp_program import _frame
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests.unit.test_native_quantized_mlp_program import source
from tests.unit.test_native_quantized_product_program import OPTION, manifest

pytestmark = pytest.mark.compiler_route


def request(text, bound):
    return text.replace("tessera.native.quantized_product}",
                        f"tessera.native.quantized_product, tessera.native.quantized_m_bound = {bound}}}")


def refusal(text):
    with pytest.raises(RuntimeError) as error:
        run_tessera_opt(find_tessera_opt(), text, OPTION)
    return str(error.value)


@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("m,bound", [(3, 8), (1, 1), (5, 64)])
def test_v5_retypes_only_row_slots_to_capacity(m, bound):
    plan = manifest(request(source(m, 32, 48, 19), f"{bound} : i64"))
    static = manifest(source(bound, 32, 48, 19))
    assert plan["schema"] == "tessera.native.quantized_product.v5"
    assert static["schema"] == "tessera.native.quantized_product.v4"
    assert plan["active_m"] == m and plan["m_bound"] == bound
    # At capacity the v5 frame is exactly the static program's frame.
    for key in ("buffers", "steps", "output", "subprograms", "member_graphs"):
        assert plan[key] == static[key]
    assert plan["row_frame"] == {
        "slots": [0, 7, 8, 11, 12, 13, 14, 17],
        "row_strides_bytes": [96, 48, 12, 128, 64, 32, 8, 38],
        "row_steps": [0, 2, 3, 4, 6]}
    assert "tessera.native.quantized_m_bound" in plan["original_graph_ir"]
    assert "tessera.native.quantized_m_bound" not in plan["source_graph_ir"]
    # The original request replays to the identical plan.
    assert manifest(plan["original_graph_ir"]) == plan
    roles, _, _ = _frame(plan)
    assert roles[0] == 0


@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("bound,match", [
    ("2 : i64", "exceed their checked capacity"),
    ("8 : i32", "i64 bound"),
    ('"8"', "i64 bound"),
])
def test_v5_refuses_malformed_bounds(bound, match):
    assert match in refusal(request(source(3, 32, 48, 19), bound))


@pytest.mark.usefixtures("production_compiler")
def test_v5_refuses_an_operation_outside_the_row_local_set():
    # Valid IR the retyper has not certified as row-local: refused before
    # any type changes, never retyped by its leading dimension alone.
    text = request(source(3, 32, 48, 19), "8 : i64")
    text = text.replace("%absolute = tessera.absolute %wide", "%absolute = tessera.relu %wide", 1)
    assert "row-local MLP SSA" in refusal(text)


@pytest.mark.usefixtures("production_compiler")
def test_bound_on_a_smaller_quantized_program_is_refused():
    from tests.unit.test_native_quantized_gate_up_program import source as gate_up
    text = gate_up(3, 32, 48).replace(
        "tessera.native.quantized_product}",
        "tessera.native.quantized_product, tessera.native.quantized_m_bound = 8 : i64}")
    assert "complete seven-member MLP" in refusal(text)


@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("edit", [
    lambda f: f.update(row_steps=[0, 1, 2, 3, 4, 6]),
    lambda f: f.update(slots=f["slots"][:-1], row_strides_bytes=f["row_strides_bytes"][:-1]),
    lambda f: f["row_strides_bytes"].__setitem__(0, 48),
])
def test_python_frame_rederives_the_exported_row_frame(edit):
    plan = manifest(request(source(3, 32, 48, 19), "8 : i64"))
    edit(plan["row_frame"])
    with pytest.raises(ValueError, match="row frame differs"):
        _frame(plan)


@pytest.mark.usefixtures("production_compiler")
def test_python_frame_refuses_active_rows_above_the_bound():
    plan = manifest(request(source(3, 32, 48, 19), "8 : i64"))
    plan["active_m"] = 9
    with pytest.raises(ValueError, match="bounded rows differ"):
        _frame(plan)
