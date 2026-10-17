"""Explicit typed K16 quantization source and native Schedule admission."""
import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt,run_tessera_opt

pytestmark = pytest.mark.compiler_route

def source(rows=3,columns=32):
    groups=rows*columns//16
    matrix=f"tensor<{rows}x{columns}xbf16>"
    blocks=f"tensor<{groups}x16xbf16>"
    wide=f"tensor<{groups}x16xf32>"
    scale=f"tensor<{groups}x1xf32>"
    codes=f"tensor<{groups}x16xf8E4M3FN>"
    result=f"tensor<{rows}x{columns}xf8E4M3FN>"
    plane=f"tensor<{rows}x{columns//16}xf32>"
    return f'''module attributes {{tessera.target = "rocm_gfx1201", tessera.arch = "gfx1201",
        tessera.native.block_quant}} {{
 func.func @block_quant(%x: {matrix}) -> ({result}, {plane}) {{
  %blocks = tessera.reshape %x {{shape = [{groups}, 16]}} : ({matrix}) -> {blocks}
  %wide = tessera.cast %blocks {{dtype = "fp32"}} : ({blocks}) -> {wide}
  %absolute = tessera.absolute %wide : ({wide}) -> {wide}
  %max = tessera.reduce %absolute {{kind = "max", axis = 1 : i64, keepdims = true}} : ({wide}) -> {scale}
  %divisor = arith.constant dense<448.0> : {scale}
  %raw = tessera.div %max, %divisor : ({scale}, {scale}) -> {scale}
  %floor = arith.constant dense<1.1754943508222875E-38> : {scale}
  %scale = tessera.maximum %raw, %floor : ({scale}, {scale}) -> {scale}
  %expanded = tessera.expand %scale {{shape = [{groups}, 16]}} : ({scale}) -> {wide}
  %normalized = tessera.div %wide, %expanded : ({wide}, {wide}) -> {wide}
  %encoded = tessera.cast %normalized {{dtype = "fp8_e4m3"}} : ({wide}) -> {codes}
  %codes = tessera.reshape %encoded {{shape = [{rows}, {columns}]}} : ({codes}) -> {result}
  %scales = tessera.reshape %scale {{shape = [{rows}, {columns//16}]}} : ({scale}) -> {plane}
  return %codes, %scales : {result}, {plane}
 }}
}}'''


@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("rows,columns", [(3,32),(19,12288),(256,12288)])
def test_typed_quant_graph_preserves_explicit_math_and_storage(rows,columns):
    graph = source(rows,columns)
    verified = run_tessera_opt(find_tessera_opt(),graph,"--verify-each")
    assert verified.count("tessera.cast") == 2
    assert "f8E4M3FN" in verified and "tessera.reduce" in verified
    schedule = run_tessera_opt(find_tessera_opt(),graph,"--tessera-graph-to-schedule")
    assert "schedule.artifact" in schedule and "native_block_quant" in schedule
    assert schedule.count("tessera.reshape") == 3
    assert "tessera.maximum" in schedule and "tessera.expand" in schedule
    assert graph == source(rows,columns)


@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("old,new", [
    ("dense<448.0>","dense<449.0>"),
    ('kind = "max"','kind = "sum"'),
    ('tessera.arch = "gfx1201"','tessera.arch = "gfx1151"'),
])
def test_quant_schedule_never_normalizes_conflicting_semantics(old,new):
    with pytest.raises(RuntimeError):
        run_tessera_opt(find_tessera_opt(),source().replace(old,new),
                        "--tessera-graph-to-schedule")


@pytest.mark.usefixtures("production_compiler")
def test_quant_schedule_accepts_equivalent_negative_last_axis():
    graph = source().replace("axis = 1 : i64","axis = -1 : i64")
    output = run_tessera_opt(find_tessera_opt(),graph,"--tessera-graph-to-schedule")
    assert "schedule.artifact" in output and "native_block_quant" in output


@pytest.mark.usefixtures("production_compiler")
def test_quant_schedule_refuses_a_consistent_sibling_target():
    graph = source().replace("rocm_gfx1201","rocm_gfx1151").replace('"gfx1201"','"gfx1151"')
    run_tessera_opt(find_tessera_opt(),graph,"--verify-each")
    with pytest.raises(RuntimeError,match="native K16 quantization"):
        run_tessera_opt(find_tessera_opt(),graph,"--tessera-graph-to-schedule")
