"""Original Graph/native replay tests; execution belongs to exact ROCm hosts."""
import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

def source(kind="sqrt", arch="gfx1151", shape="3x17", reverse=False):
    binary = kind in {"add", "div"}
    args = "%a: tensor<"+shape+"xf32>"
    if binary:
        args += ", %b: tensor<"+shape+"xf32>"
    operands = "%b, %a" if binary and reverse else "%a, %b" if binary else "%a"
    types = ", ".join(["tensor<"+shape+"xf32>"] * (2 if binary else 1))
    attrs = " {axis = -1 : i64}" if kind in {"cumsum", "cummax"} else ""
    bindings = '["a", "b", "out"]' if binary else '["a", "out"]'
    return (
        'module attributes {tessera.target = "rocm", tessera.arch = "'+arch+
        '", tessera.launch_bindings = '+bindings+'} {\n'
        ' func.func @math('+args+') -> tensor<'+shape+'xf32> {\n'
        '  %o = "tessera.'+kind+'"('+operands+')'+attrs+' : ('+types+
        ') -> tensor<'+shape+'xf32>\n'
        '  return %o : tensor<'+shape+'xf32>\n }\n}'
    ).replace("\\n", "\n")

@pytest.fixture
def tool():
    t = find_tessera_opt()
    if t is None:
        pytest.skip("requires production tessera-opt")
    return t

@pytest.mark.parametrize("arch", ["gfx1151", "gfx1201"])
@pytest.mark.parametrize("kind", ["sqrt", "exp", "add", "div", "cumsum", "cummax"])
@pytest.mark.parametrize("shape", ["3x17", "2x3x257"])
def test_original_graph_replays_math_into_tile(tool, arch, kind, shape):
    graph = source(kind, arch, shape)
    schedule = run_tessera_opt(tool, graph, "--tessera-graph-to-schedule")
    assert "tessera."+kind+" " in schedule
    assert 'family=rocm_math' in schedule
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    assert "tile.scan_kernel" in tile if kind in {"cumsum", "cummax"} else "tile.elementwise_kernel" in tile
    assert "tessera.rocm_math_contract" in tile
    assert "tessera."+kind+" " not in tile
    assert tile == run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")

@pytest.mark.parametrize("kind", ["add", "div"])
def test_binary_roles_preserve_authored_argument_order(tool, kind):
    schedule = run_tessera_opt(tool, source(kind, reverse=True), "--tessera-graph-to-schedule")
    assert "roles = [1, 0]" in schedule
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    assert "tile.elementwise_kernel %arg1, %arg0, %arg2, %arg3" in tile

@pytest.mark.parametrize("edit", [
    lambda s: s.replace("elements = 51", "elements = 52"),
    lambda s: s.replace('numeric_policy = "f32_compute"', 'numeric_policy = "approximate"'),
    lambda s: s.replace('roles = [0]', 'roles = [1]'),
    lambda s: s.replace('"a", "out"', '"a", "a"'),
    lambda s: s.replace("tessera.sqrt ", "tessera.exp "),
])
def test_schedule_mutation_cannot_change_math_contract(tool, edit):
    schedule = run_tessera_opt(tool, source(), "--tessera-graph-to-schedule")
    changed = edit(schedule)
    assert changed != schedule
    with pytest.raises(RuntimeError, match="ROCm math"):
        run_tessera_opt(tool, changed, "--tessera-schedule-to-tile")

@pytest.mark.parametrize("edit", [
    lambda s: s.replace("3x17xf32", "3x17xf16"),
    lambda s: s.replace("3x17xf32", "?x17xf32"),
    lambda s: s.replace('"a", "out"', '"a", "a"'),
    lambda s: s.replace('"tessera.sqrt"(%a)', '"tessera.sqrt"(%a) {numeric_policy = "approximate"}'),
])
def test_admission_rejects_unproved_math_envelopes(tool, edit):
    with pytest.raises(RuntimeError, match="ROCm math"):
        run_tessera_opt(tool, edit(source()), "--tessera-graph-to-schedule")

def test_scan_requires_last_axis(tool):
    with pytest.raises(RuntimeError, match="axis policy"):
        run_tessera_opt(tool, source("cumsum").replace("axis = -1", "axis = 0"), "--tessera-graph-to-schedule")


@pytest.fixture
def rocm_tool(tool):
    import subprocess
    if "lower-tile-to-rocm" not in subprocess.check_output([str(tool), "--help"], text=True):
        pytest.skip("requires built ROCm Target IR backend")
    return tool

@pytest.mark.parametrize("arch", ["gfx1151", "gfx1201"])
@pytest.mark.parametrize("kind", ["sqrt", "exp", "add", "div", "cumsum", "cummax"])
def test_native_math_target_consumes_exact_tile(rocm_tool, arch, kind):
    schedule = run_tessera_opt(rocm_tool, source(kind, arch), "--tessera-graph-to-schedule")
    tile = run_tessera_opt(rocm_tool, schedule, "--tessera-schedule-to-tile")
    target = run_tessera_opt(rocm_tool, tile, "--lower-tile-to-rocm=arch="+arch)
    family = "scan" if kind in {"cumsum", "cummax"} else "binary" if kind in {"add", "div"} else "unary"
    assert "tessera_rocm."+family in target
    assert "native_math_contract" in target
    assert "tile." not in target
    assert "llvm.func" not in target

@pytest.mark.parametrize("edit", [
    lambda s: s.replace('kind = "sqrt"', 'kind = "exp"', 1),
    lambda s: s.replace("tile.elementwise_kernel %arg0, %arg1", "tile.elementwise_kernel %arg1, %arg0"),
    lambda s: s.replace('family = "unary", kind = "sqrt", output_storage', 'family = "transcendental", kind = "sqrt", output_storage'),
])
def test_target_rejects_tile_mutation(rocm_tool, edit):
    schedule = run_tessera_opt(rocm_tool, source(), "--tessera-graph-to-schedule")
    tile = run_tessera_opt(rocm_tool, schedule, "--tessera-schedule-to-tile")
    changed = edit(tile)
    assert changed != tile
    with pytest.raises(RuntimeError, match="ROCm math|unsupported transcendental kind"):
        run_tessera_opt(rocm_tool, changed, "--lower-tile-to-rocm=arch=gfx1151")

def test_target_rejects_unowned_numerical_attribute(rocm_tool):
    schedule = run_tessera_opt(rocm_tool, source(), "--tessera-graph-to-schedule")
    tile = run_tessera_opt(rocm_tool, schedule, "--tessera-schedule-to-tile")
    changed = tile.replace('{family = "unary", kind = "sqrt", output_storage',
                           '{numeric_policy = "approximate", family = "unary", kind = "sqrt", output_storage')
    assert changed != tile
    with pytest.raises(RuntimeError, match="ROCm math"):
        run_tessera_opt(rocm_tool, changed, "--lower-tile-to-rocm=arch=gfx1151")


def test_target_rejects_sibling_architecture(rocm_tool):
    schedule = run_tessera_opt(rocm_tool, source(), "--tessera-graph-to-schedule")
    tile = run_tessera_opt(rocm_tool, schedule, "--tessera-schedule-to-tile")
    with pytest.raises(RuntimeError, match="owning architecture"):
        run_tessera_opt(rocm_tool, tile, "--lower-tile-to-rocm=arch=gfx1201")
