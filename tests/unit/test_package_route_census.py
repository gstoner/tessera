from scripts.record_package_route_census import _emitter_paths, _resolved_callers
import ast


def test_census_follows_local_helpers_without_cycles():
    tree = ast.parse('''
def package_math(): return helper()
def helper():
    package_math()
    return emit_tile_ir()
''')
    functions = {node.name: node for node in tree.body}
    assert _emitter_paths(functions, 'package_math') == [('package_math', 'helper', 'emit_tile_ir')]


def test_census_resolves_aliases_and_keeps_other_targets_distinct(tmp_path):
    root = tmp_path / 'python' / 'tessera'
    root.mkdir(parents=True)
    (root / 'caller.py').write_text('''
from tessera.compiler.x86_native import package_matmul as pack
from tessera.compiler import rocm_native as amd
pack(graph)
amd.package_matmul(graph)
''')
    symbol = 'tessera.compiler.x86_native.package_matmul'
    callers = _resolved_callers(tmp_path, [symbol])[symbol]
    assert len(callers) == 1 and callers[0]['spelling'] == 'pack'
