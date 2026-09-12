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


def test_census_does_not_leak_imports_or_resolve_shadowed_names(tmp_path):
    root = tmp_path / 'python'
    root.mkdir()
    (root / 'caller.py').write_text('''
from tessera.compiler.x86_native import package_matmul as pack
def shadow(pack):
    pack(graph)
def rebound():
    pack(graph)
    pack = other
def local():
    from tessera.compiler.rocm_native import package_matmul as pack
    pack(graph)
def sibling():
    pack(graph)
''')
    x86 = 'tessera.compiler.x86_native.package_matmul'
    rocm = 'tessera.compiler.rocm_native.package_matmul'
    rows = _resolved_callers(tmp_path, [x86, rocm])
    assert [row['line'] for row in rows[x86]] == [12]
    assert [row['line'] for row in rows[rocm]] == [10]


def test_census_package_init_relative_import(tmp_path):
    root = tmp_path / 'python/tessera/compiler'
    root.mkdir(parents=True)
    (root / '__init__.py').write_text('from .x86_native import package_matmul\npackage_matmul(graph)\n')
    symbol = 'tessera.compiler.x86_native.package_matmul'
    assert len(_resolved_callers(tmp_path, [symbol])[symbol]) == 1
