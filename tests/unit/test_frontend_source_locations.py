import inspect

import tessera
from tessera.compiler.graph_ir import GraphIRBuilder


def test_ast_location_keeps_file_line_and_nested_indent():
    def kernel(x: tessera.Tensor['M', 'K', 'f32'], y: tessera.Tensor['K', 'N', 'f32']):
        return tessera.matmul(x, y)

    builder = GraphIRBuilder()
    fn = builder.lower(kernel, prefer_abstract_trace=False)
    op = fn.body[0]
    lines, start = inspect.getsourcelines(kernel)
    offset = next(i for i, line in enumerate(lines) if 'return tessera' in line)
    assert op.source_span.source_name == __file__
    assert op.source_span.line == start + offset
    assert op.source_span.col == lines[offset].index('tessera.matmul') + 1
    assert 'loc("tests/unit/test_frontend_source_locations.py"' in fn.to_mlir(canonical=True)
    assert 'loc(' not in fn.to_mlir()


def test_explicit_source_has_virtual_document_location():
    def kernel(x, y):
        pass
    builder = GraphIRBuilder()
    fn = builder.lower(kernel, source_text='def kernel(x, y):\n    return tessera.matmul(x, y)\n',
                       prefer_abstract_trace=False)
    assert fn.body[0].source_span.source_name == '<tessera-source:kernel>'
    assert fn.body[0].source_span.line == 2


def test_source_cache_does_not_cross_call_sites():
    from tessera.compiler import graph_ir_cache as cache
    from tessera.compiler.graph_ir import GraphIRModule
    cache.store('same source', GraphIRModule(), source_location='a.py:1')
    assert cache.lookup('same source', source_location='a.py:1') is not None
    assert cache.lookup('same source', source_location='b.py:9') is None


def test_jit_file_source_locations_survive_cache_and_lazy_recovery(tmp_path):
    from tessera.compiler.graph_ir import GraphIRModule

    def kernel(x: tessera.Tensor['M', 'K', 'f32'], y: tessera.Tensor['K', 'N', 'f32']):
        pass

    source = '\n\ndef kernel(x, y):\n    return tessera.matmul(x, y)\n'
    for name in ('first.py', 'second.py'):
        path = tmp_path / name
        path.write_text(source)
        compiled = tessera.jit(source_path=str(path))(kernel)
        for recover in (False, True):
            if recover:
                compiled._legacy_graph_ir = GraphIRModule()
            module = compiled._ensure_legacy_graph_ir()
            function = module.functions[0]
            span = function.body[0].source_span
            assert span.source_name == str(path)
            assert span.line == 4
            assert span.col == 12
            assert str(path) in function.to_mlir(canonical=True)
