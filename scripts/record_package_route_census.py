#!/usr/bin/env python3
"""Record package input/call boundaries; never infer device proof from syntax."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from tessera.compiler.bootstrap_prune_audit import _BACKEND_MODULES, _first_param_type, _is_bootstrap, _is_artifact  # noqa: E402


def _resolved_callers(root, symbols):
    """Resolve lexical import names only; never claim dynamic call completeness."""
    result = {symbol: [] for symbol in symbols}
    for path in sorted((root / 'python').rglob('*.py')):
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeError):
            continue
        module = '.'.join(path.relative_to(root / 'python').with_suffix('').parts)
        package = module.split('.')[:-1]
        if path.stem == '__init__':
            module = module.removesuffix('.__init__')
            package = module.split('.')

        def scope_nodes(body):
            for node in body:
                yield node
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                    yield from scope_nodes(ast.iter_child_nodes(node))

        def visit_scope(body, inherited, arguments=(), top=False):
            nodes = list(scope_nodes(body))
            bindings = dict(inherited)
            candidates = {}
            blocked = set(arguments)
            for node in nodes:
                if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                    blocked.add(node.id)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if top:
                        candidates.setdefault(node.name, set()).add(module + '.' + node.name)
                    else:
                        blocked.add(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.asname or alias.name.split('.')[0]
                        candidates.setdefault(name, set()).add(alias.name if alias.asname else name)
                elif isinstance(node, ast.ImportFrom):
                    prefix = '.'.join(package[:len(package) - node.level + 1]) if node.level else ''
                    origin = '.'.join(part for part in (prefix, node.module) if part)
                    for alias in node.names:
                        candidates.setdefault(alias.asname or alias.name, set()).add(origin + '.' + alias.name)
                elif isinstance(node, ast.ExceptHandler) and node.name:
                    blocked.add(node.name)
            for name, values in candidates.items():
                bindings[name] = next(iter(values)) if len(values) == 1 else None
            for name in blocked:
                bindings[name] = None
            for node in nodes:
                if isinstance(node, ast.Call):
                    spelling = ast.unparse(node.func)
                    head, *tail = spelling.split('.')
                    origin = bindings.get(head)
                    resolved = '.'.join([origin, *tail]) if origin else None
                    if resolved in result:
                        result[resolved].append(dict(source=str(path.relative_to(root)), line=node.lineno,
                                                    spelling=spelling, resolution='lexical-candidate'))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = node.args
                    names = [arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
                    names += [arg.arg for arg in (args.vararg, args.kwarg) if arg is not None]
                    visit_scope(node.body, bindings, names)
                elif isinstance(node, ast.ClassDef):
                    # Methods resolve enclosing module names, not class locals.
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            visit_scope([child], bindings)
        visit_scope(tree.body, {}, top=True)
    return result


def _emitter_paths(functions, name, seen=()):
    if name in seen or name not in functions:
        return []
    paths = []
    for node in ast.walk(functions[name]):
        if not isinstance(node, ast.Call):
            continue
        callee = ast.unparse(node.func)
        if callee.split('.')[-1].startswith('emit_'):
            paths.append([name, callee])
        elif callee in functions:
            paths.extend([name, *path] for path in _emitter_paths(functions, callee, (*seen, name)))
    return sorted({tuple(path) for path in paths})


def record():
    rows = []
    for target, filename in (*_BACKEND_MODULES, ("x86", "x86_breadth.py")):
        path = ROOT / 'python/tessera/compiler' / filename
        text = path.read_text()
        functions = {node.name: node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
        for function in functions.values():
            if not isinstance(function, ast.FunctionDef) or not function.name.startswith('package_'):
                continue
            annotation = _first_param_type(function)
            calls = sorted({ast.unparse(node.func) for node in ast.walk(function) if isinstance(node, ast.Call)})
            rows.append(dict(target=target, source=str(path.relative_to(ROOT)),
                source_sha256=hashlib.sha256(text.encode()).hexdigest(), function=function.name,
                input=annotation, boundary='graph' if _is_bootstrap(annotation) else 'scheduled' if _is_artifact(annotation) else 'raw_or_unclassified',
                scheduled_calls=[c for c in calls if 'scheduled' in c],
                direct_ir_calls=[c for c in calls if 'emit_' in c or 'compile_tile_ir' in c],
                calls=calls, local_emitter_paths=_emitter_paths(functions, function.name)))
    symbols = {'tessera.compiler.' + Path(row['source']).stem + '.' + row['function']: row for row in rows}
    for symbol, callers in _resolved_callers(ROOT, symbols).items():
        symbols[symbol]['caller_candidates'] = callers
        symbols[symbol]['certificate_status'] = 'requires per-envelope reconciliation; no certificate inferred from calls'
    return dict(schema=2, scope='Lexical caller candidates and local emitter paths; conservative lexical binding; indirect dispatch and external callers require review; no execution proof',

                counts={kind:sum(row['boundary']==kind for row in rows) for kind in ('graph','scheduled','raw_or_unclassified')}, rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(record(), indent=2)+'\n')
