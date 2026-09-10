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
        bindings = {}
        # Import candidates include function-local imports. Lexical shadowing
        # still requires manual review, as declared in the report scope.
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    bindings[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
            elif isinstance(node, ast.ImportFrom):
                prefix = '.'.join(package[:len(package) - node.level + 1]) if node.level else ''
                origin = '.'.join(part for part in (prefix, node.module) if part)
                for alias in node.names:
                    bindings[alias.asname or alias.name] = origin + '.' + alias.name
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            spelling = ast.unparse(node.func)
            head, *tail = spelling.split('.')
            resolved = '.'.join([bindings.get(head, module + '.' + head), *tail])
            if resolved in result:
                result[resolved].append(dict(source=str(path.relative_to(root)), line=node.lineno,
                                            spelling=spelling, resolution='module-import-or-local-name'))
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
    return dict(schema=2, scope='Lexical caller candidates and local emitter paths; shadowing, indirect dispatch and external callers require review; no execution proof',

                counts={kind:sum(row['boundary']==kind for row in rows) for kind in ('graph','scheduled','raw_or_unclassified')}, rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(record(), indent=2)+'\n')
