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


def record():
    rows = []
    for target, filename in _BACKEND_MODULES:
        path = ROOT / 'python/tessera/compiler' / filename
        text = path.read_text()
        for function in ast.parse(text).body:
            if not isinstance(function, ast.FunctionDef) or not function.name.startswith('package_'):
                continue
            annotation = _first_param_type(function)
            calls = sorted({ast.unparse(node.func) for node in ast.walk(function) if isinstance(node, ast.Call)})
            rows.append(dict(target=target, source=str(path.relative_to(ROOT)),
                source_sha256=hashlib.sha256(text.encode()).hexdigest(), function=function.name,
                input=annotation, boundary='graph' if _is_bootstrap(annotation) else 'scheduled' if _is_artifact(annotation) else 'raw_or_unclassified',
                scheduled_calls=[c for c in calls if 'scheduled' in c],
                direct_ir_calls=[c for c in calls if 'emit_' in c or 'compile_tile_ir' in c],
                calls=calls))
    return dict(schema=1, scope='AST input and direct call inventory; no transitive semantic or execution proof',
                counts={kind:sum(row['boundary']==kind for row in rows) for kind in ('graph','scheduled','raw_or_unclassified')}, rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(record(), indent=2)+'\n')
