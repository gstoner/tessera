"""Conservative source-loop raising candidates for FRONTEND-IR-MEDIUM-1.

This compatibility analysis recognizes one exact NumPy loop spelling. It does
not replace the tracer or execute source. The resulting existing tessera.matmul
operation enters the opt-in parametric rank/prune tier, never native selection.
"""
from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
import hashlib
import inspect
import textwrap
from typing import Any, Callable, cast

from .graph_ir import (GraphIRFunction, GraphIRModule, IROp, SourceSpan,
                       ir_args_from_signature, tensor_ir_type)
from .parametric_recipe import ParametricRecipe, prepare_recipe
from .presburger import PresburgerSystem


@dataclass(frozen=True)
class LoopIdiomCandidate:
    source_oracle: str
    source_digest: str
    raised_module: GraphIRModule
    promotion_eligible: bool = False
    reason: str = 'rank/prune only; requires differential numerics and native evidence'

    def prepare(self, *, tessera_opt: str, system: PresburgerSystem | None = None) -> ParametricRecipe:
        return prepare_recipe(self.raised_module, tessera_opt=tessera_opt, system=system)


def recognize_matmul_loop(fn: Callable) -> LoopIdiomCandidate:
    """Recognize fresh zero output and a complete i/j/k += product nest.

    Exact syntax, rank-2 equal floating dtypes and equal contraction dimension
    names are required. Aliasing output, nonzero initialization, loop steps,
    side effects, transposes, masks, and epilogues are refused. Floating-point
    reassociation remains a promotion gate even for recognized candidates.
    """
    import numpy as np

    if fn.__code__.co_freevars:
        raise ValueError("loop idiom does not support captured bindings")
    source_lines, first_line = inspect.getsourcelines(fn)
    source = textwrap.dedent(''.join(source_lines))
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(functions) != 1 or len(functions[0].body) != 3:
        raise ValueError('unsupported loop idiom: expected zero allocation, loop nest, return')
    function = functions[0]
    args = ir_args_from_signature(fn)
    if len(args) != 2 or any(a.ir_type.rank != 2 for a in args):
        raise ValueError('matmul loop requires two rank-2 annotated arguments')
    a, b = args
    if a.ir_type.dtype != b.ir_type.dtype or a.ir_type.dtype not in {'fp32', 'fp64', 'f32', 'f64'}:
        raise ValueError('matmul loop requires matching f32/f64 element types')
    if len(a.dim_names) != 2 or len(b.dim_names) != 2 or a.dim_names[1] != b.dim_names[0]:
        raise ValueError('matmul loop contraction dimension names must match')
    try:
        allocation, outer, returned = (cast(Any, n) for n in function.body)
        output = allocation.targets[0].id
        numpy_name = allocation.value.func.value.id
        i = outer.target.id
        j = outer.body[0].target.id
        k = outer.body[0].body[0].target.id
    except (AttributeError, IndexError) as exc:
        raise ValueError('unsupported loop idiom structure') from exc
    if len({a.name, b.name, output, i, j, k, numpy_name, 'range'}) != 8:
        raise ValueError('loop bindings must be distinct')
    if fn.__globals__.get(numpy_name) is not np or fn.__globals__.get('range', builtins.range) is not builtins.range:
        raise ValueError('loop idiom requires NumPy allocation and builtin range')
    template = f'''def candidate({a.name}, {b.name}):
    {output} = {numpy_name}.zeros(({a.name}.shape[0], {b.name}.shape[1]), dtype={a.name}.dtype)
    for {i} in range({a.name}.shape[0]):
        for {j} in range({b.name}.shape[1]):
            for {k} in range({a.name}.shape[1]):
                {output}[{i}, {j}] += {a.name}[{i}, {k}] * {b.name}[{k}, {j}]
    return {output}
'''
    expected = cast(ast.FunctionDef, ast.parse(template).body[0]).body
    if [ast.dump(n) for n in function.body] != [ast.dump(n) for n in expected]:
        raise ValueError('unsupported loop idiom: loop body differs from the complete matrix product')
    result_type = tensor_ir_type((a.ir_type.shape[0], b.ir_type.shape[1]), a.ir_type.dtype)
    source_digest = hashlib.sha256(source.encode()).hexdigest()
    op = IROp(op_name='tessera.matmul', operands=[f'%{a.name}', f'%{b.name}'],
              operand_types=[str(a.ir_type), str(b.ir_type)], result='product',
              result_type=str(result_type), source_span=SourceSpan(
                  first_line + outer.lineno - 1, outer.col_offset + 1,
                  source_name=inspect.getsourcefile(fn)))
    graph_fn = GraphIRFunction(fn.__name__, args=args, body=[op],
                              result_types=[result_type], return_values=['%product'],
                              source_hash=source_digest)
    return LoopIdiomCandidate(source, source_digest, GraphIRModule(functions=[graph_fn]))
