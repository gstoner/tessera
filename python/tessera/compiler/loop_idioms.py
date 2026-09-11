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
from .presburger import PresburgerSystem, PresburgerConstraint, attach_presburger_system


@dataclass(frozen=True)
class LoopIdiomCandidate:
    source_oracle: str
    source_digest: str
    raised_module: GraphIRModule
    promotion_eligible: bool = False
    reason: str = 'rank/prune only; requires differential numerics and native evidence'
    required_system: PresburgerSystem | None = None

    def prepare(self, *, tessera_opt: str, system: PresburgerSystem | None = None) -> ParametricRecipe:
        import copy
        module = copy.deepcopy(self.raised_module)
        if self.required_system is not None:
            systems = (self.required_system,) if system is None else (self.required_system,system)
            symbols = tuple(sorted({name for item in systems for name in item.symbols}))
            rows = []
            for item in systems:
                for row in item.constraints:
                    coefficients = dict(zip(item.symbols,row.coefficients,strict=True))
                    rows.append(PresburgerConstraint(row.relation,tuple(coefficients.get(name,0) for name in symbols),row.constant,row.modulus,row.remainder))
            system = PresburgerSystem(symbols,tuple(rows))
            attach_presburger_system(module.functions[0],system)
        return prepare_recipe(module, tessera_opt=tessera_opt, system=system)


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


def recognize_attention_loop(fn: Callable) -> LoopIdiomCandidate:
    """Raise a bounded spelling of optionally scaled rank-four attention.

    Complete AST matching admits explicit grouped-head indexing and end-aligned
    causal masking, and refuses extra effects or dtype conversions. Recognition is not numerical/promotion proof.
    """
    import numpy as np
    if fn.__code__.co_freevars or fn.__globals__.get('np') is not np or fn.__globals__.get('range',builtins.range) is not builtins.range:
        raise ValueError('attention loop requires uncaptured NumPy and builtin range')
    lines,first_line = inspect.getsourcelines(fn)
    source = textwrap.dedent(''.join(lines))
    functions = [n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)]
    args = ir_args_from_signature(fn)
    if len(functions) != 1 or len(args) not in (3,4) or any(a.ir_type.rank != 4 or a.ir_type.dtype not in ('f32','fp32') for a in args):
        raise ValueError('attention loop requires three rank-four f32 annotations')
    for argument in args:
        argument.ir_type = tensor_ir_type(tuple(int(name) if name.isdigit() else None
                                               for name in argument.dim_names), 'fp32')
    q,k,v = args[:3]
    bias = args[3] if len(args) == 4 else None
    bindings = ['out','scores','weights','batch','head','query','key','feature','value','np','range','max']
    if len(set([a.name for a in args]+bindings)) != len(bindings)+len(args):
        raise ValueError('attention loop bindings must be distinct')
    qs,ks,vs = [a.dim_names for a in args[:3]]
    if any(len(dims) != 4 for dims in (qs,ks,vs)):
        raise ValueError("attention requires four named dimensions per argument")
    if qs[0] != ks[0] or ks[:3] != vs[:3] or qs[3] != ks[3]:
        raise ValueError('attention batch/head/key/contraction dimensions must match')
    if not qs[3].isdigit() or int(qs[3]) <= 0:
        raise ValueError('attention raising currently requires a static head width')
    if bias is not None and tuple(bias.dim_names) != (qs[0],qs[1],qs[2],ks[2]):
        raise ValueError("attention bias requires exact batch/head/query/key dimensions")
    head_index = 'head' if qs[1] == ks[1] else f'head // ({q.name}.shape[1] // {k.name}.shape[1])'
    template = f'''def candidate({q.name}, {k.name}, {v.name}):
    out = np.zeros(({q.name}.shape[0], {q.name}.shape[1], {q.name}.shape[2], {v.name}.shape[3]), dtype={q.name}.dtype)
    for batch in range({q.name}.shape[0]):
        for head in range({q.name}.shape[1]):
            for query in range({q.name}.shape[2]):
                scores = np.zeros(({k.name}.shape[2],), dtype={q.name}.dtype)
                for key in range({k.name}.shape[2]):
                    for feature in range({q.name}.shape[3]):
                        scores[key] += {q.name}[batch, head, query, feature] * {k.name}[batch, {head_index}, key, feature]
                weights = np.exp(scores - np.max(scores))
                weights = weights / np.sum(weights)
                for key in range({k.name}.shape[2]):
                    for value in range({v.name}.shape[3]):
                        out[batch, head, query, value] += weights[key] * {v.name}[batch, {head_index}, key, value]
    return out
'''
    scale = 1.0
    # One literal f32-exact post-dot scale is recognized. The complete AST
    # still owns every other statement, so this cannot swallow extra effects.
    try:
        body = cast(Any,functions[0]).body[1].body[0].body[0].body
        candidate_scale = body[2]
    except (AttributeError,IndexError):
        candidate_scale = None
    if (isinstance(candidate_scale,ast.Assign) and len(candidate_scale.targets)==1
            and isinstance(candidate_scale.targets[0],ast.Name) and candidate_scale.targets[0].id=='scores'
            and isinstance(candidate_scale.value,ast.BinOp) and isinstance(candidate_scale.value.op,ast.Mult)
            and isinstance(candidate_scale.value.left,ast.Name) and candidate_scale.value.left.id=='scores'
            and isinstance(candidate_scale.value.right,ast.Constant)):
        import math
        import struct
        literal = cast(Any,candidate_scale.value.right.value)
        if type(literal) not in (int,float) or not 0 < literal <= 1e6 or not math.isfinite(literal):
            raise ValueError('attention scale must be a finite positive f32 literal')
        scale = float(literal)
        if struct.unpack('f',struct.pack('f',scale))[0] != scale:
            raise ValueError('attention scale literal must be exactly representable as f32')
        template = template.replace('                weights = np.exp',f'                scores = scores * {literal!r}\n                weights = np.exp')
    if bias is not None:
        template = template.replace(f"{v.name}):", f"{v.name}, {bias.name}):", 1)
        template = template.replace("                weights = np.exp", f"                scores = scores + {bias.name}[batch, head, query, :]\n                weights = np.exp")
    causal,window_left,window_right = False,-1,-1
    masks = [n for n in ast.walk(functions[0]) if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='scores' for t in n.targets)]
    if masks:
        if fn.__globals__.get('max',builtins.max) is not builtins.max:
            raise ValueError('attention mask requires builtin max')
        offset = f'query + max({k.name}.shape[2] - {q.name}.shape[2], 0)'
        forms: list[tuple[str,int,str]] = []
        for node in masks:
            candidates = [v.value for v in ast.walk(node) if isinstance(v,ast.Constant) and type(v.value) is int and 0 <= v.value <= 256]
            matched = None
            choices = [('causal',0,f'scores[{offset} + 1:] = -np.inf')]
            for value in candidates:
                choices.extend([('left',value,f'scores[:max({offset} - {value}, 0)] = -np.inf'),
                    ('right',value,f'scores[{offset} + {value} + 1:] = -np.inf')])
            for kind,value,code in choices:
                if ast.dump(node) == ast.dump(ast.parse(code).body[0]):
                    matched = kind,value,code
                    break
            if matched is None or matched[0] in [item[0] for item in forms]:
                raise ValueError('attention mask differs from the complete bounded window contract')
            forms.append(matched)
        if any(kind=='causal' for kind,_,_ in forms) and any(kind=='right' for kind,_,_ in forms):
            raise ValueError('attention requires one right boundary')
        for kind,value,_ in forms:
            if kind=='causal': causal=True
            elif kind=='left': window_left=value
            else: window_right=value
        mask = ''.join('                '+code+'\n' for _,_,code in forms)
        template = template.replace('                weights = np.exp',mask+'                weights = np.exp')
    expected = cast(ast.FunctionDef,ast.parse(template).body[0]).body
    if [ast.dump(n) for n in functions[0].body] != [ast.dump(n) for n in expected]:
        raise ValueError('attention loop differs from the complete stable-softmax recurrence')
    result_type = tensor_ir_type((*q.ir_type.shape[:3],v.ir_type.shape[3]),q.ir_type.dtype)
    digest = hashlib.sha256(source.encode()).hexdigest()
    op = IROp(op_name='tessera.flash_attn',operands=[f'%{a.name}' for a in args],
              operand_types=[str(a.ir_type) for a in args],result='attention',
              result_type=str(result_type),kwargs={'scale':scale,'causal':causal,'head_dim':int(qs[3]),
                      'window_left':window_left,'window_right':window_right,'softcap':0.0,
                      'dropout_p':0.0,'dropout_seed':0},
              attrs='tessera.effect_kind = "pure"',
              source_span=SourceSpan(first_line+functions[0].body[1].lineno-1,1,
                                     source_name=inspect.getsourcefile(fn)))
    graph = GraphIRFunction(fn.__name__,args=args,body=[op],result_types=[result_type],
                            return_values=['%attention'],source_hash=digest)
    required = None
    if window_left >= 0:
        symbols = tuple(sorted({name for argument in args for name in argument.dim_names if not name.isdigit()}))
        coefficients = dict.fromkeys(symbols,0)
        constant = window_left
        for name,sign in ((ks[2],1),(qs[2],-1)):
            if name.isdigit(): constant += sign*int(name)
            else: coefficients[name] += sign
        if symbols:
            required = PresburgerSystem(symbols,(PresburgerConstraint('ge',tuple(coefficients.values()),constant),))
            attach_presburger_system(graph,required)
        elif constant < 0:
            raise ValueError('attention window would produce fully masked rows')
    return LoopIdiomCandidate(source,digest,GraphIRModule(functions=[graph]),required_system=required)
