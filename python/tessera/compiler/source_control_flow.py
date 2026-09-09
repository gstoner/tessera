"""Opt-in Python source control recovery through the existing typed tracer.

This producer owns no IR emitter: branch merges and bounded loops are recorded
by TraceBuilder. Bounded returns, handled exceptions and declared tensor state
use typed SSA. Explicit error-result specs transport builtin exception classes;
arbitrary object effects and dynamic exception objects remain excluded.
"""
import ast
import inspect
import textwrap
from dataclasses import dataclass


class SourceControlFlowError(ValueError):
    pass


@dataclass(frozen=True)
class _StateRef:
    root: str


def recover_callable(fn, *, max_steps=None, state_groups=(),error_specs=(),object_fields=()):
    from tessera import ops
    from .trace import Tracer
    from ._trace_hook import active_tracer
    source, first_line = inspect.getsourcelines(fn)
    module = ast.parse(textwrap.dedent(''.join(source)), filename=inspect.getsourcefile(fn) or '<source>')
    definitions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    if len(definitions) != 1 or definitions[0].name != fn.__name__:
        raise SourceControlFlowError('source capture requires one ordinary function')
    definition = definitions[0]
    formals=[*definition.args.posonlyargs,*definition.args.args]
    if not isinstance(object_fields,tuple) or any(not isinstance(row,tuple) or len(row)!=2 or type(row[0]) is not int or not 0<=row[0]<len(formals) or not isinstance(row[1],tuple) or not row[1] or any(type(f) is not str or not f.isidentifier() or f.startswith('_') for f in row[1]) or len(set(row[1]))!=len(row[1]) for row in object_fields) or len({row[0] for row in object_fields})!=len(object_fields):
        raise SourceControlFlowError('source fields require unique parameter indices and public field names')
    if object_fields:
        field_map={}
        expanded=[]
        specs=dict(object_fields)
        used_names={n.id for n in ast.walk(definition) if isinstance(n,ast.Name)}
        for index,arg in enumerate(formals):
            if index not in specs:
                expanded.append(arg)
                continue
            for field in specs[index]:
                name=f'__tessera_field_{index}_{field}'
                if name in used_names:raise SourceControlFlowError('source field name collides with a local')
                field_map[(arg.arg,field)]=name
                expanded.append(ast.copy_location(ast.arg(arg=name),arg))
        class Fields(ast.NodeTransformer):
            def visit_Attribute(self,node):
                if isinstance(node.value,ast.Name) and node.value.id in {a for a,_ in field_map}:
                    key=(node.value.id,node.attr)
                    if key not in field_map or not isinstance(node.ctx,ast.Load):
                        raise SourceControlFlowError('undeclared field or object field rebinding is unsupported')
                    return ast.copy_location(ast.Name(id=field_map[key],ctx=ast.Load()),node)
                return self.generic_visit(node)
            def visit_Subscript(self,node):
                if isinstance(node.value,ast.Name) and node.value.id in {a for a,_ in field_map}:
                    if not isinstance(node.slice,ast.Constant) or (node.value.id,node.slice.value) not in field_map or not isinstance(node.ctx,ast.Load):
                        raise SourceControlFlowError('object access requires a declared literal field')
                    return ast.copy_location(ast.Name(id=field_map[(node.value.id,node.slice.value)],ctx=ast.Load()),node)
                return self.generic_visit(node)
        definition=Fields().visit(definition)
        definition.args.posonlyargs=[]
        definition.args.args=expanded
    if definition.args.vararg or definition.args.kwarg or definition.args.kwonlyargs:
        raise SourceControlFlowError('source capture requires positional arguments')
    if max_steps is not None and (type(max_steps) is not int or not 1 <= max_steps <= 1024):
        raise SourceControlFlowError('source loop bound must be an integer from 1 to 1024')
    allowed = {name: getattr(ops, name) for name in (
        'add', 'sub', 'mul', 'div', 'neg', 'exp', 'sin', 'cos', 'tanh', 'relu',
        'lt', 'le', 'gt', 'ge', 'eq', 'ne',
    ) if hasattr(ops, name)}
    allowed_ids = {id(value) for value in allowed.values()}
    binary = {ast.Add: 'add', ast.Sub: 'sub', ast.Mult: 'mul', ast.Div: 'div'}
    comparison = {ast.Lt: 'lt', ast.LtE: 'le', ast.Gt: 'gt',
                  ast.GtE: 'ge', ast.Eq: 'eq', ast.NotEq: 'ne'}
    closure = inspect.getclosurevars(fn)
    outer = {**closure.globals, **closure.nonlocals}
    # Validate the whole source before tracing either path, including dead code.
    permitted = (ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign,
                 ast.If, ast.While, ast.Break, ast.Continue, ast.Assert, ast.Try, ast.Raise, ast.ExceptHandler, ast.Expr, ast.Name, ast.Load, ast.Store,
                 ast.Call, ast.keyword, ast.Attribute, ast.Constant, ast.Tuple, ast.Subscript, ast.Slice,
                 ast.BinOp, ast.Compare, ast.UnaryOp, ast.USub,
                 *binary, *comparison)
    import builtins
    exception_types={name:getattr(builtins,name) for name in ('Exception','ValueError','RuntimeError','AssertionError','TypeError','IndexError','KeyError','OverflowError','ZeroDivisionError')}
    raised_calls={id(node.exc) for node in ast.walk(definition) if isinstance(node,ast.Raise)}
    local_names={arg.arg for arg in (*definition.args.posonlyargs,*definition.args.args)}
    local_names.update(part.id for node in ast.walk(definition) if isinstance(node,ast.Assign) for part in node.targets if isinstance(part,ast.Name))
    def exception_name(node):
        if (not isinstance(node,ast.Name) or node.id not in exception_types or node.id in local_names
                or outer.get(node.id,exception_types[node.id]) is not exception_types[node.id]):
            raise SourceControlFlowError('exception edges require an unshadowed supported builtin class')
        return node.id
    for node in ast.walk(definition):
        if node in definition.decorator_list:
            continue
        if not isinstance(node, permitted):
            raise SourceControlFlowError(f'source capture excludes {type(node).__name__} at line {first_line + node.lineno - 1 if hasattr(node, "lineno") else first_line}')
        if isinstance(node,ast.Raise):
            if node.exc is None and node.cause is None:
                continue
            if node.cause is not None or not isinstance(node.exc,ast.Call) or node.exc.keywords or len(node.exc.args)>1 or any(not isinstance(a,ast.Constant) or not isinstance(a.value,str) for a in node.exc.args):
                raise SourceControlFlowError('raise requires a builtin exception with a literal message')
            exception_name(node.exc.func)
        if isinstance(node,ast.ExceptHandler):
            if node.name is not None:
                raise SourceControlFlowError('exception objects cannot escape into tensor state')
            if node.type is not None:
                for kind in node.type.elts if isinstance(node.type,ast.Tuple) else (node.type,):exception_name(kind)
        if isinstance(node, ast.Call) and id(node) not in raised_calls:
            target = None
            if isinstance(node.func, ast.Name):
                target = outer.get(node.func.id)
            elif (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                  and outer.get(node.func.value.id) is ops):
                target = allowed.get(node.func.attr)
            if id(target) not in allowed_ids or any(k.arg is None for k in node.keywords):
                raise SourceControlFlowError('source capture excludes an unmodelled call effect')
        if isinstance(node, ast.Assert) and node.msg is not None and not (isinstance(node.msg, ast.Constant) and isinstance(node.msg.value, str)):
            raise SourceControlFlowError('source assertion messages must be literal strings')
        if isinstance(node, ast.While) and (node.orelse or max_steps is None):
            raise SourceControlFlowError('source while requires max_steps and no loop else')
        if isinstance(node, ast.Assign):
            if len(node.targets)!=1:
                raise SourceControlFlowError('source capture requires one assignment target')
            target=node.targets[0]
            if isinstance(target,ast.Subscript):
                if (not state_groups or not isinstance(target.value,ast.Name) or not isinstance(target.slice,ast.Slice)
                        or any(v is not None for v in (target.slice.lower,target.slice.upper,target.slice.step))):
                    raise SourceControlFlowError('external mutation requires a declared full-tensor state slot')
            elif not isinstance(target,ast.Name):
                raise SourceControlFlowError('source capture only assigns local names or declared state slots')
        if isinstance(node, ast.Expr) and not isinstance(node.value, ast.Constant):
            raise SourceControlFlowError('source capture excludes effect-only expressions')
    def literal(node):
        assert isinstance(node,ast.Constant)
        return node.value
    edges=[]
    for node in ast.walk(definition):
        if isinstance(node,ast.Raise) and isinstance(node.exc,ast.Call):
            value=(exception_name(node.exc.func),tuple(literal(arg) for arg in node.exc.args))
        elif isinstance(node,ast.Assert):
            value=('AssertionError',(literal(node.msg),) if node.msg is not None else ())
        else:continue
        if value not in edges:edges.append(value)
    if len(edges)>32:raise SourceControlFlowError('source exception table exceeds 32 static edges')
    budget = [0]

    def expression(node, env):
        from .trace import _source_control_span
        from .graph_ir import SourceSpan
        token = _source_control_span.set(SourceSpan(
            line=first_line + node.lineno - 1, col=node.col_offset + 1,
            source_name=inspect.getsourcefile(fn)))
        try:
            return evaluate(node, env)
        finally:
            _source_control_span.reset(token)

    def evaluate(node, env):
        if isinstance(node, ast.Name):
            if node.id in env:
                value=env[node.id]
                return env[value.root] if isinstance(value,_StateRef) else value
            if node.id in outer:
                return outer[node.id]
            raise SourceControlFlowError(f'unknown source value {node.id}')
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float, bool, str, type(None)):
                raise SourceControlFlowError('unsupported source constant')
            return node.value
        if isinstance(node, ast.Tuple):
            return tuple(expression(value, env) for value in node.elts)
        if isinstance(node, ast.Attribute):
            parent = expression(node.value, env)
            if parent is ops and node.attr in allowed:
                return allowed[node.attr]
            raise SourceControlFlowError('source capture excludes object attribute access')
        if isinstance(node, ast.Call):
            target = expression(node.func, env)
            if id(target) not in allowed_ids or any(keyword.arg is None for keyword in node.keywords):
                raise SourceControlFlowError('source capture call is not an admitted pure tensor operation')
            return target(*(expression(value, env) for value in node.args),
                          **{keyword.arg: expression(keyword.value, env) for keyword in node.keywords if keyword.arg is not None})
        if isinstance(node, ast.BinOp):
            return allowed[binary[type(node.op)]](expression(node.left, env), expression(node.right, env))
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            return allowed[comparison[type(node.ops[0])]](expression(node.left, env), expression(node.comparators[0], env))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return allowed['sub'](0, expression(node.operand, env))
        raise SourceControlFlowError('unsupported source expression')

    def condition(node, env):
        value = expression(node, env)
        if isinstance(value, Tracer):
            builder = active_tracer()
            assert builder is not None
            producers = builder.body
            if not any(value.ssa in op.result_names and op.op_name in
                       {'tessera.lt','tessera.le','tessera.gt','tessera.ge','tessera.eq','tessera.ne'}
                       for op in producers):
                raise SourceControlFlowError('source tensor predicates require an explicit comparison')
        elif type(value) is not bool:
            raise SourceControlFlowError('source predicate must be boolean')
        return value

    def guard(value, message, *, stopped=False):
        # Registered MLIR operations carry the observable assertion effect.
        # No Python callback is executed, and the condition remains SSA-owned.
        from .graph_ir import IROp
        builder = active_tracer()
        assert builder is not None
        if not isinstance(value, Tracer) or value.shape != (1,) or value.dtype not in ('f32', 'fp32'):
            if type(value) is bool:
                if value == stopped:
                    raise SourceControlFlowError(message)
                return
            raise SourceControlFlowError('source assertions require shape-(1,) fp32 comparison masks')
        def emit(name, operands, types, result_type=None, attrs=None):
            result = builder._fresh() if result_type else None
            builder.body.append(IROp(result=result, op_name=name, operands=operands,
                operand_types=types, result_type=result_type, attrs=attrs))
            return '%'+result if result else None
        index = emit('arith.constant', [], [], 'index', 'value = 0 : index')
        scalar = emit('tensor.extract', ['%'+value.ssa, index], ['tensor<1xf32>', 'index'], 'f32')
        zero = emit('arith.constant', [], [], 'f32', 'value = 0.0 : f32')
        ok = emit('arith.cmpf', [scalar, zero], ['f32', 'f32'], 'i1', f'predicate = {5 if stopped else 2} : i64')
        import json
        emit('cf.assert', [ok], ['i1'], attrs='msg = '+json.dumps(message))

    def statements(nodes, env, *, loop_result=None, loop_next=None, loop_break=None, return_to=None, on_done=None, raise_to=None):
        budget[0] += 1
        if budget[0] > 256:
            raise SourceControlFlowError('source continuation expansion exceeds 256 regions')
        if not nodes:
            if on_done is not None:
                return on_done(env)
            if loop_next is not None:
                return loop_next(env)
            if loop_result is not None:
                return env[loop_result]
            raise SourceControlFlowError('source path has no tensor return')
        node, tail = nodes[0], nodes[1:]
        if isinstance(node, ast.Break):
            if loop_break is None: raise SourceControlFlowError("break requires an expanded source loop")
            return loop_break(env)
        if isinstance(node, ast.Continue):
            if loop_next is None: raise SourceControlFlowError("continue requires an expanded source loop")
            return loop_next(env)
        if isinstance(node, ast.Assert):
            if node.msg is not None and not (isinstance(node.msg, ast.Constant) and isinstance(node.msg.value, str)):
                raise SourceControlFlowError("source assertion messages must be literal strings")
            predicate=condition(node.test,env)
            if raise_to is not None:
                def success():
                    return statements(tail,env,loop_result=loop_result,loop_next=loop_next,loop_break=loop_break,return_to=return_to,on_done=on_done,raise_to=raise_to)
                if isinstance(predicate,Tracer):
                    builder=active_tracer()
                    assert builder is not None
                    return builder.record_cond(predicate,success,lambda: raise_to(env,('AssertionError',(literal(node.msg),) if node.msg is not None else ())),())
                return success() if predicate else raise_to(env,('AssertionError',(literal(node.msg),) if node.msg is not None else ()))
            guard(predicate, node.msg.value if node.msg else "source assertion failed")
            return statements(tail, env, loop_result=loop_result, loop_next=loop_next, loop_break=loop_break, return_to=return_to, on_done=on_done, raise_to=raise_to)
        if isinstance(node,ast.Raise):
            if raise_to is None:
                raise SourceControlFlowError('uncaught source exception requires an exported error ABI')
            if node.exc is None:
                if '@exception' not in env:raise SourceControlFlowError('bare raise requires a statically active handler')
                return raise_to(env,env['@exception'])
            assert isinstance(node.exc,ast.Call)
            return raise_to(env,(exception_name(node.exc.func),tuple(literal(arg) for arg in node.exc.args)))
        if isinstance(node,ast.Try):
            def returned(state,value):
                return return_to(state,value) if return_to is not None else value
            def raised(state,kind):
                if raise_to is None:
                    raise SourceControlFlowError('uncaught source exception requires an exported error ABI')
                return raise_to(state,kind)
            def after(state):
                return statements(tail,state,loop_result=loop_result,loop_next=loop_next,loop_break=loop_break,return_to=return_to,on_done=on_done,raise_to=raise_to)
            def cleanup(continuation, *, pending_exception=False):
                def run(state,*payload):
                    def restore(callback):
                        if callback is None:return None
                        def leave(updated,*args):
                            updated=dict(updated)
                            updated.pop('@exception',None)
                            if '@exception' in state:updated['@exception']=state['@exception']
                            return callback(updated,*args)
                        return leave
                    active={**state,'@exception':payload[0]} if pending_exception else state
                    return statements(node.finalbody,active,on_done=restore(lambda updated: continuation(updated,*payload)),
                        loop_next=restore(loop_next),loop_break=restore(loop_break),return_to=restore(return_to),raise_to=restore(raise_to))
                return run
            cleanup_return=cleanup(returned)
            cleanup_raise=cleanup(raised,pending_exception=True)
            cleanup_break=cleanup(loop_break) if loop_break is not None else None
            cleanup_continue=cleanup(loop_next) if loop_next is not None else None
            finish=cleanup(after)
            def handled(state,kind):
                for handler in node.handlers:
                    matches=handler.type is None or any(issubclass(exception_types[kind[0]],exception_types[exception_name(t)]) for t in (handler.type.elts if isinstance(handler.type,ast.Tuple) else (handler.type,)))
                    if matches:
                        def restore(continuation):
                            if continuation is None:return None
                            def run(updated,*payload):
                                updated=dict(updated)
                                updated.pop('@exception',None)
                                if '@exception' in state:updated['@exception']=state['@exception']
                                return continuation(updated,*payload)
                            return run
                        return statements(handler.body,{**state,'@exception':kind},on_done=restore(finish),loop_next=restore(cleanup_continue),
                            loop_break=restore(cleanup_break),return_to=restore(cleanup_return),raise_to=restore(cleanup_raise))
                return cleanup_raise(state,kind)
            def normal(state):
                return statements(node.orelse,state,on_done=finish,loop_next=cleanup_continue,
                    loop_break=cleanup_break,return_to=cleanup_return,raise_to=cleanup_raise)
            return statements(node.body,env,on_done=normal,loop_next=cleanup_continue,
                loop_break=cleanup_break,return_to=cleanup_return,raise_to=handled)
        if isinstance(node, ast.Return):
            if loop_result is not None:
                raise SourceControlFlowError('source loop early returns require multi-value CFG state')
            result_nodes=node.value.elts if isinstance(node.value,ast.Tuple) else (node.value,)
            if any(isinstance(part,ast.Name) and isinstance(env.get(part.id),_StateRef) for part in result_nodes):
                raise SourceControlFlowError('return aliases of mutable state require a result-alias contract')
            value = expression(node.value, env)
            return return_to(env, value) if return_to is not None else value
        if isinstance(node, ast.Expr):
            return statements(tail, env, loop_result=loop_result, loop_next=loop_next, loop_break=loop_break, return_to=return_to, on_done=on_done, raise_to=raise_to)
        if isinstance(node, ast.Assign):
            target=node.targets[0]
            if isinstance(target,ast.Subscript):
                assert isinstance(target.value,ast.Name)
                ref=env.get(target.value.id)
                if not isinstance(ref,_StateRef):
                    raise SourceControlFlowError('mutation target is not a declared state alias')
                value=expression(node.value,env)
                previous=env[ref.root]
                if not isinstance(value,Tracer) or (value.shape,value.dtype)!=(previous.shape,previous.dtype):
                    raise SourceControlFlowError('state mutation must preserve tensor shape and dtype')
                env={**env,ref.root:value}
            else:
                assert isinstance(target,ast.Name)
                value=env[node.value.id] if isinstance(node.value,ast.Name) and isinstance(env.get(node.value.id),_StateRef) else expression(node.value,env)
                env={**env,target.id:value}
            return statements(tail, env, loop_result=loop_result, loop_next=loop_next, loop_break=loop_break, return_to=return_to, on_done=on_done, raise_to=raise_to)
        if isinstance(node, ast.If):
            predicate = condition(node.test, env)
            def branch(body):
                return statements([*body, *tail], dict(env), loop_result=loop_result, loop_next=loop_next, loop_break=loop_break, return_to=return_to, on_done=on_done, raise_to=raise_to)
            if isinstance(predicate, Tracer):
                builder = active_tracer()
                assert builder is not None
                return builder.record_cond(predicate, lambda: branch(node.body), lambda: branch(node.orelse), ())
            if type(predicate) is not bool:
                raise SourceControlFlowError('source predicate must be a boolean or traced tensor')
            return branch(node.body if predicate else node.orelse)
        if isinstance(node, ast.While):
            names = {part.targets[0].id for statement in node.body for part in ast.walk(statement) if isinstance(part, ast.Assign) and isinstance(part.targets[0], ast.Name)}
            names.update(key for key in env if key.startswith('@state'))
            expanded = bool(state_groups) or (raise_to is not None and any(isinstance(part,(ast.Raise,ast.Assert)) for part in ast.walk(node))) or len(names) != 1 or any(isinstance(part, (ast.Break, ast.Continue, ast.Return, ast.Try)) for part in ast.walk(node))
            if expanded:
                assert max_steps is not None
                if max_steps > 16:
                    raise SourceControlFlowError('expanded source CFG requires max_steps <= 16')
                returns = [part for part in ast.walk(node) if isinstance(part, ast.Return)]
                ordered = sorted(names)
                if any(name not in env or not isinstance(env[name], Tracer) for name in ordered):
                    raise SourceControlFlowError('expanded loop variables require initialized tensor values')
                builder = active_tracer()
                assert builder is not None
                from .graph_ir import IROp
                def flag(value):
                    assert builder is not None
                    ssa=builder._fresh()
                    builder.body.append(IROp(ssa, 'arith.constant', [], [], 'tensor<1xf32>',
                        attrs=f'value = dense<{1.0 if value else 0.0}> : tensor<1xf32>'))
                    return Tracer((1,), 'f32', ssa)
                yes, no = flag(True), flag(False)
                payload: tuple[Tracer,...] = ()
                if returns:
                    # Infer only the payload types in a discarded trace region.
                    # Do not place speculative return arithmetic in the program.
                    _, prototype = builder._trace_region(lambda: expression(returns[0].value,env))
                    prototype = prototype if isinstance(prototype,tuple) else (prototype,)
                    from .graph_ir import tensor_ir_type
                    values=[]
                    for value in prototype:
                        if not isinstance(value,Tracer):
                            raise SourceControlFlowError('loop return payloads must be tensors')
                        ty=str(tensor_ir_type(tuple(map(str,value.shape)),value.dtype))
                        if not ty.endswith(('xf32>','xf64>','<f32>','<f64>')):
                            raise SourceControlFlowError('loop return payloads require floating tensor storage')
                        ssa=builder._fresh()
                        builder.body.append(IROp(ssa,'arith.constant',[],[],ty,attrs=f'value = dense<0.0> : {ty}'))
                        values.append(Tracer(value.shape,value.dtype,ssa))
                    payload=tuple(values)
                state, active, returned = dict(env), yes, no
                error_kinds=tuple(edges) if raise_to is not None else ()
                error_flags=tuple(no for _ in error_kinds)
                escaping: set[tuple[str,tuple[str,...]]]=set()
                def loop_predicate(current):
                    value = condition(node.test,current)
                    return (yes if value else no) if type(value) is bool else value
                def packed(updated, running, did_return=None, result=None, errors=None):
                    return tuple(updated[name] for name in ordered)+(running, returned if did_return is None else did_return)+(error_flags if errors is None else errors)+(payload if result is None else result)
                def loop_raise(updated,kind):
                    escaping.add(kind)
                    return packed(updated,no,no,errors=tuple(yes if name==kind else no for name in error_kinds))
                def loop_return(updated, value):
                    result=value if isinstance(value,tuple) else (value,)
                    if len(result)!=len(payload):
                        raise SourceControlFlowError('loop return payload arity disagrees')
                    return packed(updated,no,yes,result,tuple(no for _ in error_kinds))
                for _ in range(max_steps):
                    current = dict(state)
                    pred = builder.record_cond(active, lambda: loop_predicate(current), lambda: no, ())
                    result = builder.record_cond(pred,
                        lambda: statements(node.body, dict(current),
                            loop_next=lambda updated: packed(updated,yes),
                            loop_break=lambda updated: packed(updated,no), return_to=loop_return, raise_to=loop_raise if raise_to is not None else None),
                        lambda: packed(current,no), ())
                    # Merge each iteration before constructing the next one;
                    # recursively cloning its continuations is exponential.
                    if not isinstance(result, tuple): result=(result,)
                    state.update(zip(ordered,result[:len(ordered)]))
                    active,returned=result[len(ordered):len(ordered)+2]
                    offset=len(ordered)+2
                    error_flags=tuple(result[offset:offset+len(error_kinds)])
                    payload=tuple(result[offset+len(error_kinds):])
                pending=builder.record_cond(active, lambda: loop_predicate(state), lambda: no, ())
                guard(pending, 'source while exceeded max_steps', stopped=True)
                def finish():
                    return statements(tail,state,loop_result=loop_result,loop_next=loop_next,loop_break=loop_break,return_to=return_to,on_done=on_done,raise_to=raise_to)
                def finish_return():
                    assert builder is not None
                    if returns:
                        value=payload[0] if len(payload)==1 else payload
                        return builder.record_cond(returned,lambda: return_to(state,value) if return_to is not None else value,finish,())
                    return finish()
                def dispatch(index):
                    assert builder is not None
                    if index==len(error_kinds):return finish_return()
                    kind=error_kinds[index]
                    if kind not in escaping:return dispatch(index+1)
                    return builder.record_cond(error_flags[index],lambda: raise_to(state,kind),lambda: dispatch(index+1),())
                return dispatch(0)
            if next(iter(names)) not in env:
                raise SourceControlFlowError('source while requires initialized tensor carries')
            name = next(iter(names))
            builder = active_tracer()
            assert builder is not None
            updated = builder.record_while(
                lambda value: condition(node.test, {**env, name: value}),
                lambda value: statements(node.body, {**env, name: value}, loop_result=name),
                env[name], max_steps)
            return statements(tail, {**env, name: updated}, loop_result=loop_result, loop_next=loop_next, loop_break=loop_break, return_to=return_to, on_done=on_done, raise_to=raise_to)
        raise SourceControlFlowError('unsupported source statement')

    def recovered(*args):
        budget[0] = 0
        names = [arg.arg for arg in (*definition.args.posonlyargs, *definition.args.args)]
        if len(args) != len(names):
            raise SourceControlFlowError('source capture arity disagrees')
        env=dict(zip(names,args))
        roots=[]
        used: set[int]=set()
        for group_index,group in enumerate(state_groups):
            if not isinstance(group,tuple) or not group or any(type(index) is not int or not 0<=index<len(args) or index in used for index in group) or len(set(group))!=len(group):
                raise SourceControlFlowError('state groups require distinct argument indices')
            used.update(group)
            root=f'@state{group_index}'
            roots.append(root)
            env[root]=args[group[0]]
            for index in group:env[names[index]]=_StateRef(root)
        from .graph_ir import IROp, tensor_ir_type
        builder=active_tracer()
        assert builder is not None
        def constant(shape,dtype,value=0):
            builder=active_tracer()
            assert builder is not None
            ty=str(tensor_ir_type(tuple(map(str,shape)),dtype))
            ssa=builder._fresh()
            builder.body.append(IROp(ssa,'arith.constant',[],[],ty,attrs=f'value = dense<{float(value)}> : {ty}'))
            return Tracer(tuple(shape),dtype,ssa)
        placeholders=[]
        for shape,dtype in error_specs:
            if dtype not in ('f32','f64') or any(type(n) is not int or n<1 for n in shape):
                raise SourceControlFlowError('exception results require static floating tensor specs')
            placeholders.append(constant(shape,dtype))
        def finish(state,value,code=0):
            values=value if isinstance(value,tuple) else (value,)
            if error_specs and (len(values)!=len(placeholders) or any((v.shape,v.dtype)!=(p.shape,p.dtype) for v,p in zip(values,placeholders,strict=True))):
                raise SourceControlFlowError('exception result specification disagrees')
            return (*values,*(state[root] for root in roots),*((constant((1,),'f32',code),) if error_specs else ()))
        def escape(state,kind):
            return finish(state,tuple(placeholders),edges.index(kind)+1)
        return statements(definition.body,env,return_to=finish if roots or error_specs else None,
                          raise_to=escape if error_specs else None)
    setattr(recovered,"source_error_table",tuple(edges))
    return recovered


def to_native_source_ir(traced, *, name='source_program', autodiff=None):
    """Serialize typed trace regions to SCF for the native compiler consumer.

    This shares the tracer's operations and SSA edges. It never reparses Python
    expressions or emits backend kernels. Bounded while exhaustion is explicit.
    """
    from .graph_ir import tensor_ir_type
    if autodiff not in (None,'forward','reverse'):raise SourceControlFlowError('invalid native source AD mode')
    function_attrs='' if autodiff is None else ' attributes {tessera.autodiff = "'+autodiff+'"}'
    if not name.isidentifier():
        raise SourceControlFlowError('native source entry must be an identifier')
    types = {ssa: str(tensor_ir_type(tuple(map(str, shape)), dtype)) for ssa, shape, dtype in traced.args}
    def collect(body):
        for op in body:
            result_types = [str(t) for t in op.inferred_types or ()]
            if len(result_types) != len(op.result_names):
                result_types = [op.result_type] if len(op.result_names) == 1 else []
            if len(result_types) != len(op.result_names):
                raise SourceControlFlowError('native source result types are incomplete')
            types.update(zip(op.result_names, result_types))
            if op.kwargs.get('_region') == 'if':
                collect(op.kwargs['_then_body']); collect(op.kwargs['_else_body'])
            elif op.kwargs.get('_region') == 'while':
                types[op.kwargs['_carry_ssa']] = op.operand_types[0]
                collect(op.kwargs['_body']); collect(op.kwargs['_cond'])
    collect(traced.body)
    count = [0]
    def fresh():
        count[0] += 1
        return f'%__source_cfg_{count[0]}'
    def predicate(ssa, lines, indent):
        # Public tensor comparisons use the existing shape-(1,) f32 mask ABI.
        if types[ssa] != 'tensor<1xf32>':
            raise SourceControlFlowError('native source predicates require shape-(1,) f32 comparison masks')
        zero, flag, value, truth = fresh(), fresh(), fresh(), fresh()
        lines.extend([f'{indent}{zero} = arith.constant 0 : index',
                      f'{indent}{flag} = tensor.extract %{ssa}[{zero}] : tensor<1xf32>',
                      f'{indent}{value} = arith.constant 0.0 : f32',
                      f'{indent}{truth} = arith.cmpf ogt, {flag}, {value} : f32'])
        return truth
    def emit(body, indent):
        lines: list[str] = []
        for op in body:
            kw = op.kwargs
            if kw.get('_region') == 'if':
                pred = predicate(kw['_flag_ssa'], lines, indent)
                result_types = ', '.join(types[result] for result in op.result_names)
                lhs = ', '.join('%'+result for result in op.result_names)
                lines.append(f'{indent}{lhs} = scf.if {pred} -> ({result_types}) {{')
                for side in ('then', 'else'):
                    if side == 'else': lines.append(f'{indent}}} else {{')
                    lines.extend(emit(kw[f'_{side}_body'], indent+'  '))
                    outputs = kw[f'_{side}_ssas']
                    lines.append(f'{indent}  scf.yield '+', '.join('%'+v for v in outputs)+' : '+result_types)
                lines.append(f'{indent}}}')
            elif kw.get('_region') == 'while':
                carry, ty = kw['_carry_ssa'], types[op.result]
                zero, limit, iteration, final_iteration = fresh(), fresh(), fresh(), fresh()
                lines.extend([f'{indent}{zero} = arith.constant 0 : index',
                              f'{indent}{limit} = arith.constant {kw["_max_iters"]} : index',
                              f'{indent}%{op.result}, {final_iteration} = scf.while (%{carry} = {op.operands[0]}, {iteration} = {zero}) : ({ty}, index) -> ({ty}, index) {{'])
                lines.extend(emit(kw['_cond'], indent+'  '))
                pred = predicate(kw['_pred_ssa'], lines, indent+'  ')
                available, stopped, valid, truth = fresh(), fresh(), fresh(), fresh()
                lines.extend([f'{indent}  {available} = arith.cmpi ult, {iteration}, {limit} : index',
                              f'{indent}  {truth} = arith.constant true',
                              f'{indent}  {stopped} = arith.xori {pred}, {truth} : i1',
                              f'{indent}  {valid} = arith.ori {stopped}, {available} : i1',
                              f'{indent}  cf.assert {valid}, "source while exceeded max_steps"',
                              f'{indent}  scf.condition({pred}) %{carry}, {iteration} : {ty}, index',
                              f'{indent}}} do {{', f'{indent}^bb0(%{carry}: {ty}, {iteration}: index):'])
                lines.extend(emit(kw['_body'], indent+'  '))
                one, next_iteration = fresh(), fresh()
                lines.extend([f'{indent}  {one} = arith.constant 1 : index',
                              f'{indent}  {next_iteration} = arith.addi {iteration}, {one} : index',
                              f'{indent}  scf.yield %{kw["_next_ssa"]}, {next_iteration} : {ty}, index', f'{indent}}}'])
            else:
                if any(key.startswith('_') for key in kw):
                    raise SourceControlFlowError('unsupported trace region in native source emitter')
                if op.op_name in {'tessera.lt','tessera.le','tessera.gt','tessera.ge','tessera.eq','tessera.ne'}:
                    # The eager tracer carries comparison masks as f32. The
                    # registered native operation returns i1; preserve the
                    # public mask ABI with an explicit typed conversion.
                    from dataclasses import replace
                    native = fresh()
                    boolean_type = op.result_type.replace('xf32>', 'xi1>')
                    if boolean_type == op.result_type:
                        raise SourceControlFlowError('native source comparison mask dtype is unsupported')
                    comparison = replace(op, result=native[1:], result_type=boolean_type)
                    lines.append(comparison.to_mlir(indent=indent, canonical=True))
                    lines.append(f'{indent}%{op.result} = arith.uitofp {native} : {boolean_type} to {op.result_type}')
                else:
                    lines.append(op.to_mlir(indent=indent, canonical=True))
        return lines
    args = ', '.join(f'%{ssa}: {types[ssa]}' for ssa, _, _ in traced.args)
    outputs = ', '.join(types[ssa] for ssa in traced.outputs)
    signature = outputs if len(traced.outputs) == 1 else '('+outputs+')'
    attributes='tessera.frontend.authority = "tracer"'
    groups=getattr(traced,'source_state_groups',())
    import json
    contract=dict(schema=1,groups=groups,arguments=[dict(shape=shape,dtype=dtype) for _,shape,dtype in traced.args],
                  object_fields=getattr(traced,'source_object_fields',()),error_specs=getattr(traced,'source_error_specs',()),error_table=getattr(traced,'source_error_table',()),
                  result_count=len(traced.outputs)-len(groups)-bool(getattr(traced,'source_error_specs',())),outputs=[types[ssa] for ssa in traced.outputs])
    attributes+=', tessera.source_state = '+json.dumps(json.dumps(contract,sort_keys=True,separators=(',',':')))
    lines = ['module attributes {'+attributes+'} {', f'  func.func @{name}({args}) -> {signature}{function_attrs} {{', *emit(traced.body, '    '),
             '    return '+', '.join('%'+ssa for ssa in traced.outputs)+' : '+outputs, '  }', '}']
    return '\n'.join(lines)+'\n'
