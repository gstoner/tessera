"""Opt-in Python source control recovery through the existing typed tracer.

This producer owns no IR emitter: branch merges and bounded loops are recorded
by TraceBuilder. Bounded returns, handled exceptions and declared tensor state
use typed SSA. Explicit error-result specs transport builtin exception classes;
arbitrary object effects and dynamic exception objects remain excluded.
"""
import ast
import inspect
import math
import textwrap
from dataclasses import dataclass


class SourceControlFlowError(ValueError):
    pass


@dataclass(frozen=True)
class _StateRef:
    root: str
    view: tuple | None = None


@dataclass(frozen=True)
class _ExceptionRef:
    edge: tuple
    payload: object = None


@dataclass(frozen=True)
class _ExceptionChoice:
    code: object
    candidates: set


def recover_callable(fn, *, max_steps=None, state_groups=(),state_views=(),error_specs=(),object_fields=()):
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
                 ast.If, ast.While, ast.Break, ast.Continue, ast.Assert, ast.Try, ast.Raise, ast.ExceptHandler, ast.Pass, ast.Expr, ast.Name, ast.Load, ast.Store,
                 ast.Call, ast.keyword, ast.Attribute, ast.Constant, ast.Tuple, ast.Subscript, ast.Slice,
                 ast.BinOp, ast.Compare, ast.UnaryOp, ast.USub, ast.Is, ast.IsNot,
                 *binary, *comparison)
    import builtins
    exception_types={name:getattr(builtins,name) for name in ('Exception','ValueError','RuntimeError','AssertionError','TypeError','IndexError','KeyError','OverflowError','ZeroDivisionError')}
    custom_types={name:value for name,value in outer.items()
                  if isinstance(value,type) and issubclass(value,Exception)
                  and name not in exception_types}
    exception_types.update(custom_types)
    raised_calls={id(value) for node in ast.walk(definition) if isinstance(node,ast.Raise) for value in (node.exc,node.cause)}
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
            if (node.exc is None or isinstance(node.exc,ast.Name)) and node.cause is None:
                continue
            if not isinstance(node.exc,ast.Call) or node.exc.keywords or len(node.exc.args)>1 or any(isinstance(a,ast.Constant) and not isinstance(a.value,str) for a in node.exc.args):
                raise SourceControlFlowError('raise requires a builtin exception with one literal message or tensor payload')
            exception_name(node.exc.func)
            if node.cause is not None and not isinstance(node.cause,ast.Name) and not (isinstance(node.cause,ast.Constant) and node.cause.value is None):
                if not isinstance(node.cause,ast.Call) or node.cause.keywords or len(node.cause.args)>1 or any(not isinstance(a,ast.Constant) or type(a.value) is not str for a in node.cause.args):
                    raise SourceControlFlowError('exception cause requires a literal builtin constructor or None')
                exception_name(node.cause.func)
        if isinstance(node,ast.ExceptHandler):
            if node.name in exception_types:
                raise SourceControlFlowError('exception bindings cannot shadow builtin exception classes')
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
    def check_custom_handlers(node, handlers=()):
        if isinstance(node,ast.Try):
            caught=tuple(kind for handler in node.handlers for kind in
                ((Exception,) if handler.type is None else tuple(exception_types[exception_name(part)]
                 for part in (handler.type.elts if isinstance(handler.type,ast.Tuple) else (handler.type,)))))
            for child in node.body:check_custom_handlers(child,handlers+caught)
            for handler in node.handlers:
                for child in handler.body:check_custom_handlers(child,handlers)
            for child in (*node.orelse,*node.finalbody):check_custom_handlers(child,handlers)
            return
        if isinstance(node,ast.Raise) and isinstance(node.exc,ast.Call) and isinstance(node.exc.func,ast.Name):
            kind=exception_types[node.exc.func.id]
            has_custom=any(isinstance(value,ast.Call) and isinstance(value.func,ast.Name)
                           and value.func.id in custom_types for value in (node.exc,node.cause))
            if has_custom and any(issubclass(kind,handler) for handler in handlers):
                raise SourceControlFlowError('custom exception constructors in handled native paths are unsupported')
        for descendant in ast.iter_child_nodes(node):check_custom_handlers(descendant,handlers)
    check_custom_handlers(definition)

    edges=[]
    dynamic_edges={}
    raise_edges={}
    value: tuple
    cause: tuple
    for node in ast.walk(definition):
        if isinstance(node,ast.Raise) and isinstance(node.exc,ast.Call):
            if node.exc.args and not isinstance(node.exc.args[0],ast.Constant):
                value=(exception_name(node.exc.func),('@tensor',str(node.lineno)))
                dynamic_edges[id(node)]=value
            else:value=(exception_name(node.exc.func),tuple(literal(arg) for arg in node.exc.args))
        elif isinstance(node,ast.Assert):
            value=('AssertionError',(literal(node.msg),) if node.msg is not None else ())
        else:continue
        if isinstance(node,ast.Raise):
            if node.cause is not None:
                if isinstance(node.cause,ast.Constant):cause=('suppress',())
                elif isinstance(node.cause,ast.Name):cause=('binding',(node.cause.id,))
                else:
                    assert isinstance(node.cause,ast.Call)
                    cause=(exception_name(node.cause.func),tuple(literal(a) for a in node.cause.args))
                value=(*value,cause)
            if len(value)==2:value=(*value,None)
            value=(*value,(inspect.getsourcefile(fn) or '<source>',first_line+node.lineno-1),None)
            raise_edges[id(node)]=value
            if id(node) in dynamic_edges:dynamic_edges[id(node)]=value
        if value not in edges:edges.append(value)
    if len(edges)>32:raise SourceControlFlowError('source exception table exceeds 32 static edges')
    # Expanded loops are bounded, so each syntactic raise/generation gets a
    # distinct payload slot. A later iteration must not overwrite a retained
    # exception from an earlier one.
    import itertools
    loop_depths={}
    def locate(node,depth=0):
        if id(node) in dynamic_edges:loop_depths[id(node)]=depth
        for child in ast.iter_child_nodes(node):locate(child,depth+isinstance(node,ast.While))
    locate(definition)
    def site_name(line,generation):
        return line+(':'+'.'.join(map(str,generation)) if generation else '')
    if error_specs and sum((max_steps or 1)**depth for depth in loop_depths.values())>32:
        raise SourceControlFlowError('source exception generations exceed 32 payload slots')
    payload_sites=tuple(site_name(edge[1][1],g) for key,edge in dynamic_edges.items()
        for g in itertools.product(range(max_steps or 1),repeat=loop_depths[key])) if error_specs else ()
    if len(payload_sites)>32:raise SourceControlFlowError('source exception generations exceed 32 payload slots')
    generation: list[int]=[]
    budget = [0]

    def exception_code(number):
        from .graph_ir import IROp
        builder=active_tracer()
        assert builder is not None
        ssa=builder._fresh()
        builder.body.append(IROp(ssa,'arith.constant',[],[],'tensor<1xf32>',attrs=f'value = dense<{float(number)}> : tensor<1xf32>'))
        return Tracer((1,),'f32',ssa)

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

    def view_ref(node,env):
        import numpy as np
        if not isinstance(node.value,ast.Name) or not isinstance(env.get(node.value.id),(_StateRef,Tracer)):
            raise SourceControlFlowError('source slices require a named tensor root')
        ref=env[node.value.id]
        if isinstance(ref,Tracer):
            key='@value:'+ref.ssa
            env[key]=ref
            ref=_StateRef(key)
        root=env[ref.root]
        slices=node.slice.elts if isinstance(node.slice,ast.Tuple) else [node.slice]
        selections=[]
        for item in slices:
            if not isinstance(item,ast.Slice):raise SourceControlFlowError('source views require rank-preserving static slices')
            parts: list[int | None | Tracer]=[]
            for part in (item.lower,item.upper,item.step):
                if part is None:parts.append(None)
                elif isinstance(part,ast.Constant) and type(part.value) is int:parts.append(part.value)
                elif isinstance(part,ast.UnaryOp) and isinstance(part.op,ast.USub) and isinstance(part.operand,ast.Constant) and type(part.operand.value) is int:parts.append(-part.operand.value)
                else:
                    value=expression(part,env)
                    if not isinstance(value,Tracer) or value.shape!=(1,) or value.dtype not in ('i64','int64'):
                        raise SourceControlFlowError('runtime slice bounds require single-element int64 tensors')
                    parts.append(value)
            if type(parts[2]) is int and parts[2]==0:raise SourceControlFlowError('source slice stride cannot be zero')
            selections.append(slice(*parts))
        if (ref.view is not None and ref.view[0]=='dynamic') or any(isinstance(v,Tracer) for selection in selections for v in (selection.start,selection.stop,selection.step)):
            return runtime_view(ref,root,selections)
        if len(root.shape)==1 and len(selections)==1 and (ref.view is None or isinstance(ref.view[0],int)):
            offset,length,stride=ref.view or (0,root.shape[0],1)
            start,stop,step=selections[0].indices(length)
            count=len(range(start,stop,step))
            if not count:raise SourceControlFlowError('source state slices must be nonempty')
            if stride*step>0:return _StateRef(ref.root,(offset+start*stride,count,stride*step))
        if ref.view is None or ref.view[0]=='rect':
            offsets,shape,strides=((0,)*len(root.shape),root.shape,(1,)*len(root.shape)) if ref.view is None else ref.view[1:]
            if len(selections)>len(shape):raise SourceControlFlowError('source slice rank disagrees')
            full=[*selections,*[slice(None)]*(len(shape)-len(selections))]
            bounds=[selection.indices(n) for selection,n in zip(full,shape,strict=True)]
            counts=tuple(len(range(*bound)) for bound in bounds)
            if not all(counts):raise SourceControlFlowError('source state slices must be nonempty')
            if all(bound[2]>0 for bound in bounds):
                return _StateRef(ref.root,('rect',tuple(o+b[0]*st for o,b,st in zip(offsets,bounds,strides,strict=True)),counts,tuple(b[2]*st for b,st in zip(bounds,strides,strict=True))))
        # Bounded static maps retain root coordinates and lower to standard
        # singleton slices, whose transpose is owned by the native paired pass.
        shape,coordinates=view_coordinates(ref,root.shape)
        if len(selections)>len(shape):raise SourceControlFlowError('source slice rank disagrees')
        selected=np.arange(len(coordinates)).reshape(shape)[tuple(selections)]
        if not selected.size or selected.size>256:raise SourceControlFlowError('mapped source views require one through 256 elements')
        return _StateRef(ref.root,(tuple(coordinates[int(i)] for i in selected.flat),tuple(selected.shape)))

    def runtime_view(ref,root,selections):
        # Compose signed logical maps into the original immutable allocation.
        # Dynamic maps are gathered, never encoded as negative memref strides.
        if len(selections)>len(root.shape) or any(type(n) is not int or n<1 for n in root.shape):
            raise SourceControlFlowError('runtime slicing requires a static ranked root')
        previous: tuple
        if ref.view is None:
            previous=((0,)*len(root.shape),root.shape,(1,)*len(root.shape))
        elif ref.view[0] in ('dynamic','rect'):previous=ref.view[1:]
        elif isinstance(ref.view[0],int):previous=tuple((v,) for v in ref.view)
        else:raise SourceControlFlowError('runtime slicing cannot compose an enumerated coordinate map')
        from .graph_ir import IROp
        def emit(name,args=(),ty='index',attrs=None):
            builder=active_tracer()
            assert builder is not None
            ssa=builder._fresh() if ty else None
            builder.body.append(IROp(ssa,name,[v[0] for v in args],[v[1] for v in args],ty,attrs=attrs))
            return ('%'+ssa,ty) if ssa is not None else None
        def integer(value):return emit('arith.constant',attrs=f'value = {value} : index')
        zero,one,minus_one=integer(0),integer(1),integer(-1)
        def index(value):
            if type(value) is int:return integer(value)
            if type(value) is str:return (value,'index')
            scalar=emit('tensor.extract',[("%"+value.ssa,'tensor<1xi64>'),zero],ty='i64')
            return emit('arith.index_cast',[scalar])
        def compare(a,b,predicate):return emit('arith.cmpi',[a,b],ty='i1',attrs=f'predicate = {predicate} : i64')
        def select(flag,a,b):return emit('arith.select',[flag,a,b])
        offsets=[];lengths=[];strides=[]
        full=[*selections,*[slice(None)]*(len(root.shape)-len(selections))]
        for dim,(selection,root_n) in enumerate(zip(full,root.shape,strict=True)):
            n=index(previous[1][dim]);root_end=integer(root_n)
            last=emit('arith.subi',[n,one])
            raw=one if selection.step is None else index(selection.step)
            emit('cf.assert',[compare(raw,zero,1)],ty=None,attrs='msg = "runtime slice step cannot be zero"')
            negative=compare(raw,zero,2)
            # Clip before negation, including INT64_MIN. Steps beyond the parent
            # extent select at most one element and are equivalent after clipping.
            cap=integer(root_n+1);floor=integer(-root_n-1)
            bounded=emit('arith.minsi',[emit('arith.maxsi',[raw,floor]),cap])
            magnitude=select(negative,emit('arith.subi',[zero,bounded]),bounded)
            # An explicit unsigned range exposes [1, root_n+1] to allocation proofs.
            magnitude=emit('arith.addi',[emit('arith.minui',[emit('arith.subi',[magnitude,one]),root_end]),one])
            signed_step=select(negative,emit('arith.subi',[zero,magnitude]),magnitude)
            def normalize(value,positive_default,negative_default):
                if value is None:return select(negative,negative_default,positive_default)
                raw_index=index(value)
                shifted=emit('arith.addi',[raw_index,n])
                adjusted=select(compare(raw_index,zero,2),shifted,raw_index)
                lower=select(negative,minus_one,zero);upper=select(negative,last,n)
                return emit('arith.minsi',[emit('arith.maxsi',[adjusted,lower]),upper])
            start=normalize(selection.start,zero,last);stop=normalize(selection.stop,n,minus_one)
            distance=select(negative,emit('arith.subi',[start,stop]),emit('arith.subi',[stop,start]))
            distance=emit('arith.minui',[emit('arith.maxsi',[distance,zero]),root_end])
            rounded=emit('arith.addi',[distance,emit('arith.subi',[magnitude,one])])
            count=emit('arith.divui',[rounded,magnitude])
            offset=emit('arith.addi',[index(previous[0][dim]),emit('arith.muli',[start,index(previous[2][dim])])])
            step=emit('arith.muli',[signed_step,index(previous[2][dim])])
            step=emit('arith.minsi',[emit('arith.maxsi',[step,floor]),cap])
            offsets.append(offset[0]);lengths.append(count[0]);strides.append(step[0])
        return _StateRef(ref.root,('dynamic',tuple(offsets),tuple(lengths),tuple(strides)))

    def view_coordinates(ref,shape):
        import itertools
        if ref.view is None:
            if math.prod(shape)>256:raise SourceControlFlowError('mapped source roots require at most 256 elements')
            return tuple(shape),tuple(itertools.product(*(range(n) for n in shape)))
        if ref.view[0]=='rect':
            offsets,shape,strides=ref.view[1:]
            if math.prod(shape)>256:raise SourceControlFlowError('mapped source roots require at most 256 elements')
            return shape,tuple(tuple(o+i*st for o,i,st in zip(offsets,index,strides,strict=True)) for index in itertools.product(*(range(n) for n in shape)))
        if isinstance(ref.view[0],int):
            offset,length,stride=ref.view
            if length>256:raise SourceControlFlowError('mapped source roots require at most 256 elements')
            return (length,),tuple((offset+i*stride,) for i in range(length))
        coordinates,shape=ref.view
        return shape,coordinates

    def state_read(ref,env):
        root=env[ref.root]
        if ref.view is None:return root
        if ref.view[0]=='dynamic':
            offsets,lengths,strides=ref.view[1:]
            return slice_op('tensor.generate',[root],('?',)*len(lengths),offsets,lengths,strides)
        if ref.view[0]=='rect':
            offsets,shape,strides=ref.view[1:]
            return slice_op('tensor.extract_slice',[root],shape,offsets,shape,strides)
        if isinstance(ref.view[0],int):
            offset,length,stride=ref.view
            return slice_op('tensor.extract_slice',[root],(length,),offset,length,stride)
        return mapped_slice(ref,root)

    def mapped_slice(ref,root,value=None):
        import itertools
        shape,coordinates=view_coordinates(ref,root.shape)
        if len(shape)!=len(root.shape):raise SourceControlFlowError('mapped views currently preserve root rank')
        if value is None:
            result=constant_tensor(shape,root.dtype)
        else:result=root
        unit=(1,)*len(shape)
        for local,physical in zip(itertools.product(*(range(n) for n in shape)),coordinates,strict=True):
            source=value if value is not None else root
            chunk=slice_op('tensor.extract_slice',[source],unit,local if value is not None else physical,unit,unit)
            result=slice_op('tensor.insert_slice',[chunk,result],result.shape,physical if value is not None else local,unit,unit)
        return result

    def constant_tensor(shape,dtype):
        from .graph_ir import IROp,tensor_ir_type
        builder=active_tracer()
        assert builder is not None
        ty=str(tensor_ir_type(tuple(map(str,shape)),dtype));ssa=builder._fresh()
        builder.body.append(IROp(ssa,'arith.constant',[],[],ty,attrs=f'value = dense<0.0> : {ty}'))
        return Tracer(tuple(shape),dtype,ssa)

    def slice_op(name,values,shape,offset,length,stride=1):
        from .graph_ir import IROp,tensor_ir_type
        builder=active_tracer()
        assert builder is not None
        ty=str(tensor_ir_type(tuple(map(str,shape)),values[0].dtype))
        operand_types=[str(tensor_ir_type(tuple(map(str,v.shape)),v.dtype)) for v in values]
        ssa=builder._fresh()
        builder.body.append(IROp(ssa,name,['%'+v.ssa for v in values],operand_types,ty,kwargs=dict(offset=offset,length=length,stride=stride)))
        return Tracer(tuple(shape),values[0].dtype,ssa)

    def evaluate(node, env):
        if isinstance(node, ast.Name):
            if node.id in env:
                value=env[node.id]
                return state_read(value,env) if isinstance(value,_StateRef) else value
            if node.id in outer:
                return outer[node.id]
            raise SourceControlFlowError(f'unknown source value {node.id}')
        if isinstance(node,ast.Subscript):return state_read(view_ref(node,env),env)
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
            if isinstance(node.ops[0],(ast.Is,ast.IsNot)):
                left,right=expression(node.left,env),expression(node.comparators[0],env)
                if isinstance(left,_ExceptionChoice) or isinstance(right,_ExceptionChoice):
                    def identity(value):
                        if isinstance(value,_ExceptionChoice):return value.code
                        if isinstance(value,_ExceptionRef):return exception_code(edges.index(value.edge)+1)
                        if value is None:return exception_code(0)
                        raise SourceControlFlowError('exception identity requires a caught reference or None')
                    return allowed['eq' if isinstance(node.ops[0],ast.Is) else 'ne'](identity(left),identity(right))
                if not isinstance(left,_ExceptionRef) or not isinstance(right,_ExceptionRef):raise SourceControlFlowError('source identity comparison requires caught exceptions')
                return (left is right) if isinstance(node.ops[0],ast.Is) else (left is not right)
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
                if '@handled_payload' in env:env={**env,'@error_payload':env['@handled_payload']}
                return raise_to({**env,'@raised_reference':env.get('@caught_reference',_ExceptionRef(env['@exception']))},env['@exception'])
            if isinstance(node.exc,ast.Name):
                reference=env.get(node.exc.id)
                if isinstance(reference,_ExceptionChoice):
                    missing=('TypeError',('exceptions must derive from BaseException',))
                    if missing not in edges:
                        if len(edges)>=32:raise SourceControlFlowError('source exception table exceeds 32 static edges')
                        edges.append(missing)
                    choices=sorted(reference.candidates,key=edges.index)
                    def select_reference(i):
                        if i==len(choices):return raise_to(env,missing)
                        edge=choices[i]
                        state=dict(env)
                        if edge[1][:1]==('@tensor',):state['@error_payload']=state['@payload:'+edge[1][1]]
                        builder=active_tracer()
                        assert builder is not None
                        predicate=allowed['eq'](reference.code,exception_code(edges.index(edge)+1))
                        return builder.record_cond(predicate,lambda: raise_to(state,edge),lambda: select_reference(i+1),())
                    return select_reference(0)
                if not isinstance(reference,_ExceptionRef):raise SourceControlFlowError('raise name requires a caught exception binding')
                state={**env,'@error_payload':reference.payload} if reference.payload is not None else env
                return raise_to({**state,'@raised_reference':reference},reference.edge)
            assert isinstance(node.exc,ast.Call)
            kind=raise_edges[id(node)]
            if id(node) in dynamic_edges and generation:
                kind=(kind[0],('@tensor',site_name(kind[1][1],generation)),*kind[2:])
            cause=kind[2]
            if cause is not None and cause[0]=='binding':
                name=cause[1][0]
                if name not in env:raise SourceControlFlowError('unknown exception cause binding')
                reference=env[name]
                if isinstance(reference,_ExceptionChoice):
                    choices=sorted(reference.candidates,key=edges.index)
                    def select_cause(i):
                        if i==len(choices):
                            state={**env,name:None}
                            return statements(nodes,state,loop_result=loop_result,loop_next=loop_next,loop_break=loop_break,return_to=return_to,on_done=on_done,raise_to=raise_to)
                        edge=choices[i]
                        payload=env.get('@payload:'+edge[1][1]) if edge[1][:1]==('@tensor',) else None
                        state={**env,name:_ExceptionRef(edge,payload)}
                        builder=active_tracer()
                        assert builder is not None
                        predicate=allowed['eq'](reference.code,exception_code(edges.index(edge)+1))
                        return builder.record_cond(predicate,lambda: statements(nodes,state,loop_result=loop_result,loop_next=loop_next,loop_break=loop_break,return_to=return_to,on_done=on_done,raise_to=raise_to),lambda:select_cause(i+1),())
                    return select_cause(0)
                if reference is None:cause=('suppress',())
                elif isinstance(reference,_ExceptionRef):cause=('edge',reference.edge)
                else:raise SourceControlFlowError('exception cause binding requires a caught exception')
            context=env.get('@exception')
            kind=(*kind[:2],cause,kind[3],context,site_name(str(node.lineno)+":"+str(node.col_offset),generation))
            if kind not in edges:
                if len(edges)>=32:raise SourceControlFlowError('source exception table exceeds 32 static edges')
                edges.append(kind)
            if id(node) in dynamic_edges:
                argument=node.exc.args[0]
                if isinstance(argument,ast.Name) and isinstance(env.get(argument.id),_StateRef):
                    raise SourceControlFlowError('dynamic exception payload aliases mutable state; use an explicit value expression')
                raised_payload=expression(node.exc.args[0],env)
                if not isinstance(raised_payload,Tracer) or raised_payload.shape!=(1,) or raised_payload.dtype not in ('f32','fp32'):
                    raise SourceControlFlowError('dynamic exceptions require a rank-one single-element f32 tensor payload')
                return raise_to({**env,'@error_payload':raised_payload,**({'@payload:'+kind[1][1]:raised_payload} if payload_sites else {}),'@raised_reference':_ExceptionRef(kind,raised_payload)},kind)
            return raise_to({**env,'@raised_reference':_ExceptionRef(kind)},kind)
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
                            updated.pop('@handled_payload',None)
                            if '@handled_payload' in state:updated['@handled_payload']=state['@handled_payload']
                            return callback(updated,*args)
                        return leave
                    active={**state,'@exception':payload[0],**({'@handled_payload':state['@error_payload']} if dynamic_edges else {})} if pending_exception else state
                    return statements(node.finalbody,active,on_done=restore(lambda updated: continuation(updated,*payload)),
                        loop_next=restore(loop_next),loop_break=restore(loop_break),return_to=restore(return_to),raise_to=restore(raise_to))
                return run
            cleanup_return=cleanup(returned)
            cleanup_raise=cleanup(raised,pending_exception=True)
            cleanup_break=cleanup(loop_break) if loop_break is not None else None
            cleanup_continue=cleanup(loop_next) if loop_next is not None else None
            finish=cleanup(after)
            def handled(state,kind):
                reference=state.get('@raised_reference')
                if not isinstance(reference,_ExceptionRef) or reference.edge!=kind:reference=_ExceptionRef(kind,state.get('@error_payload') if kind[1][:1]==('@tensor',) else None)
                for handler in node.handlers:
                    matches=handler.type is None or any(issubclass(exception_types[kind[0]],exception_types[exception_name(t)]) for t in (handler.type.elts if isinstance(handler.type,ast.Tuple) else (handler.type,)))
                    if matches:
                        def restore(continuation):
                            if continuation is None:return None
                            def run(updated,*payload):
                                updated=dict(updated)
                                updated.pop('@exception',None)
                                if handler.name:updated.pop(handler.name,None)
                                updated.pop('@caught_reference',None)
                                if '@caught_reference' in state:updated['@caught_reference']=state['@caught_reference']
                                if '@exception' in state:updated['@exception']=state['@exception']
                                updated.pop('@handled_payload',None)
                                if '@handled_payload' in state:updated['@handled_payload']=state['@handled_payload']
                                return continuation(updated,*payload)
                            return run
                        return statements(handler.body,{**state,**({handler.name:reference} if handler.name else {}),'@exception':kind,'@caught_reference':reference,**({'@handled_payload':state['@error_payload']} if dynamic_edges else {})},on_done=restore(finish),loop_next=restore(cleanup_continue),
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
            if any(isinstance(part,ast.Name) and isinstance(env.get(part.id),_StateRef) and env[part.id].root.startswith('@state') for part in result_nodes):
                raise SourceControlFlowError('return aliases of mutable state require a result-alias contract')
            value = expression(node.value, env)
            if any(isinstance(v,_ExceptionRef) for v in (value if isinstance(value,tuple) else (value,))):raise SourceControlFlowError('exception objects cannot escape into tensor state')
            return return_to(env, value) if return_to is not None else value
        if isinstance(node, (ast.Expr,ast.Pass)):
            return statements(tail, env, loop_result=loop_result, loop_next=loop_next, loop_break=loop_break, return_to=return_to, on_done=on_done, raise_to=raise_to)
        if isinstance(node, ast.Assign):
            target=node.targets[0]
            if isinstance(target,ast.Subscript):
                assert isinstance(target.value,ast.Name)
                ref=env.get(target.value.id)
                if not isinstance(ref,_StateRef) or not ref.root.startswith('@state') or (ref.view is not None and ref.view[0]=='dynamic'):
                    raise SourceControlFlowError('mutation target is not a declared state alias')
                value=expression(node.value,env)
                previous=state_read(ref,env)
                if not isinstance(value,Tracer) or (value.shape,value.dtype)!=(previous.shape,previous.dtype):
                    raise SourceControlFlowError('state mutation must preserve tensor shape and dtype')
                if ref.view is not None:
                    if ref.view[0]=='rect':value=slice_op('tensor.insert_slice',[value,env[ref.root]],env[ref.root].shape,*ref.view[1:])
                    else:value=slice_op('tensor.insert_slice',[value,env[ref.root]],env[ref.root].shape,*ref.view) if isinstance(ref.view[0],int) else mapped_slice(ref,env[ref.root],value)
                env={**env,ref.root:value}
            else:
                assert isinstance(target,ast.Name)
                value=view_ref(node.value,env) if isinstance(node.value,ast.Subscript) else env[node.value.id] if isinstance(node.value,ast.Name) and isinstance(env.get(node.value.id),_StateRef) else expression(node.value,env)
                previous=env.get(target.id)
                if isinstance(previous,_ExceptionChoice):
                    if isinstance(value,_ExceptionRef):
                        previous.candidates.add(value.edge)
                        value=_ExceptionChoice(exception_code(edges.index(value.edge)+1),previous.candidates)
                    elif value is None:value=_ExceptionChoice(exception_code(0),previous.candidates)
                    elif isinstance(value,_ExceptionChoice):
                        previous.candidates.update(value.candidates)
                        value=_ExceptionChoice(value.code,previous.candidates)
                    else:raise SourceControlFlowError('exception carry cannot change to a tensor')
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
            names.update(key for key in env if key.startswith('@state') or key=='@error_payload' or key.startswith('@payload:'))
            expanded = bool(state_groups) or (raise_to is not None and any(isinstance(part,(ast.Raise,ast.Assert)) for part in ast.walk(node))) or len(names) != 1 or any(isinstance(part, (ast.Break, ast.Continue, ast.Return, ast.Try)) for part in ast.walk(node))
            if expanded:
                assert max_steps is not None
                if max_steps > 16:
                    raise SourceControlFlowError('expanded source CFG requires max_steps <= 16')
                returns = [part for part in ast.walk(node) if isinstance(part, ast.Return)]
                ordered = sorted(names)
                env=dict(env)
                for name in ordered:
                    if name in env and env[name] is None:env[name]=_ExceptionChoice(exception_code(0),set())
                exception_carries={name:env[name].candidates for name in ordered if isinstance(env.get(name),_ExceptionChoice)}
                if any(name not in env or not isinstance(env[name], (Tracer,_ExceptionChoice)) for name in ordered):
                    raise SourceControlFlowError('expanded loop variables require initialized tensor values')
                builder = active_tracer()
                assert builder is not None
                from .graph_ir import IROp
                def flag(value):
                    builder=active_tracer()
                    assert builder is not None
                    ssa=builder._fresh()
                    builder.body.append(IROp(ssa, 'arith.constant', [], [], 'tensor<1xf32>',
                        attrs=f'value = dense<{float(value)}> : tensor<1xf32>'))
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
                error_code=no
                escaping: set[tuple]=set()
                def loop_predicate(current):
                    value = condition(node.test,current)
                    return (yes if value else no) if type(value) is bool else value
                def packed(updated, running, did_return=None, result=None, errors=None):
                    return tuple(updated[name].code if isinstance(updated[name],_ExceptionChoice) else updated[name] for name in ordered)+(running, returned if did_return is None else did_return)+((error_code if errors is None else errors),)+(payload if result is None else result)
                def loop_raise(updated,kind):
                    escaping.add(kind)
                    return packed(updated,no,no,errors=flag(edges.index(kind)+1))
                def loop_return(updated, value):
                    result=value if isinstance(value,tuple) else (value,)
                    if len(result)!=len(payload):
                        raise SourceControlFlowError('loop return payload arity disagrees')
                    return packed(updated,no,yes,result,no)
                for iteration in range(max_steps):
                    current = dict(state)
                    pred = builder.record_cond(active, lambda: loop_predicate(current), lambda: no, ())
                    def iteration_body():
                        generation.append(iteration)
                        try:
                            return statements(node.body, dict(current),
                                loop_next=lambda updated: packed(updated,yes),
                                loop_break=lambda updated: packed(updated,no), return_to=loop_return, raise_to=loop_raise if raise_to is not None else None)
                        finally:generation.pop()
                    result = builder.record_cond(pred,iteration_body,lambda: packed(current,no), ())
                    # Merge each iteration before constructing the next one;
                    # recursively cloning its continuations is exponential.
                    if not isinstance(result, tuple): result=(result,)
                    state.update((name,_ExceptionChoice(value,exception_carries[name]) if name in exception_carries else value) for name,value in zip(ordered,result[:len(ordered)],strict=True))
                    active,returned=result[len(ordered):len(ordered)+2]
                    offset=len(ordered)+2
                    error_code=result[offset]
                    payload=tuple(result[offset+1:])
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
                error_kinds=tuple(edges) if raise_to is not None else ()
                def dispatch(index):
                    builder=active_tracer()
                    assert builder is not None
                    if index==len(error_kinds):return finish_return()
                    kind=error_kinds[index]
                    if kind not in escaping:return dispatch(index+1)
                    return builder.record_cond(allowed['eq'](error_code,flag(index+1)),lambda: raise_to(state,kind),lambda: dispatch(index+1),())
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
        if not isinstance(state_views,tuple) or any(not isinstance(row,tuple) or len(row) not in (3,4,5) or type(row[0]) is not int for row in state_views):
            raise SourceControlFlowError('state views require integer argument, offset and length triples')
        normalized_views=tuple((*row,1) if len(row)==3 and isinstance(row[1],int) else row for row in state_views)
        if len({row[0] for row in state_views})!=len(state_views) or any(row[0] not in {i for g in state_groups for i in g} for row in state_views):
            raise SourceControlFlowError('state view indices must name distinct declared aliases')
        roots=[]
        used: set[int]=set()
        for group_index,group in enumerate(state_groups):
            if not isinstance(group,tuple) or not group or any(type(index) is not int or not 0<=index<len(args) or index in used for index in group) or len(set(group))!=len(group):
                raise SourceControlFlowError('state groups require distinct argument indices')
            used.update(group)
            root=f'@state{group_index}'
            roots.append(root)
            env[root]=args[group[0]]
            for index in group:
                view=next((row[1:] for row in normalized_views if row[0]==index),None)
                if view is not None and view[0]=='rect':
                    offsets,shape,strides=view[1:]
                    root_shape=args[group[0]].shape
                    if tuple(args[index].shape)!=shape or any(len(v)!=len(root_shape) for v in (offsets,shape,strides)) or any(type(v) is not int for row in (offsets,shape,strides) for v in row) or any(o<0 or n<1 or st<1 or o+(n-1)*st>=r for o,n,st,r in zip(offsets,shape,strides,root_shape,strict=True)) or args[index].dtype!=args[group[0]].dtype:
                        raise SourceControlFlowError('rectangular state view escapes its typed root')
                elif view is not None and not isinstance(view[0],int):
                    coordinates,shape=view
                    if tuple(args[index].shape)!=shape or len(shape)!=len(args[group[0]].shape) or len(coordinates)!=math.prod(shape) or not 1<=len(coordinates)<=256 or len(set(coordinates))!=len(coordinates) or any(len(c)!=len(shape) or any(type(v) is not int or not 0<=v<n for v,n in zip(c,args[group[0]].shape,strict=True)) for c in coordinates) or args[index].dtype!=args[group[0]].dtype:
                        raise SourceControlFlowError('state view escapes its typed containing input')
                elif view is not None:
                    offset,length,stride=view
                    if len(args[group[0]].shape)!=1 or args[index].shape!=(length,) or offset<0 or length<1 or stride<1 or offset+(length-1)*stride>=args[group[0]].shape[0] or args[index].dtype!=args[group[0]].dtype:
                        raise SourceControlFlowError('state view escapes its typed containing input')
                elif (args[index].shape,args[index].dtype)!=(args[group[0]].shape,args[group[0]].dtype):
                    raise SourceControlFlowError('state alias requires an explicit view map')
                env[names[index]]=_StateRef(root,view)
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
        if dynamic_edges:
            env['@error_payload']=constant((1,),'f32')
            for site in payload_sites:env['@payload:'+site]=constant((1,),'f32')
        placeholders=[]
        for shape,dtype in error_specs:
            if dtype not in ('f32','f64') or any(type(n) is not int or n<1 for n in shape):
                raise SourceControlFlowError('exception results require static floating tensor specs')
            placeholders.append(constant(shape,dtype))
        def finish(state,value,code=0):
            values=value if isinstance(value,tuple) else (value,)
            if error_specs and (len(values)!=len(placeholders) or any((v.shape,v.dtype)!=(p.shape,p.dtype) for v,p in zip(values,placeholders,strict=True))):
                raise SourceControlFlowError('exception result specification disagrees')
            return (*values,*(state[root] for root in roots),*((constant((1,),'f32',code),) if error_specs else ()),*((state['@error_payload'],) if error_specs and dynamic_edges else ()),*(state['@payload:'+site] for site in payload_sites))
        def escape(state,kind):
            return finish(state,tuple(placeholders),edges.index(kind)+1)
        result=statements(definition.body,env,return_to=finish if roots or error_specs else None,
                          raise_to=escape if error_specs else None)
        setattr(recovered,'source_error_table',tuple(edges))
        return result
    setattr(recovered,"source_error_payload_sites",payload_sites)
    setattr(recovered,"source_error_dynamic",bool(dynamic_edges and error_specs))
    setattr(recovered,"source_exception_types",custom_types)
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
    def slice_text(value):
        return ', '.join(map(str,value)) if isinstance(value,(tuple,list)) else str(value)
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
            elif op.kwargs.get('_region') in ('while', 'for'):
                types[op.kwargs['_carry_ssa']] = op.operand_types[0]
                collect(op.kwargs['_body'])
                if op.kwargs.get('_region') == 'while': collect(op.kwargs['_cond'])
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
            elif kw.get('_region') == 'for':
                trip = kw['_trip']
                if type(trip) is not int or trip < 0:
                    raise SourceControlFlowError('native for requires a nonnegative integer trip count')
                carry, ty = kw['_carry_ssa'], types[op.result]
                zero, limit, step, iv = fresh(), fresh(), fresh(), fresh()
                lines.extend([f'{indent}{zero} = arith.constant 0 : index',
                              f'{indent}{limit} = arith.constant {trip} : index',
                              f'{indent}{step} = arith.constant 1 : index',
                              f'{indent}%{op.result} = scf.for {iv} = {zero} to {limit} step {step} iter_args(%{carry} = {op.operands[0]}) -> ({ty}) {{'])
                lines.extend(emit(kw['_body'], indent+'  '))
                lines.extend([f'{indent}  scf.yield %{kw["_next_ssa"]} : {ty}', f'{indent}}}'])
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
            elif op.op_name in {'arith.index_cast','arith.cmpi','arith.select','arith.maxsi','arith.minsi','arith.minui','arith.addi','arith.subi','arith.muli','arith.divui'}:
                attrs=' {'+op.attrs+'}' if op.attrs else ''
                lines.append(f'{indent}%{op.result} = "{op.op_name}"('+', '.join(op.operands)+')'+attrs+' : ('+', '.join(op.operand_types)+') -> '+op.result_type)
            elif op.op_name=='tensor.generate':
                dims=kw['length'];ivs=[fresh() for _ in dims]
                lines.append(f"{indent}%{op.result} = tensor.generate "+', '.join(dims)+" {")
                lines.append(f"{indent}^bb0("+', '.join(v+': index' for v in ivs)+'):' )
                coordinates=[]
                root_shape=op.operand_types[0].removeprefix('tensor<').split('x')[:-1]
                for iv,offset,stride,n in zip(ivs,kw['offset'],kw['stride'],root_shape,strict=True):
                    product,physical,zero,upper,nonnegative,bounded=[fresh() for _ in range(6)]
                    lines.extend([f'{indent}  {product} = arith.muli {iv}, {stride} : index',
                        f'{indent}  {physical} = arith.addi {offset}, {product} : index',
                        f'{indent}  {zero} = arith.constant 0 : index',
                        f'{indent}  {upper} = arith.constant {int(n)-1} : index',
                        f'{indent}  {nonnegative} = arith.maxsi {physical}, {zero} : index',
                        f'{indent}  {bounded} = arith.minui {nonnegative}, {upper} : index'])
                    coordinates.append(bounded)
                element=fresh();element_type=op.operand_types[0].split('x')[-1][:-1]
                lines.append(f"{indent}  {element} = tensor.extract {op.operands[0]}["+', '.join(coordinates)+'] : '+op.operand_types[0])
                lines.append(f'{indent}  tensor.yield {element} : {element_type}')
                lines.append(f'{indent}}} : {op.result_type}')
            elif op.op_name=='tensor.extract_slice':
                lines.append(f"{indent}%{op.result} = tensor.extract_slice {op.operands[0]}[{slice_text(kw['offset'])}] [{slice_text(kw['length'])}] [{slice_text(kw.get('stride',1))}] : {op.operand_types[0]} to {op.result_type}")
            elif op.op_name=='tensor.insert_slice':
                lines.append(f"{indent}%{op.result} = tensor.insert_slice {op.operands[0]} into {op.operands[1]}[{slice_text(kw['offset'])}] [{slice_text(kw['length'])}] [{slice_text(kw.get('stride',1))}] : {op.operand_types[0]} into {op.result_type}")
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
    instruction_sites={str(edge[3][1]): index for index,edge in enumerate(getattr(traced,'source_error_table',()))
                       if len(edge)>3 and isinstance(edge[3],(list,tuple)) and len(edge[3])==2}
    contract=dict(schema=1,function_name=name,instruction_sites=instruction_sites,groups=groups,state_views=getattr(traced,'source_state_views',()),arguments=[dict(shape=shape,dtype=dtype) for _,shape,dtype in traced.args],
                  object_fields=getattr(traced,'source_object_fields',()),error_specs=getattr(traced,'source_error_specs',()),error_table=getattr(traced,'source_error_table',()),error_dynamic=getattr(traced,'source_error_dynamic',False),error_payload_sites=getattr(traced,'source_error_payload_sites',()),
                  result_count=len(traced.outputs)-len(groups)-bool(getattr(traced,'source_error_specs',()))-bool(getattr(traced,'source_error_dynamic',False))-len(getattr(traced,'source_error_payload_sites',())),outputs=[types[ssa] for ssa in traced.outputs])
    from .source_exception_heap import pack_exception_table
    contract['exception_heap']=pack_exception_table(contract.pop('error_table'))
    attributes+=', tessera.source_state = '+json.dumps(json.dumps(contract,sort_keys=True,separators=(',',':')))
    lines = ['module attributes {'+attributes+'} {', f'  func.func @{name}({args}) -> {signature}{function_attrs} {{', *emit(traced.body, '    '),
             '    return '+', '.join('%'+ssa for ssa in traced.outputs)+' : '+outputs, '  }', '}']
    return '\n'.join(lines)+'\n'


def to_native_autodiff_ir(module):
    """Project an owned Graph module's nested trace regions into native SCF.

    Preserve argument, function and module contracts while reusing the source
    serializer. No source re-execution and no derivative recipes live here.
    """
    from .trace import TracedFunction
    if len(module.functions) != 1:
        raise SourceControlFlowError('native region AD requires one source function')
    fn = module.functions[0]
    if (module.module_attrs.get('tessera.frontend.authority') or
            fn.fn_attrs.get('tessera.frontend.authority')) != '"tracer"':
        raise SourceControlFlowError('native region AD requires tracer-owned source')
    verification = module.verify()
    if not verification.ok:
        raise SourceControlFlowError(verification.format())
    args = []
    for arg in fn.args:
        shape = arg.ir_type.shape
        if shape is None or any(not str(d).isdigit() for d in shape):
            raise SourceControlFlowError('native region AD requires static tensor arguments')
        args.append((arg.name.lstrip('%'), tuple(int(d) for d in shape), arg.ir_type.dtype))
    traced = TracedFunction(args, fn.body, [v.lstrip('%') for v in fn.return_values])
    text = to_native_source_ir(traced, name=fn.name)
    lines = text.splitlines()
    attrs = {k: v for k, v in module._emitted_module_attrs().items()
             if k != 'tessera.frontend.authority'}
    if attrs:
        lines[0] = lines[0][:-3] + ', ' + ', '.join(f'{k} = {v}' for k, v in attrs.items()) + '} {'
    # The serializer owns regions; the original Graph function owns the ABI.
    results = ', '.join(str(t) for t in fn.result_types)
    fn_attrs = ' attributes {' + ', '.join(f'{k} = {v}' for k, v in fn.fn_attrs.items()) + '}' if fn.fn_attrs else ''
    lines[1] = f'  func.func @{fn.name}(' + ', '.join(a.to_mlir() for a in fn.args) + ') -> (' + results + ')' + fn_attrs + ' {'
    return '\n'.join(lines) + '\n'
