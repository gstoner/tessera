"""Replay-bound SSD VJP using chunk checkpoints and scalar recomputation.

This host JIT entry differentiates all three forward results. It does not
register an automatic public mixer rule or establish GPU backward execution.
"""
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any
from .scheduled_matmul import run_tessera_opt
from .scheduled_ssd import ScheduledSSD, lower_scheduled_ssd


def _emit(parent, compiler):
    signature = re.search(r'func.func @ssd\((.*?)\) -> \((.*?)\)',parent.schedule_ir,re.S)
    if signature is None:
        raise ValueError('SSD adjoint requires its canonical entry')
    inputs = re.findall(r'tensor<[^>]+>',signature[1])
    outputs = re.findall(r'tensor<[^>]+>',signature[2])
    if len(inputs)!=5 or len(outputs)!=3:
        raise ValueError('SSD adjoint entry roles disagree')
    shapes = [tuple(map(int,re.findall(r'\d+',t))) for t in inputs]
    # Strip the element spelling's 32 from the native tensor dimensions.
    shapes = [s[:-1] for s in shapes]
    T,H,P = shapes[0]
    N = shapes[2][2]
    matched_chunk = re.search(r'chunk_size = (\d+)',parent.schedule_ir)
    if matched_chunk is None:
        raise ValueError('SSD checkpoint policy is missing')
    chunk = int(matched_chunk[1])
    canonical = lower_scheduled_ssd(T,H,N,P,chunk,compiler=compiler)
    if canonical.schedule_ir != parent.schedule_ir:
        raise ValueError('SSD adjoint requires the canonical forward recurrence')
    names = ('x','decay','b','c','initial','saved','dy','df','dcps')
    types = inputs+[outputs[2],outputs[0],outputs[1],outputs[2]]
    lines = ['module {', 'func.func @ssd_vjp('+', '.join('%'+n+': '+t for n,t in zip(names,types,strict=True))+') -> ('+', '.join(inputs)+') {']
    for n,v in [('z',0),('one',1),('T',T),('H',H),('P',P),('N',N),('chunk',chunk)]:
        lines.append(f'%{n} = arith.constant {v} : index')
    lines.append('%fz = arith.constant 0.0 : f32')
    for i,t in enumerate(inputs):
        lines += [f'%empty{i} = tensor.empty() : {t}',f'%g{i} = linalg.fill ins(%fz : f32) outs(%empty{i} : {t}) -> {t}']
    alltypes = ', '.join(inputs)
    def loop(prefix,iv,limit,initial):
        lines.append(f'%{prefix}:5 = scf.for %{iv} = %z to %{limit} step %one iter_args('+', '.join(f'%{prefix}g{i} = {initial[i]}' for i in range(5))+f') -> ({alltypes}) {{')
    loop('hloop','h','H',[f'%g{i}' for i in range(5)])
    loop('nloop','n','N',[f'%hloopg{i}' for i in range(5)])
    loop('ploop','p','P',[f'%nloopg{i}' for i in range(5)])
    lines += [f'%seed = tensor.extract %df[%h, %n, %p] : {inputs[4]}',
        '%reverse:6 = scf.for %r = %z to %T step %one iter_args('+', '.join(f'%rg{i} = %ploopg{i}' for i in range(5))+f', %ds = %seed) -> ({alltypes}, f32) {{',
        '%tr = arith.subi %T, %r : index','%t = arith.subi %tr, %one : index',
        '%ci = arith.divui %t, %chunk : index','%begin = arith.muli %ci, %chunk : index',
        '%first = arith.cmpi eq, %ci, %z : index',
        '%base = scf.if %first -> f32 {',
        f'%init = tensor.extract %initial[%h, %n, %p] : {inputs[4]}','scf.yield %init : f32',
        '} else {','%prior = arith.subi %ci, %one : index',
        f'%cp = tensor.extract %saved[%prior, %h, %n, %p] : {outputs[2]}','scf.yield %cp : f32','}',
        '%old = scf.for %j = %begin to %t step %one iter_args(%state = %base) -> f32 {',
        f'%jd = tensor.extract %decay[%j, %h] : {inputs[1]}',
        f'%jb = tensor.extract %b[%j, %h, %n] : {inputs[2]}',
        f'%jx = tensor.extract %x[%j, %h, %p] : {inputs[0]}',
        '%carried = arith.mulf %jd, %state : f32','%added = arith.mulf %jb, %jx : f32',
        '%next = arith.addf %carried, %added : f32','scf.yield %next : f32','}',
        f'%d = tensor.extract %decay[%t, %h] : {inputs[1]}',
        f'%bv = tensor.extract %b[%t, %h, %n] : {inputs[2]}',
        f'%cv = tensor.extract %c[%t, %h, %n] : {inputs[3]}',
        f'%xv = tensor.extract %x[%t, %h, %p] : {inputs[0]}',
        f'%dyv = tensor.extract %dy[%t, %h, %p] : {inputs[0]}',
        '%a = arith.mulf %d, %old : f32','%bx = arith.mulf %bv, %xv : f32','%current = arith.addf %a, %bx : f32',
        '%end = arith.addi %t, %one : index','%rem = arith.remui %end, %chunk : index',
        '%boundary = arith.cmpi eq, %rem, %z : index','%last = arith.cmpi eq, %end, %T : index',
        '%checkpoint = arith.ori %boundary, %last : i1','%cpgrad = scf.if %checkpoint -> f32 {',
        f'%cpv = tensor.extract %dcps[%ci, %h, %n, %p] : {outputs[2]}','scf.yield %cpv : f32',
        '} else {','scf.yield %fz : f32','}',
        '%fromy = arith.mulf %dyv, %cv : f32','%withy = arith.addf %ds, %fromy : f32',
        '%total = arith.addf %withy, %cpgrad : f32',
        '%gx = arith.mulf %total, %bv : f32','%gd = arith.mulf %total, %old : f32',
        '%gb = arith.mulf %total, %xv : f32','%gc = arith.mulf %dyv, %current : f32',
        '%previous = arith.mulf %total, %d : f32']
    for i,(val,indices) in enumerate([('gx','%t, %h, %p'),('gd','%t, %h'),('gb','%t, %h, %n'),('gc','%t, %h, %n')]):
        lines += [f'%was{i} = tensor.extract %rg{i}[{indices}] : {inputs[i]}',
                  f'%sum{i} = arith.addf %was{i}, %{val} : f32',
                  f'%updated{i} = tensor.insert %sum{i} into %rg{i}[{indices}] : {inputs[i]}']
    lines += ['scf.yield '+', '.join(f'%updated{i}' for i in range(4))+f', %rg4, %previous : {alltypes}, f32','}',
              f'%di = tensor.insert %reverse#5 into %reverse#4[%h, %n, %p] : {inputs[4]}',
              'scf.yield '+', '.join(f'%reverse#{i}' for i in range(4))+f', %di : {alltypes}','}',
              'scf.yield '+', '.join(f'%ploop#{i}' for i in range(5))+f' : {alltypes}','}',
              'scf.yield '+', '.join(f'%nloop#{i}' for i in range(5))+f' : {alltypes}','}',
              'return '+', '.join(f'%hloop#{i}' for i in range(5))+f' : {alltypes}','}','}']
    return '\n'.join(lines)


@dataclass(frozen=True)
class SSDCheckpointVJP:
    parent: ScheduledSSD
    lowered_ir: str

    def validate(self, compiler):
        self.parent.validate(compiler)
        if run_tessera_opt(Path(compiler),_emit(self.parent,compiler),'--canonicalize') != self.lowered_ir:
            raise ValueError('SSD checkpoint adjoint disagrees with parent replay')


def lower_checkpoint_vjp(parent, *, compiler):
    parent.validate(compiler)
    return SSDCheckpointVJP(parent,run_tessera_opt(Path(compiler),_emit(parent,compiler),'--canonicalize'))


class SSDCheckpointProgram:
    """Synchronous native CPU forward/VJP with privately owned checkpoints."""
    def __init__(self, parent, *, compiler):
        import threading
        from tessera import _jit_boundary as jit
        self._lock = threading.RLock()
        self._closed = False
        self.parent = parent
        self.adjoint = lower_checkpoint_vjp(parent,compiler=compiler)
        self.adjoint.validate(compiler)
        self._forward = jit.compile_module(parent.lowered_ir)
        try:
            self._backward = jit.compile_module(self.adjoint.lowered_ir)
        except BaseException:
            jit.destroy(self._forward)
            self._closed = True
            raise
        signature = re.search(r'func.func @ssd\((.*?)\) -> \((.*?)\)',parent.schedule_ir,re.S)
        if signature is None:
            raise ValueError('SSD canonical signature is missing')
        self.input_shapes,self.output_shapes = [
            [tuple(map(int,re.findall(r'\d+',t)[:-1])) for t in re.findall(r'tensor<[^>]+>',signature[i])]
            for i in (1,2)]

    def vjp(self, inputs, seeds):
        import numpy as np
        from tessera import _jit_boundary as jit
        with self._lock:
            if self._closed:
                raise ValueError('SSD checkpoint program is closed')
            def snapshots(values,shapes):
                if len(values) != len(shapes) or any(not isinstance(v,np.ndarray) or v.dtype != np.float32 or v.shape != s for v,s in zip(values,shapes,strict=True)):
                    raise ValueError('SSD checkpoint input/seed ABI disagrees')
                return [np.array(v,copy=True,order='C') for v in values]
            values = snapshots(inputs,self.input_shapes)
            cotangents = snapshots(seeds,self.output_shapes)
            results = [np.empty(s,np.float32) for s in self.output_shapes]
            grads = [np.empty(s,np.float32) for s in self.input_shapes]
            jit.invoke(self._forward,'ssd',values,results)
            jit.invoke(self._backward,'ssd_vjp',values+[results[2]]+cotangents,grads)
            return results,grads

    def __call__(self, *inputs):
        """Expose Y to the host autodiff tape with a native checkpoint rule.

        The returned tape entry owns private primal/checkpoint snapshots. Keep
        this program open until backward completes; higher-order AD is unsupported.
        """
        import numpy as np
        from tessera import _jit_boundary as jit
        from tessera.autodiff.tape import record_custom_vjp_call
        saved: dict[str, Any] = {}
        def forward(*values):
            with self._lock:
                if self._closed:
                    raise ValueError('SSD checkpoint program is closed')
                if len(values) != 5 or any(not isinstance(v,np.ndarray) or v.dtype != np.float32 or v.shape != shape for v,shape in zip(values,self.input_shapes,strict=True)):
                    raise ValueError('SSD checkpoint input ABI disagrees')
                copied = [np.array(v,copy=True,order='C') for v in values]
                results = [np.empty(shape,np.float32) for shape in self.output_shapes]
                jit.invoke(self._forward,'ssd',copied,results)
                saved['values'],saved['checkpoints'] = copied,results[2]
                return results[0]
        def backward(dout,*unused):
            with self._lock:
                if self._closed:
                    raise ValueError('SSD checkpoint program is closed before backward')
                if not isinstance(dout,np.ndarray) or dout.shape != self.output_shapes[0] or dout.dtype != np.float32:
                    raise ValueError('SSD checkpoint cotangent ABI disagrees')
                values = saved['values']
                seeds = [np.array(dout,copy=True),np.zeros(self.output_shapes[1],np.float32),np.zeros(self.output_shapes[2],np.float32)]
                grads = [np.empty(shape,np.float32) for shape in self.input_shapes]
                jit.invoke(self._backward,'ssd_vjp',values+[saved['checkpoints']]+seeds,grads)
                return tuple(grads)
        return record_custom_vjp_call('native_ssd_checkpoint',forward,backward,*inputs)

    def close(self):
        from tessera import _jit_boundary as jit
        with self._lock:
            if not self._closed:
                self._closed = True
                jit.destroy(self._backward)
                jit.destroy(self._forward)

    def __enter__(self):
        if self._closed:
            raise ValueError('SSD checkpoint program is closed')
        return self

    def __exit__(self,*exc):
        self.close()
