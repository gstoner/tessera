"""Resident native row residuals reuse checked capture ownership and streams."""
from __future__ import annotations
import ctypes as ct
import math

from .resident_attention import ResidentAttentionTape
from .native_attention_rows import validate_program


class ResidentAttentionRowsTape(ResidentAttentionTape):
    def __init__(self, program, *args, **kwargs):
        self._row_program=program
        super().__init__(program.pair,*args,**kwargs)
        policy=program.pair.backward.descriptor.provenance["row_contract"]
        self._backward_threads=policy["block_threads"]
        self._gradient_launch=policy["gradient_launch"]
        self._backward_scalars=self.dims

    def _checkpoint_shapes(self,pair,*,runtime_dims=None):
        return validate_program(self._row_program,runtime_dims=runtime_dims)

    def _packages(self,pair):
        correction=self._row_program.correction
        return (pair.forward,pair.backward)+((correction,) if correction is not None else ())

    def _execute(self,backward,buffers):
        if not backward:
            return super()._execute(False,buffers)
        roots=[buffers[0],*buffers[1:4]]
        if self._has_bias:roots.append(buffers[5])
        roots.append(buffers[5+int(self._has_bias)])
        seed_index=6+int(self._has_bias)
        output_index=seed_index+int(self._has_lse_cotangent)
        if self._row_program.correction is None:
            # Value-only native code never reads the row operand. Reuse the
            # existing valid row-shaped LSE allocation without a producer.
            row=self._saved[4]
        else:
            row=self._allocate(self.shapes[4])
            producer=[*roots,*([buffers[seed_index]] if self._has_lse_cotangent else []),row]
            arguments=[ct.c_void_p(value.pointer.value) for value in producer]+[ct.c_int64(x) for x in self.dims]
            pointers=(ct.c_void_p*len(arguments))(*(ct.cast(ct.pointer(x),ct.c_void_p) for x in arguments))
            count=math.prod(self.shapes[4])
            self.check(self._launch(self._functions[2],(count+31)//32,1,1,32,1,1,0,self._stream,pointers,None))
        super()._execute(True,[*roots,row,*buffers[output_index:]])
        if self._row_program.correction is not None:
            if self.asynchronous:self.check(self._free_async(row.pointer,self._stream))
            else:self.check(self.free(row.pointer))
            row.pointer=ct.c_void_p()
            self.buffers.remove(row)
