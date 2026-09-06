"""JIT-owned isolated attention JVP with an automatically bound resident frame."""
from dataclasses import dataclass
import inspect
import re
from .native_attention_jvp import materialize_generated
from .native_device_tape import _Buffer
from .native_storage_contract import generate_tensor_binding
from .nvidia_native import package_generated_attention_checkpoint_pair, _checkpoint_identity, AttentionCheckpointPair
from .native_gpu_storage import NativeGPUStoragePackage


@dataclass(frozen=True)
class NativeAttentionJVPProgram:
    pair: AttentionCheckpointPair
    tangent: NativeGPUStoragePackage
    active: tuple[int,...]

    def capture(self,q,k,v):
        frame=self.pair.capture(q,k,v)
        try:
            self.tangent.validate()
            expected=_checkpoint_identity(frame.dims,frame._scale,frame._causal)
            if f'tessera.attention_checkpoint_identity = "{expected}"' not in self.tangent.arena_ir:
                raise ValueError('automatic attention program forward/tangent generations disagree')
            names=('q','k','v','primal','lse','dq','dk','dv','tangent','scratch')
            signature=inspect.Signature([inspect.Parameter(n,inspect.Parameter.POSITIONAL_ONLY) for n in names])
            frame._jvp_binding=generate_tensor_binding(self.tangent,signature)
            zeros={i:_Buffer(frame,frame.shapes[i]) for i in range(3) if i not in self.active}
            # Inactive loads were removed by native product lowering. Distinct
            # allocated placeholders preserve the no-alias argument ABI.
            return AutomaticAttentionFrame(frame,self.active,zeros)
        except BaseException:
            frame.close()
            raise


class AutomaticAttentionFrame:
    def __init__(self,frame,active,zeros):
        self._frame,self._active,self._zeros=frame,active,zeros

    @property
    def primal(self):
        return self._frame.primal

    def jvp(self,*tangents):
        if len(tangents)!=len(self._active):
            raise ValueError('automatic attention tangent arity disagrees with wrt')
        values=dict(self._zeros)
        values.update(zip(self._active,tangents,strict=True))
        return self._frame.jvp(*(values[i] for i in range(3)))

    def close(self):
        self._frame.close()

    def __enter__(self):
        self._frame._ready()
        return self

    def __exit__(self,*exc):
        self.close()


def compile_attention_program(source,active,*,compiler,llvm_bin):
    from pathlib import Path
    from .scheduled_matmul import find_tessera_opt
    selected=find_tessera_opt()
    if selected is None or Path(selected).resolve()!=Path(compiler).resolve():
        raise ValueError('attention checkpoint and JVP must use the same TESSERA_OPT compiler')
    active=tuple(active)
    if (not active or len(set(active))!=len(active) or
            any(type(i) is not int or i not in range(3) for i in active) or
            not any(i in (0,1) for i in active)):
        raise ValueError('automatic attention program requires an active Q or K input')
    # Switch the request only; both products still come from native AD over the
    # same traced body. No GraphIRModule is reconstructed from the derivative.
    reverse,count=re.subn(r'tessera.autodiff = "forward"','tessera.autodiff = "reverse"',source)
    if count not in (1, 2):
        raise ValueError('automatic attention program requires a forward request on its function and/or module')
    pair=package_generated_attention_checkpoint_pair(reverse,pipeline_name='tessera-nvidia-pipeline-sm120')
    from .resident_attention import checkpoint_shapes
    dims,_=checkpoint_shapes(pair)
    policy=pair.forward.descriptor.provenance
    tangent=materialize_generated(source,dims,policy['scale'],policy['causal'],compiler=compiler,llvm_bin=llvm_bin)
    return NativeAttentionJVPProgram(pair,tangent,active)
