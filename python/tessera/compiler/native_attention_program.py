"""JIT-owned isolated attention JVP with an automatically bound resident frame."""
from dataclasses import dataclass
import inspect
import re
from .native_attention_jvp import materialize_generated
from .native_device_tape import _Buffer
from .native_storage_contract import generate_tensor_binding
from .nvidia_native import package_generated_attention_checkpoint_pair, _checkpoint_identity, AttentionCheckpointPair, AttentionForwardCheckpoint
from .native_gpu_storage import NativeGPUStoragePackage


def _policy_bool(policy, key: str) -> bool:
    value = policy.get(key, False)
    if type(value) is not bool:
        raise ValueError("native attention boolean policy disagrees: " + key)
    return value


def _policy_indices(policy, key: str) -> tuple[int, ...]:
    value = policy.get(key, ())
    if not isinstance(value, (list, tuple)) or any(type(i) is not int for i in value):
        raise ValueError("native attention integer sequence policy disagrees: " + key)
    return tuple(value)


@dataclass(frozen=True)
class NativeAttentionJVPProgram:
    pair: AttentionCheckpointPair | AttentionForwardCheckpoint
    tangent: NativeGPUStoragePackage
    active: tuple[int,...]
    input_indices: tuple[int,...] = ()
    input_names: tuple[str,...] = ()

    def validate(self):
        from .native_attention_jvp_artifact import payload
        payload(self)

    @property
    def program_digest(self):
        from .native_attention_jvp_artifact import digest
        return digest(self)

    def to_json(self):
        from .native_attention_jvp_artifact import to_json
        return to_json(self)

    @classmethod
    def from_json(cls,text,*,expected_digest):
        from .native_attention_jvp_artifact import from_json
        return from_json(text,expected_digest=expected_digest)

    def capture(self,*args,**kwargs):
        mapping=self.input_indices or tuple(range(
            3+int(_policy_bool(self.pair.forward.descriptor.provenance, "bias"))))
        count=len(mapping)
        if count not in (3,4) or any(type(i) is not int for i in mapping) or sorted(mapping)!=list(range(count)):
            raise ValueError("attention JVP capture requires a native frontend permutation")
        self.validate()
        biased=count==4
        names=self.input_names or (("q","k","v","bias") if biased else ("q","k","v"))
        signature=inspect.Signature([inspect.Parameter(n,inspect.Parameter.POSITIONAL_OR_KEYWORD) for n in names])
        bound=signature.bind(*args,**kwargs)
        inputs=tuple(bound.arguments[n] for n in names)
        roles=tuple(inputs[i] for i in mapping)
        frame=self.pair.capture(*roles[:3],**({"bias":roles[3]} if biased else {}))
        try:
            self.tangent.validate()
            expected=_checkpoint_identity(frame.dims,frame._scale,frame._causal,
                bias=biased,bias_shape=frame._bias_shape if biased else ())
            if f'tessera.attention_checkpoint_identity = "{expected}"' not in self.tangent.arena_ir:
                raise ValueError('automatic attention program forward/tangent generations disagree')
            tensor_names=('q','k','v','primal','lse','dq','dk','dv') + (
                ('bias','dbias') if biased else ()) + ('tangent','scratch')
            signature=inspect.Signature([inspect.Parameter(n,inspect.Parameter.POSITIONAL_ONLY) for n in tensor_names])
            frame._jvp_binding=generate_tensor_binding(self.tangent,signature)
            shapes=(*frame.shapes[:3], *((frame._bias_shape,) if biased else ()))
            zeros={i:_Buffer(frame,shapes[i]) for i in range(count) if i not in self.active}
            # Inactive loads are removed by native lowering. Distinct allocated
            # placeholders preserve the no-alias argument ABI.
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
        extra={"dbias":values[3]} if 3 in values else {}
        return self._frame.jvp(*(values[i] for i in range(3)),**extra)

    def close(self):
        self._frame.close()

    def __enter__(self):
        self._frame._ready()
        return self

    def __exit__(self,*exc):
        self.close()


def compile_attention_program(source,active,*,compiler,llvm_bin,input_names=(),retain_reverse=False):
    from pathlib import Path
    from .scheduled_matmul import find_tessera_opt
    selected=find_tessera_opt()
    if selected is None or Path(selected).resolve()!=Path(compiler).resolve():
        raise ValueError('attention checkpoint and JVP must use the same TESSERA_OPT compiler')
    active=tuple(active)
    if (not active or len(set(active))!=len(active) or
            any(type(i) is not int or i not in range(4) for i in active)):
        raise ValueError('automatic attention program requires an active primal input')
    # Switch the request only; both products still come from native AD over the
    # same traced body. No GraphIRModule is reconstructed from the derivative.
    reverse,count=re.subn(r'tessera.autodiff = "forward"','tessera.autodiff = "reverse"',source)
    if count not in (1, 2):
        raise ValueError('automatic attention program requires a forward request on its function and/or module')
    if type(retain_reverse) is not bool:
        raise ValueError("attention reverse retention must be boolean")
    if retain_reverse:
        pair=package_generated_attention_checkpoint_pair(reverse,pipeline_name='tessera-nvidia-pipeline-sm120')
    else:
        from .nvidia_native import package_generated_attention_forward_checkpoint
        pair=package_generated_attention_forward_checkpoint(reverse,pipeline_name='tessera-nvidia-pipeline-sm120')
    from .resident_attention import checkpoint_shapes
    dims,_=checkpoint_shapes(pair)
    policy=pair.forward.descriptor.provenance
    biased=_policy_bool(policy, "bias")
    count=3+int(biased)
    mapping=_policy_indices(policy, "frontend_argument_indices")
    if len(mapping)!=count or any(type(i) is not int for i in mapping) or sorted(mapping)!=list(range(count)) or any(i>=count for i in active):
        raise ValueError("native attention JVP frontend argument mapping disagrees")
    physical_active=tuple(mapping.index(i) for i in active)
    bias_shape=_policy_indices(policy, "bias_shape") or ((dims[0],dims[1],dims[3],dims[4]) if biased else ())
    tangent=materialize_generated(source,dims,policy['scale'],policy['causal'],
        compiler=compiler,llvm_bin=llvm_bin,bias_shape=bias_shape)
    # Both independently generated native products must agree on activity after
    # projecting frontend indices into physical Q/K/V roles.
    contract=re.findall(r'tessera.attention_jvp_contract = \{([^\n]*?)\}',tangent.arena_ir)
    expected="active = ["+", ".join(str(i in physical_active).lower() for i in range(count))+"]"
    if len(contract)!=1 or expected not in contract[0]:
        raise ValueError("native attention JVP tangent activity disagrees with frontend request")
    return NativeAttentionJVPProgram(pair,tangent,physical_active,mapping,tuple(input_names))


@dataclass(frozen=True)
class NativeAttentionVJPProgram:
    """Compiler-generated reverse attention with requested result ordering."""
    pair: AttentionCheckpointPair
    active: tuple[int, ...]
    input_indices: tuple[int, ...] = ()

    def validate(self):
        from .native_attention_vjp_artifact import payload
        payload(self)

    @property
    def program_digest(self):
        import hashlib
        from .native_attention_vjp_artifact import canonical,payload
        return hashlib.sha256(canonical(payload(self)).encode()).hexdigest()

    def to_json(self):
        from .native_attention_vjp_artifact import to_json
        return to_json(self)

    @classmethod
    def from_json(cls,text,*,expected_digest):
        from .native_attention_vjp_artifact import from_json
        return from_json(text,expected_digest=expected_digest)

    def capture(self, *args, bias=None, asynchronous=False):
        mapping = self.input_indices or tuple(range(len(args) + int(bias is not None)))
        if bias is not None:
            if len(mapping) != 4 or len(args) != 3:
                raise ValueError("attention capture bias keyword disagrees with frontend arity")
            index = mapping[3]
            args = (*args[:index], bias, *args[index:])
        if len(args) != len(mapping):
            raise ValueError("attention capture arity disagrees with frontend argument mapping")
        roles = tuple(args[index] for index in mapping)
        if len(roles) not in (3, 4):
            raise ValueError("attention capture requires Q/K/V and optional bias")
        return AutomaticAttentionVJPFrame(
            self.pair.capture(*roles[:3], bias=roles[3] if len(roles)==4 else None, asynchronous=asynchronous), self.active)


class AutomaticAttentionVJPFrame:
    def __init__(self, frame, active):
        self._frame, self._active = frame, active

    @property
    def primal(self):
        return self._frame.primal

    def backward(self, cotangent):
        gradients = self._frame.backward(cotangent)
        roles = getattr(self._frame, "_gradient_roles", tuple(range(len(gradients))))
        return tuple(gradients[roles.index(i)] for i in self._active)

    def wait_on(self, stream):
        self._frame.wait_on(stream)

    def synchronize(self):
        self._frame.synchronize()

    def close(self):
        self._frame.close()

    def __enter__(self):
        self._frame._ready()
        return self

    def __exit__(self, *exc):
        self.close()


def compile_attention_vjp_program(source, active, *, compiler, compact_gradients=False, compact_launch="packed_v1", compact_threads=128):
    from pathlib import Path
    from .scheduled_matmul import find_tessera_opt
    selected = find_tessera_opt()
    if selected is None or Path(selected).resolve() != Path(compiler).resolve():
        raise ValueError("attention reverse program requires the selected TESSERA_OPT compiler")
    active = tuple(active)
    if (not active or len(set(active)) != len(active) or
            any(type(i) is not int or i not in range(4) for i in active)):
        raise ValueError("automatic attention reverse program requires unique Q/K/V/bias inputs")
    if 'tessera.autodiff = "reverse"' not in source:
        raise ValueError("automatic attention reverse program requires a reverse request")
    if type(compact_gradients) is not bool:
        raise ValueError("compact gradient output selection must be boolean")
    if not compact_gradients and (compact_launch != "packed_v1" or compact_threads != 128):
        raise ValueError("compact launch requires compact gradient outputs")
    pair = package_generated_attention_checkpoint_pair(
        source, pipeline_name="tessera-nvidia-pipeline-sm120", prune_inactive=True,
        compact_gradients=compact_gradients, compact_launch=compact_launch, compact_threads=compact_threads)
    mapping = _policy_indices(pair.backward.descriptor.provenance, "frontend_argument_indices")
    if (any(type(i) is not int for i in mapping) or
            sorted(mapping) != list(range(3 + int(_policy_bool(pair.backward.descriptor.provenance, "bias")))) or
            any(i not in mapping for i in active)):
        raise ValueError("native checkpoint frontend argument mapping disagrees")
    physical_active = tuple(mapping.index(index) for index in active)
    activity = _policy_indices(pair.backward.descriptor.provenance, "gradient_activity")
    expected = tuple(int(i in physical_active) for i in range(
        3 + int(_policy_bool(pair.backward.descriptor.provenance, "bias_gradient"))))
    if activity != expected:
        raise ValueError("native checkpoint gradient activity disagrees with requested frontend results")
    if 3 in physical_active and not pair.backward.descriptor.provenance.get("bias_gradient", False):
        raise ValueError("requested bias gradient is absent from the native product")
    return NativeAttentionVJPProgram(pair, physical_active, mapping)
