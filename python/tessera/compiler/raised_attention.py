"""Executable target-specific binding of a native attention recipe bucket.

No GraphIRModule reconstruction and no automatic arbiter promotion. The native
parent is replayed before packaging; ABI and policy are read from its Schedule.
"""
from dataclasses import dataclass
from typing import Any
import json
import re
from pathlib import Path
from .parametric_recipe import ParametricRecipe, BucketInstance
from .scheduled_attention import ScheduledAttentionArtifact
from .scheduled_matmul import find_tessera_opt, run_tessera_opt


def lower_attention_bucket(recipe: ParametricRecipe, instance: BucketInstance, *, compiler, target="nvidia_sm120"):
    architectures = {'nvidia_sm120':'sm_120','x86':'zen5-avx512','apple_gpu':'apple7'}
    if target not in architectures:
        raise ValueError('raised f32 attention has no admitted backend for '+str(target))
    compiler = Path(compiler)
    replay = recipe.instantiate_buckets([dict(instance.bindings)],tessera_opt=str(compiler))
    if replay != (instance,):
        raise ValueError('attention bucket disagrees with native recipe replay')
    if any(key in instance.mlir for key in ('tessera.target','tessera.arch','tessera.launch_bindings')):
        raise ValueError('attention bucket already contains target policy')
    argument_header = re.search(r'func.func @\w+\(([^\n]*)\) ->', instance.mlir)
    if argument_header is None:
        raise ValueError('attention bucket function header is missing')
    has_bias = len(re.findall(r'tensor<[^>]+>', argument_header.group(1))) == 4
    launch_names = ['q','k','v'] + (['bias'] if has_bias else []) + ['out']
    graph = instance.mlir.replace('module attributes {',
        'module attributes {tessera.target = '+json.dumps(target)+', tessera.arch = '+json.dumps(architectures[target])+', tessera.launch_bindings = '+json.dumps(launch_names)+', ',1)
    if graph == instance.mlir:
        raise ValueError('attention recipe module header is missing')
    schedule = run_tessera_opt(compiler,graph,'--tessera-graph-to-schedule')
    attrs_list = re.findall(r' = schedule.attention [^{]+\{([^{}]+)\}',schedule)
    headers = re.findall(r'func.func @\w+\(([^\n]*)\) ->',schedule)
    if len(attrs_list)!=1 or len(headers)!=1:
        raise ValueError('attention binding requires one native attention function')
    attrs = attrs_list[0]
    def value(name):
        match = re.search(r'(?:^|, )'+name+r' = ("[^"]*"|true|false|[-+0-9.eE]+)(?: : (i64|f32))?(?:,|$)',attrs)
        if match is None:
            raise ValueError('missing native attention field '+name)
        raw,ty = match.groups()
        return int(raw) if ty=='i64' else float(raw) if ty=='f32' else json.loads(raw)
    types = re.findall(r'tensor<(\d+)x(\d+)x(\d+)x(\d+)xf32>',headers[0])
    if len(types)!=(4 if has_bias else 3) or value('bias') != has_bias or value('storage')!='f32':
        raise ValueError('raised attention requires three dense f32 inputs')
    q,k,v = [tuple(map(int,t)) for t in types[:3]]
    dims = (q[0],q[1],k[1],q[2],k[2],q[3],v[3])
    tile = run_tessera_opt(compiler,schedule,'--tessera-schedule-to-tile')
    semantic = run_tessera_opt(compiler,graph,
        f'--tessera-tile-ir-lowering=tile-q={value("tile_q")} tile-kv={value("tile_kv")} sm=90')
    digest = value('artifact_hash')
    artifact = ScheduledAttentionArtifact(graph_ir=graph,schedule_ir=schedule,semantic_ir=semantic,tile_ir=tile,
        target=target,architecture=architectures[target],function_name=(f'tessera_tile_attention_f32_{"causal" if value("causal") else "full"}_{digest[:10]}' if target == 'nvidia_sm120' else re.findall(r'func.func @(\w+)',schedule)[0]),
        q_name='q',k_name='k',v_name='v',bias_name='bias' if has_bias else None,output_name='out',dtype='fp32',storage='f32',accum=value('accum'),dims=dims,
        scale=value('scale'),causal=value('causal'),window_left=value('window_left'),window_right=value('window_right'),
        softcap=value('softcap'),dropout_p=value('dropout_p'),dropout_seed=value('dropout_seed'),
        tile_q=value('tile_q'),tile_kv=value('tile_kv'),workgroup_size=value('workgroup_size'),recurrence=value('recurrence'),
        backward_lse_policy=value('backward_lse_policy'),backward_lse_selection=value('backward_lse_selection'),schedule_digest=digest)
    artifact.validate()
    return artifact


def validate_mask_rows(artifact, bias):
    """Refuse empty rows after composing additive, causal and window masks.

    Negative infinity is the existing additive-mask representation in native
    attention. This gate does not convert Boolean or broadcast inputs in Python.
    """
    import numpy as np
    _, _, _, sq, sk, _, _ = artifact.dims
    if not np.any(np.isneginf(bias)):
        return
    query = np.arange(sq)[:, None] + max(sk - sq, 0)
    key = np.arange(sk)[None, :]
    valid = np.ones((sq, sk), dtype=bool)
    if artifact.causal:
        valid &= key <= query
    if artifact.window_left >= 0:
        valid &= key >= query - artifact.window_left
    if artifact.window_right >= 0:
        valid &= key <= query + artifact.window_right
    if not np.all(np.any(np.isfinite(bias) & valid, axis=-1)):
        raise ValueError('raised attention mask produces a fully masked row')


@dataclass(frozen=True)
class RaisedAttentionBinding:
    artifact: ScheduledAttentionArtifact
    package: Any
    recipe_digest: str
    bucket_digest: str

    def __call__(self,q,k,v,bias=None):
        import numpy as np
        from tessera import runtime as rt
        b,h,_,sq,_,_,dv = self.artifact.dims
        out = np.empty((b,h,sq,dv),dtype=np.float32)
        artifact = rt.RuntimeArtifact(metadata={'target':self.artifact.target},native_image=self.package.image,
            launch_descriptor=self.package.descriptor,tile_ir=self.package.tile_ir,target_ir=self.package.target_ir)
        arguments = dict(q=q,k=k,v=v,out=out)
        if self.artifact.bias_name is not None:
            expected = (b,h,sq,self.artifact.dims[4])
            if not isinstance(bias,np.ndarray) or bias.dtype != np.float32 or bias.shape != expected or np.any(np.isnan(bias)) or np.any(np.isposinf(bias)):
                raise ValueError('raised attention bias requires full-shape fp32 finite or negative-infinity data')
            if np.any(np.isneginf(bias)) and self.artifact.target != "nvidia_sm120":
                raise ValueError("negative-infinity attention masks require NVIDIA device proof")
            validate_mask_rows(self.artifact, bias)
            arguments['bias'] = bias
        elif bias is not None:
            raise ValueError('this attention artifact has no bias operand')
        arguments.update(zip(('B','Hq','Hkv','Sq','Sk','D','Dv'),self.artifact.dims,strict=True))
        result = rt.launch(artifact,arguments)
        expected_kind = 'native_cpu' if self.artifact.target == 'x86' else 'native_gpu'
        if not result.get('ok') or result.get('execution_kind')!=expected_kind:
            raise RuntimeError('raised attention native execution failed: '+str(result))
        return out


def bind_attention_bucket(recipe, instance, *, compiler, target="nvidia_sm120"):
    selected = find_tessera_opt()
    if selected is None or selected.resolve()!=Path(compiler).resolve():
        raise ValueError('attention packaging must use the recipe compiler')
    artifact = lower_attention_bucket(recipe,instance,compiler=compiler,target=target)
    package: Any
    if target == 'nvidia_sm120':
        from .nvidia_native import package_scheduled_attention as nvidia_package
        package = nvidia_package(artifact,pipeline_name='tessera-nvidia-pipeline-sm120')
    elif target == 'x86':
        from .x86_native import package_scheduled_attention as x86_package
        package = x86_package(artifact,pipeline_name='tessera-lower-to-x86')
    else:
        from .apple_native import package_scheduled_attention as apple_package
        package = apple_package(artifact,pipeline_name='tessera-lower-to-apple_gpu')
    return RaisedAttentionBinding(artifact,package,recipe.digest,instance.digest)
