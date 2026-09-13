"""Shared native attention descriptor projection and target ancestry."""
from .scheduled_attention_backward import ScheduledAttentionBackwardArtifact


def verify_attention_ancestry(artifact, *, target, architecture):
    import re
    from .scheduled_matmul import find_tessera_opt, run_tessera_opt
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError('attention packaging requires native Schedule replay')
    replayed = run_tessera_opt(tool, artifact.schedule_ir, '--tessera-schedule-to-tile')
    if replayed != artifact.tile_ir:
        raise ValueError('attention Tile IR disagrees with native Schedule replay')
    header = re.match(r'\s*module attributes \{([^{}]*)\}', replayed)
    if header is None or any(
        re.search(r'(?:^|,)\s*' + re.escape(key) + r' = "' + re.escape(value) + r'"(?:,|$)', header.group(1)) is None
        for key, value in [('tessera.target', target), ('tessera.arch', architecture)]
    ):
        raise ValueError('attention native parent target disagrees')
    project_attention_descriptor(artifact, run_tessera_opt(tool, artifact.schedule_ir, '--canonicalize'))


def project_attention_descriptor(artifact, parent) -> None:
    """Project the supported attention ABI from the native printed parent."""
    import re
    import struct
    backward = isinstance(artifact, ScheduledAttentionBackwardArtifact)
    opname = 'attention_backward' if backward else 'attention'
    functions = re.findall(r'func.func @(\w+)\(([^\n]*)\) -> ([^\n]+)\n((?:(?!  func.func)[\s\S])*?)(?=\n  \})', parent)
    selected = [(name, args, results, body) for name, args, results, body in functions
                if re.search(r' = schedule\.' + opname + r' ', body)]
    if len(selected) != 1:
        raise ValueError('native attention projection requires one native function')
    name, args, results, body = selected[0]
    attrs_list = re.findall(r' = schedule\.' + opname + r' [^{]+\{([^{}]*)\}', body)
    if len(attrs_list) != 1:
        raise ValueError('native attention projection requires one native schedule')
    attrs = attrs_list[0]
    tensor = r'tensor<([1-9][0-9]*)x([1-9][0-9]*)x([1-9][0-9]*)x([1-9][0-9]*)x(f32|f16|bf16)>'
    inputs = re.findall(tensor, args)
    outputs = re.findall(tensor, results)
    bias = artifact.bias_name is not None
    if len(inputs) != (4 if backward else 3) + int(bias):
        raise ValueError('native attention descriptor bias/input count disagrees with native IR')
    q, k, v = inputs[1:4] if backward else inputs[:3]
    b, hq, sq, d, storage = q
    bk, hkv, sk, dk, ks = k
    bv, hv, sv, dv, vs = v
    out = (b, hq, sq, dv, storage)
    forward_out = out[:-1] + ('f32',) if artifact.target == 'rocm' else out
    expected_outputs = [q[:-1] + ('f32',), k[:-1] + ('f32',), v[:-1] + ('f32',)] if backward else [forward_out]
    if (b != bk or b != bv or hkv != hv or sk != sv or d != dk
            or storage != ks or storage != vs or outputs != expected_outputs
            or (backward and inputs[0] != out)
            or (bias and inputs[-1] != (b, hq, sq, sk, 'f32'))):
        raise ValueError('native attention native tensor ABI is unsupported')
    expected = dict(function_name=name, dims=tuple(map(int, (b, hq, hkv, sq, sk, d, dv))),
                    storage=storage, dtype={'f32': 'fp32', 'f16': 'fp16', 'bf16': 'bf16'}[storage])
    if backward:
        # The backward producer synthesizes a native entry; function_name is
        # the public source alias and is not used to bind the runtime symbol.
        expected.pop('function_name')
    for field, value in expected.items():
        if type(getattr(artifact, field)) is not type(value) or getattr(artifact, field) != value:
            raise ValueError(f'native attention descriptor {field} disagrees with native IR')
    fields = ['scale', 'causal', 'window_left', 'window_right', 'softcap', 'dropout_p',
              'dropout_seed', 'workgroup_size', 'recurrence']
    fields += (['query_block', 'key_block', 'split_count', 'workspace_bytes',
                'lse_checkpoint_policy', 'lse_checkpoint_selection'] if backward else
               ['tile_q', 'tile_kv', 'accum', 'backward_lse_policy', 'backward_lse_selection'])
    for field in fields:
        value = getattr(artifact, field)
        match = re.search(r'(?:^|, )' + field + r' = ("[^"]*"|true|false|[-+0-9.eE]+)(?: : (f32|i64))?(?:,|$)', attrs)
        if match is None:
            raise ValueError(f'native attention native field {field} is missing')
        raw, typ = match.groups()
        if raw.startswith('"'):
            projected = raw[1:-1]
        elif raw in ('true', 'false'):
            projected = raw == 'true'
        elif typ == 'i64':
            projected = int(raw)
        elif typ == 'f32':
            projected = struct.unpack('f', struct.pack('f', float(raw)))[0]
            if type(value) is float:
                value = struct.unpack('f', struct.pack('f', value))[0]
        else:
            raise ValueError('native attention native attribute type is unsupported')
        if type(value) is not type(projected) or value != projected:
            raise ValueError(f'native attention descriptor {field} disagrees with native IR')
    if f'bias = {str(bias).lower()}' not in attrs or 'accum = "f32"' not in attrs:
        raise ValueError('native attention bias/accumulation disagrees with native IR')
    if backward and (artifact.reduction_order != (0, 1) or
                     'reduction_order = array<i64: 0, 1>' not in attrs):
        raise ValueError('native attention reduction order disagrees with native IR')
    aliases = (list(artifact.input_names) + list(artifact.output_names) if backward else
               [artifact.q_name, artifact.k_name, artifact.v_name, artifact.output_name])
    if bias:
        aliases.append(artifact.bias_name)
    if any(type(n) is not str or not n.isidentifier() for n in aliases) or len(set(aliases)) != len(aliases):
        raise ValueError('native attention host binding aliases must be distinct identifiers')
