"""Bounded native unary ancestry and descriptor projection shared by consumers."""
import re
from .scheduled_matmul import find_tessera_opt, run_tessera_opt


def verify_unary_ancestry(artifact, *, target, architecture):
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("unary packaging requires native Schedule replay")
    runner = run_tessera_opt
    if target == "x86":
        # The replay is memoized on the exact Schedule text and compiler digest.
        from .x86_compile_cache import run as runner
    tile = runner(tool, artifact.schedule_ir, "--tessera-schedule-to-tile")
    if tile != artifact.tile_ir:
        raise ValueError("unary Tile IR disagrees with native Schedule replay")
    header = re.match(r'\s*module attributes \{([^{}]*)\}', tile)
    if header is None or any(
        re.search(r'(?:^|,)\s*' + re.escape(key) + r' = "' + re.escape(value) + r'"(?:,|$)', header[1]) is None
        for key, value in (("tessera.target", target), ("tessera.arch", architecture))
    ):
        raise ValueError("unary native parent target disagrees")
    if (artifact.target, artifact.architecture) != (target, architecture):
        raise ValueError("unary descriptor target disagrees")
    verify_unary_projection(artifact, runner(tool, artifact.schedule_ir, "--canonicalize"))


def verify_unary_projection(artifact, parent: str) -> None:
    """Project the bounded unary ABI from native-printed IR, not Graph objects."""
    import math
    import re
    tensor = r'tensor<((?:[1-9][0-9]*x)*)(f32|f16|bf16)>'
    functions = re.findall(
        r'func.func @([\w]+)\(%[\w]+: ' + tensor
        + r'\) -> ' + tensor + r'(?: attributes \{[^{}]*\})? \{',
        parent,
    )
    if len(functions) != 1 or parent.count('func.func ') != 1:
        raise ValueError('Native unary descriptor requires one static tensor function')
    name, input_dims, storage, output_dims, output_storage = functions[0]
    input_shape = tuple(int(d) for d in input_dims.split('x') if d)
    output_shape = tuple(int(d) for d in output_dims.split('x') if d)
    ops = re.findall(r' = schedule\.(softmax|reduce|norm) %\w+ \{([^{}]*)\}', parent)
    if len(ops) != 1 or not input_shape:
        raise ValueError('Native unary descriptor requires one native unary schedule')
    family, attrs = ops[0]
    rocm = artifact.target == "rocm" and artifact.architecture in {"gfx1151", "gfx1201"}
    # gfx1151 carries the narrow-storage/keepdims envelope (E2E-REAL-6); a
    # reduction always accumulates into f32 storage, softmax preserves storage.
    rocm_unary = artifact.target == "rocm" and artifact.architecture == "gfx1151"
    if family == 'reduce' and rocm_unary:
        if output_storage != 'f32':
            raise ValueError('Native unary reduction output must be f32 storage')
    elif storage != output_storage:
        raise ValueError('Native unary input/output storage disagrees')
    workgroup = 256 if rocm else 1
    required = ['accum = "f32"', f'storage = "{storage}"', f'workgroup_size = {workgroup} : i64']
    if family == 'softmax':
        required += ['exp_mode = "accurate"', 'ftz = false']
    if any(re.search(r'(?:^|, )' + re.escape(value) + r'(?:,|$)', attrs) is None
           for value in required):
        raise ValueError('Native unary native arithmetic policy is unsupported')
    axis_match = re.search(r'(?:^|, )axis = (-?[0-9]+) : i64(?:,|$)', attrs)
    if axis_match is None:
        raise ValueError('Native unary descriptor lacks native axis')
    axis = int(axis_match[1])
    expected = dict(function_name=name, input_shape=input_shape,
                    output_shape=output_shape, family=family, dtype={'f32': 'fp32', 'f16': 'fp16', 'bf16': 'bf16'}[storage],
                    storage=storage, accum='f32', keepdims=False,
                    workgroup_size=workgroup, schedule='serial', epsilon=0.0)
    if family == 'norm':
        # E2E-REAL-6 x86 (2026-09-28): the unweighted row normalization.
        import struct
        kind = re.search(r'(?:^|, )kind = "(rmsnorm|layernorm)"(?:,|$)', attrs)
        eps = re.search(r'(?:^|, )epsilon = ([-+0-9.eE]+) : f32(?:,|$)', attrs)
        norm_target_ok = artifact.target == 'x86' or (
            artifact.target == 'rocm' and artifact.architecture == 'gfx1201'
            and storage in {'f16', 'bf16', 'f32'} and kind is not None and kind[1] == 'rmsnorm'
        )
        if (not norm_target_ok or axis != -1 or output_shape != input_shape
                or kind is None or eps is None
                or struct.pack('f', float(eps[1])) != struct.pack('f', artifact.epsilon)):
            raise ValueError('Native norm kind/axis/epsilon is unsupported')
        expected.update(kind=kind[1], axis=-1, rows=math.prod(input_shape[:-1]),
                        columns=input_shape[-1], outer=1, axis_extent=1, inner=1,
                        epsilon=artifact.epsilon)
    elif family == 'softmax':
        if axis != -1 or output_shape != input_shape:
            raise ValueError('Native native softmax shape/axis is unsupported')
        expected.update(kind='softmax', axis=-1, rows=math.prod(input_shape[:-1]),
                        columns=input_shape[-1], outer=1, axis_extent=1, inner=1)
    else:
        axis = axis + len(input_shape) if axis < 0 else axis
        kind_pattern = "sum|mean|max|min" if rocm else "sum|mean|max"
        kind = re.search(r'(?:^|, )kind = "(' + kind_pattern + r')"(?:,|$)', attrs)
        keep = re.search(r'(?:^|, )keepdims = (true|false)(?:,|$)', attrs)
        keepdims = keep is not None and keep[1] == 'true'
        if keepdims and artifact.target != 'x86' and not rocm_unary:
            raise ValueError('native keepdims projection requires x86 or gfx1151')
        expected['keepdims'] = keepdims
        expected_shape = input_shape[:axis] + ((1,) if keepdims else ()) + input_shape[axis+1:]
        if (not 0 <= axis < len(input_shape) or (not rocm and axis != len(input_shape)-1)
                or output_shape != expected_shape or kind is None):
            raise ValueError('Native native reduction shape/axis/kind is unsupported')
        expected.update(kind=kind[1], axis=axis, rows=1, columns=1,
                        outer=math.prod(input_shape[:axis]), axis_extent=input_shape[axis],
                        inner=math.prod(input_shape[axis+1:]))
    for field, value in expected.items():
        actual = getattr(artifact, field)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f'Native unary descriptor field {field} disagrees with native IR')
    # Names remain explicit host binding aliases; physical order and shapes
    # above come from IR. Do not relabel aliases as native SSA provenance.
    if (type(artifact.input_name) is not str or type(artifact.output_name) is not str
            or not artifact.input_name.isidentifier() or not artifact.output_name.isidentifier()
            or artifact.input_name == artifact.output_name):
        raise ValueError('Native unary buffer aliases must be distinct identifiers')
