"""Consume an existing compiler JVP launch plan with native storage children.

This binds the compiler's paired program; it does not invent a derivative or
accept two unrelated callables as an AD rule. Buffers remain caller-owned.
"""
from __future__ import annotations
import inspect
from .native_jvp import NativeJVPArtifact
from .native_gpu_storage import NativeGPUStoragePackage
from .native_storage_contract import generate_tensor_binding, read_tensor_contract


class NativeStorageJVP:
    def __init__(self, artifact: NativeJVPArtifact):
        if not isinstance(artifact, NativeJVPArtifact):
            raise ValueError('paired storage requires a compiler NativeJVPArtifact')
        artifact.validate()
        self.artifact = artifact
        self._identity = artifact.artifact_hash
        self.steps = []
        names = artifact.contract['arg_names']
        if len(set(names)) != len(names):
            raise ValueError('paired argument names are not unique')
        self.signature = inspect.Signature([inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD) for n in names])
        actions = artifact.contract['tile_program']['actions']
        expected = [(s['id'], s['child_digest']) for s in artifact.contract['steps']]
        if [(s['id'], s['child_digest']) for s in actions] != expected:
            raise ValueError('paired Tile plan and child launch lineage disagree')
        known = set(names)
        produced = set()
        for step in artifact.contract['steps']:
            child = step.get('child_artifact', {})
            if set(child) != {'native_storage_json'}:
                raise ValueError('paired storage needs serialized native child artifacts')
            package = NativeGPUStoragePackage.from_json(child['native_storage_json'], expected_digest=step['child_digest'])
            target = 'nvidia_sm120' if package.backend == 'nvidia' else 'rocm'
            if target != artifact.contract['target'] or package.chip != ('sm_120' if target == 'nvidia_sm120' else 'gfx1151'):
                raise ValueError('paired child backend differs from its compiler product')
            parameters = read_tensor_contract(package)['arguments']
            signature = inspect.Signature([inspect.Parameter(p['name'], inspect.Parameter.POSITIONAL_OR_KEYWORD) for p in parameters])
            binding = generate_tensor_binding(package, signature)
            inputs, outputs = step.get('inputs', []), step.get('outputs', [])
            if len(inputs) != len(parameters) or any(n not in known for n in inputs):
                raise ValueError('paired child has unresolved argument bindings')
            if len(outputs) != sum(bool(p.get('writable', False)) for p in parameters) or any(n in produced for n in outputs):
                raise ValueError('paired child output ownership disagrees')
            produced.update(outputs)
            known.update(outputs)
            self.steps.append((binding, tuple(inputs), tuple(outputs)))
        if artifact.contract.get('output_order') != ['primal', 'tangent'] or not {'primal', 'tangent'} <= produced:
            raise ValueError('paired storage must produce both primal and tangent')

    def __call__(self, *args, **kwargs):
        self.artifact.validate()
        if self.artifact.artifact_hash != self._identity:
            raise ValueError('paired program changed after native binding')
        values = dict(self.signature.bind(*args, **kwargs).arguments)
        calls = []
        # Preflight every tensor ABI and resolve all output aliases before the
        # first device write. No Python numerical execution or allocation.
        for binding, inputs, outputs in self.steps:
            arguments = tuple(values[n] for n in inputs)
            prepared = binding.prepare(*arguments)
            values.update(zip(outputs, prepared[-1], strict=True))
            calls.append((binding, arguments))
        for binding, arguments in calls:
            binding(*arguments)
        return values['primal'], values['tangent']

    def close(self):
        for binding, _, _ in self.steps:
            binding.close()


def build_native_storage_jvp(source_graph_ir, *, compiler, llvm_bin, backend, chip):
    """Compile one bounded forward product and bind its generated physical child.

    The C++ transform supplies the primal and tangent SSA graph, scalarizes its
    admitted operations, and emits the shared-storage child. Python only binds
    that compiler output to the native package ABI; it defines no derivative.
    """
    import re
    from .native_gpu_storage import _run, _decode_image, build_native_gpu_storage
    from .native_gpu_tensor import TensorSpec, IndexSpec
    from .native_storage_contract import attach_tensor_contract
    from .native_jvp import build_native_jvp_artifact

    if (backend, chip) not in (('nvidia', 'sm_120'), ('rocm', 'gfx1151')):
        raise ValueError('native storage JVP needs an owning GPU target')
    if 'tessera.frontend.authority = "tracer"' not in source_graph_ir:
        raise ValueError('native storage JVP requires tracer-owned source IR')
    generated = _run(compiler, '--allow-unregistered-dialect',
                     '--tessera-autodiff-forward=emit-storage-child=true', source=source_graph_ir)
    def integer(name):
        match = re.search(r'tessera.native_jvp_' + name + r' = (\d+) : i64', generated)
        if not match:
            raise ValueError('compiler child lacks its ABI dimensions')
        return int(match[1])
    width, count = integer('width'), integer('inputs')
    output_width = integer('output_width')
    pair = re.search(r'tessera.native_jvp_pair = "((?:\\.|[^"\\])*)"', generated)
    wrt = re.search(r'tessera.native_jvp_wrt = \[([0-9, ]+)\]', generated)
    if pair is None or wrt is None:
        raise ValueError('compiler child lacks its paired-program lineage')
    paired_ir = _decode_image(pair[1]).decode()
    # Compiler metadata is consumed here; the package records its own ABI
    # manifest, while the parent records both source and paired-program hashes.
    recipe, replaced = re.subn(r'^module attributes .* \{$', 'module {', generated, count=1, flags=re.M)
    if replaced != 1:
        raise ValueError('compiler child module header is malformed')
    specs: tuple[TensorSpec | IndexSpec, ...] = tuple(TensorSpec(f'arg{i}', 'float32', (width,)) for i in range(count)) + (
        TensorSpec('out', 'float32', (output_width,), True), TensorSpec('dout', 'float32', (output_width,), True),
        IndexSpec('n', width, width))
    recipe = attach_tensor_contract(recipe, specs, grid=(1, 1, 1), block=(width, 1, 1))
    package = build_native_gpu_storage(recipe, compiler=compiler, llvm_bin=llvm_bin, backend=backend, chip=chip)
    names = [s.name for s in specs]
    return build_native_jvp_artifact(
        target='nvidia_sm120' if backend == 'nvidia' else 'rocm',
        architecture='sm120' if backend == 'nvidia' else chip, family='native_storage',
        source_graph_ir=source_graph_ir, paired_jvp_ir=paired_ir,
        wrt_indices=[int(i) for i in wrt[1].split(',')], arg_names=names,
        steps=[dict(id='paired_child', child_digest=package.binding_digest,
                    child_artifact={'native_storage_json': package.to_json()}, inputs=names,
                    outputs=['primal', 'tangent'])])
