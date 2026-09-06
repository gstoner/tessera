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
