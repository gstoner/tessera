"""Portable trusted executable records for the source-active SM120 JVP.

Digests detect corruption and inconsistent witnesses. They do not authenticate
arbitrary native code supplied by another party. No compiler/provider is used
when validating or restoring the captured package metadata.
"""
import copy
import hashlib
from pathlib import Path
import re

from .native_sm120_active_source import NativeActiveSourceProgram, validate_plan
from .portable_nvidia_tensor_jvp import digest

SCHEMA = 'tessera.nvidia.tensor_active_source_jvp.v1'
COMPILER_PATH = 'nvidia_sm120_active_source_jvp_compiled'


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _abi(plan, index, entry):
    source = plan['buffers'][plan['role_indices'][0]]
    rhs = plan['buffers'][plan['role_indices'][1]]
    m, k = source['shape']
    n = rhs['shape'][1]
    step = plan['steps'][index]
    return {
        'schema': 'tessera.nvidia.active_source_program_step.v1',
        'step_bytes': 112, 'kind': 'row_pair' if index == 0 else 'matmul',
        'entry': entry, 'inputs': step['inputs'], 'outputs': step['outputs'],
        'dimensions': [m, k, 0] if index == 0 else [m, n, k],
        'storage': source['storage'], 'block_size': 128 if index == 0 else 32,
        'parameters': ['ptr'] * (4 if index == 0 else 3) + ['i64'] * (2 if index == 0 else 3),
    }


def seal_members(plan_json, records):
    """Attach checked-frame metadata to compiler-produced native members."""
    plan = validate_plan(plan_json)
    result = []
    for index, original in enumerate(records):
        record = copy.deepcopy(original)
        record['abi'] = _abi(plan, index, record['entry'])
        record['provenance'] = {
            'program_digest': sha(plan_json),
            'source_graph_ir_digest': sha(plan['source_graph_ir']),
            'paired_graph_ir_digest': sha(plan['paired_graph_ir']),
            **{name + '_digest': sha(record[name]) for name in
               ('graph_ir', 'schedule_ir', 'tile_ir', 'target_ir', 'native_ir')},
        }
        record['member_digest'] = digest(record)
        result.append(record)
    validate_members(plan_json, result)
    return tuple(result)


def validate_members(plan_json, records):
    plan = validate_plan(plan_json)
    if not isinstance(records, (tuple, list)) or len(records) != 3:
        raise ValueError('source-active portable member count differs')
    keys = {'graph_ir', 'schedule_ir', 'tile_ir', 'target_ir', 'native_ir', 'entry',
            'ptx', 'image_digest', 'resources', 'compiler', 'toolchain', 'abi',
            'provenance', 'member_digest'}
    for i, record in enumerate(records):
        if (not isinstance(record, dict) or set(record) != keys
                or any(not isinstance(record[name], str) or not record[name] for name in
                       ('graph_ir','schedule_ir','tile_ir','target_ir','native_ir','entry',
                        'ptx','image_digest','compiler','toolchain','member_digest'))
                or digest({k: v for k, v in record.items() if k != 'member_digest'}) != record['member_digest']):
            raise ValueError('source-active portable member schema/integrity differs')
        expected_provenance = {
            'program_digest': sha(plan_json),
            'source_graph_ir_digest': sha(plan['source_graph_ir']),
            'paired_graph_ir_digest': sha(plan['paired_graph_ir']),
            **{name + '_digest': sha(record[name]) for name in
               ('graph_ir', 'schedule_ir', 'tile_ir', 'target_ir', 'native_ir')},
        }
        if (record['provenance'] != expected_provenance
                or record['graph_ir'] != plan['member_graphs'][i]
                or record['image_digest'] != sha(record['ptx'])
                or digest(record['abi']) != digest(_abi(plan, i, record['entry']))):
            raise ValueError('source-active portable member provenance/ABI differs')
        resources = record['resources']
        if (not isinstance(resources, dict) or set(resources) != {
                'registers_per_thread','static_shared_memory_bytes','spill_store_bytes','spill_load_bytes'}
                or any(type(v) is not int or v < 0 for v in resources.values())):
            raise ValueError('source-active portable resource record differs')
        entry = record['entry']
        if ((i == 0 and not re.fullmatch(r'tessera_native_row_[0-9a-f]{16}', entry))
                or (i > 0 and entry != 'nvidia_sm120_scheduled_matmul_native_tensor_consumer_kernel')
                or re.findall(r'\.visible\s+\.entry\s+([A-Za-z0-9_]+)', record['ptx']) != [entry]
                or not re.search(r'\.target\s+sm_120a\b', record['ptx'])
                or re.findall(r'\bllvm.func @(\w+)\(', record['native_ir']) != [entry]):
            raise ValueError('source-active portable native entry/target differs')
        # Bind the unique row marker and its Tile-to-Target body seal. Actual
        # body correctness was verified by native MLIR before image generation.
        if i == 0:
            graph_hash = re.findall(r'body_hash = "([0-9a-f]{64})"', record['graph_ir'])
            source_hash = re.findall(r'source_hash = "([0-9a-f]{64})"', record['tile_ir'])
            tile_hash = re.findall(r'(?<![\w_])body_hash = "([0-9a-f]{64})"', record['tile_ir'])
            parent_hash = re.findall(r'tile_body_hash = "([0-9a-f]{64})"', record['target_ir'])
            if (len(graph_hash) != 1 or source_hash != graph_hash or len(tile_hash) != 1
                    or parent_hash != tile_hash or entry != 'tessera_native_row_' + graph_hash[0][:16]
                    or 'schedule.artifact' not in record['schedule_ir']
                    or 'tile.row_region_kernel' not in record['tile_ir']
                    or 'tessera_nvidia.row_region_kernel' not in record['target_ir']):
                raise ValueError('source-active portable row body seal differs')
        elif ('schedule.matmul' not in record['schedule_ir']
              or 'tile.mma ' not in record['tile_ir']
              or 'rhs_storage_order = "row_major"' not in record['graph_ir']):
            raise ValueError('source-active portable matmul route differs')
    return plan


def manifest(plan_json, records):
    validate_members(plan_json, records)
    data = {'schema': SCHEMA, 'plan_json': plan_json, 'members': copy.deepcopy(list(records))}
    data['contract_digest'] = digest(data)
    return data


def restore(data):
    if (not isinstance(data, dict)
            or set(data) != {'schema','plan_json','members','contract_digest'}
            or data['schema'] != SCHEMA
            or digest({k: v for k, v in data.items() if k != 'contract_digest'}) != data['contract_digest']):
        raise ValueError('source-active portable schema/integrity differs')
    data = copy.deepcopy(data)
    plan = validate_members(data['plan_json'], data['members'])
    native = NativeActiveSourceProgram(plan['source_graph_ir'], '', data['plan_json'], Path('/unavailable/compiler'), '')
    return native, None, tuple(data['members'])
