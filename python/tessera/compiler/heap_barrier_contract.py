"""Consumed heap protocols: stream epochs, per-object pins and atomic transactions.

Schema 3 adds a system-scope acquire/release try-lock shared by graph writers
and final retirement. All competing metadata users must participate; it does
not enable arbitrary concurrent access to the epoch-based resident pool.
"""
import json
import re
from .native_gpu_storage import _decode_image

ATTRIBUTE = 'tessera.heap_protocol'
MODES: tuple[str, ...] = ('atomic_graph_incremental', 'atomic_retire_marked', 'allocate', 'collect', 'graph', 'mark', 'collect_seeded', 'collect_slice',
         'graph_checked', 'retire', 'reclaim', 'pin', 'unpin', 'allocate_marked', 'graph_incremental',
         'mark_begin', 'mark_step', 'retire_marked', 'reclaim_pinned', 'inspect')

ATOMIC_MODES = ('allocate', 'allocate_marked', 'graph_checked', 'graph_incremental',
                'mark_begin', 'mark_step', 'retire_marked', 'reclaim_pinned', 'pin', 'unpin', 'inspect')
MODES += tuple('atomic_' + mode for mode in ATOMIC_MODES if 'atomic_' + mode not in MODES)


def heap_contract(slots, width, references, mode):
    if (type(slots) is not int or not 1 <= slots <= 256 or
            type(width) is not int or width < 1 or slots * width > 262144 or
            type(references) is not int or not 1 <= references <= 32 or mode not in MODES):
        raise ValueError('invalid heap protocol dimensions or mode')
    data = dict(schema=1, slots=slots, width=width, references=references, mode=mode,
                state_layout='generation_length_lifecycle', handle='slot_generation_i64_pair',
                synchronization='exclusive_stream_epoch', reclamation='all_recorded_readers',
                publication='kernel_completion', address_space=1, generation_limit=(1 << 31) - 1,
                final_remark='exclusive', payload='immutable_until_reuse')
    if mode in ('pin', 'unpin', 'allocate_marked', 'graph_incremental', 'mark_begin', 'mark_step', 'retire_marked', 'reclaim_pinned'):
        data.update(schema=2, synchronization='exclusive_metadata_epoch',
                    reclamation='per_slot_pins_and_recorded_readers',
                    marking='incremental_update_tricolor', payload='immutable_until_unpinned_reuse',
                    final_remark='exclusive_metadata')
    if mode.startswith('atomic_'):
        data.update(schema=3, synchronization='system_scope_acq_rel_try_lock',
                    reclamation='per_slot_pins_and_recorded_readers',
                    marking='incremental_update_tricolor', payload='immutable_until_unpinned_reuse',
                    final_remark='same_gate_as_graph_writers', publication='release_gate',
                    contention_status=3, gate='aligned_i64_zero_initialized_shared_by_all_metadata_users')
    return data


def attach_heap_contract(source, slots, width, references, mode):
    if ATTRIBUTE in source:
        raise ValueError('duplicate heap protocol')
    data = heap_contract(slots, width, references, mode)
    encoded = json.dumps(data, sort_keys=True, separators=(',', ':')).replace('"', '\\22')
    if source.count('module attributes {') != 1:
        raise ValueError('heap protocol requires one manifest-bearing module')
    return source.replace('module attributes {', f'module attributes {{{ATTRIBUTE} = "{encoded}", ', 1)


def read_heap_contract(source):
    matches = re.findall(r'tessera\.heap_protocol\s*=\s*"((?:\\.|[^"\\])*)"', source)
    if len(matches) != 1:
        raise ValueError('native heap requires exactly one protocol manifest')
    try:
        data = json.loads(_decode_image(matches[0]).decode('utf8'))
        expected = heap_contract(data['slots'], data['width'], data['references'], data['mode'])
        # JSON bool is numerically equal to int: compare canonical encodings.
        if json.dumps(data, sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError('unsupported heap protocol')
    except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('malformed heap protocol') from exc
    return data
