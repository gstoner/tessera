"""Shared SSD Schedule producer; no target package or promotion is implied."""
from dataclasses import dataclass
import hashlib
from pathlib import Path

from .scheduled_matmul import find_tessera_opt, run_tessera_opt


@dataclass(frozen=True)
class ScheduledSSD:
    schedule_ir: str
    lowered_ir: str
    compiler_digest: str

    def validate(self, compiler=None):
        tool = Path(compiler) if compiler is not None else find_tessera_opt()
        if tool is None or hashlib.sha256(tool.read_bytes()).hexdigest() != self.compiler_digest:
            raise ValueError('SSD compiler identity changed')
        if run_tessera_opt(tool, self.schedule_ir, '--tessera-schedule-to-tile') != self.lowered_ir:
            raise ValueError('SSD lowered artifact disagrees with Schedule replay')


def lower_scheduled_ssd(time, heads, states, width, chunk_size, *, compiler=None):
    """Produce static f32 recurrence with immutable initial/final carry lineage.

    Inputs: X[T,H,P], multiplicative decay[T,H], B/C[T,H,N], initial[H,N,P].
    Outputs: Y[T,H,P], carry[H,N,P], checkpoints[ceil(T/chunk),H,N,P].
    Native Schedule verification remains authoritative for shape and size limits.
    """
    if any(type(v) is not int or not 0 < v <= (1 << 24)
           for v in (time, heads, states, width, chunk_size)):
        raise ValueError('SSD requires positive bounded integer dimensions')
    if chunk_size > time:
        raise ValueError('SSD chunk size cannot exceed sequence length')
    tool = Path(compiler) if compiler is not None else find_tessera_opt()
    if tool is None:
        raise RuntimeError('SSD requires the native Schedule compiler')
    def tensor(*shape):
        return 'tensor<' + 'x'.join(map(str, shape)) + 'xf32>'
    x, decay = tensor(time, heads, width), tensor(time, heads)
    bc, carry = tensor(time, heads, states), tensor(heads, states, width)
    checkpoints = tensor((time - 1) // chunk_size + 1, heads, states, width)
    inputs = [x, decay, bc, bc, carry]
    outputs = [x, carry, checkpoints]
    args = ', '.join(f'%arg{i}: {t}' for i, t in enumerate(inputs))
    values = ', '.join(f'%arg{i}' for i in range(5))
    results = ', '.join(outputs)
    source = f'''module {{
  func.func @ssd({args}) -> ({results}) {{
    %r:3 = "schedule.ssd"({values}) {{chunk_size = {chunk_size} : i64}}
      : ({', '.join(inputs)}) -> ({results})
    return %r#0, %r#1, %r#2 : {results}
  }}
}}'''
    schedule = run_tessera_opt(tool, source, '--canonicalize')
    lowered = run_tessera_opt(tool, schedule, '--tessera-schedule-to-tile')
    return ScheduledSSD(schedule, lowered, hashlib.sha256(tool.read_bytes()).hexdigest())
