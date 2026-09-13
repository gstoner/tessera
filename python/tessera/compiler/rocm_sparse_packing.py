"""RDNA4 wave32 sparse A/index packing for f16 or bf16 SWMMAC probes.

This produces physical register payloads, not general Schedule/Tile admission.
The native probe consumes these bytes. Layout: RDNA4 ISA 7.12.2/7.12.3,
independently checked against AMD's matrix instruction calculator. OPSEL=0.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SparseWMMAInputs:
    a: bytes
    b: bytes
    indices: bytes
    dtype: str


def pack_sparse_wmma_inputs(a, b) -> SparseWMMAInputs:
    """Pack logical A[16,32], B[32,16]; refuse non-2:4 data without pruning."""
    a,b = np.asarray(a),np.asarray(b)
    if a.shape != (16,32) or b.shape != (32,16) or a.dtype != b.dtype:
        raise ValueError("sparse WMMA requires matching 16x32/32x16 operands")
    if str(a.dtype) not in {"float16","bfloat16"}:
        raise ValueError("sparse WMMA packing supports f16/bf16 storage")
    # Immutable output bytes own the captured representation, including signed
    # zero and nonfinite bit patterns at explicitly selected positions.
    av=np.zeros((32,8),np.uint16); bv=np.zeros((32,16),np.uint16)
    indices=np.zeros(32,np.uint32)
    abits=np.ascontiguousarray(a).view(np.uint16)
    bbits=np.ascontiguousarray(b).view(np.uint16)
    for row in range(16):
        for group in range(8):
            start=group*4
            # Preserve signed zero as a stored value rather than discarding its
            # sign. More than two nonzero bit patterns is outside this format.
            selected=np.flatnonzero(abits[row,start:start+4])
            if len(selected)>2:
                raise ValueError("sparse WMMA requires at most two stored values per four")
            selected=sorted(list(selected)+[i for i in range(4) if i not in selected][:2-len(selected)])
            lane=row+16*((start//8)%2)
            reg=2*(start//16)+(start//4)%2
            for i,col in enumerate(selected):
                av[lane,2*reg+i]=abits[row,start+col]
            indices[lane] |= np.uint32((selected[0] | selected[1]<<2) << (4*reg))
    for k in range(32):
        for col in range(16):
            lane=col+16*((k//8)%2)
            reg=4*(k//16)+(k//2)%4
            bv[lane,2*reg+k%2]=bbits[k,col]
    return SparseWMMAInputs(av.tobytes(),bv.tobytes(),indices.tobytes(),str(a.dtype))


def sparse_wmma_target_ir(dtype: str) -> str:
    """Produce the verified Target IR consumer for this exact packing contract.

    This internal four-buffer entry consumes packed A/B/indices and writes
    lane-major f32 accumulators. It is not the public dense matmul ABI.
    """
    if dtype not in {"float16", "bfloat16"}:
        raise ValueError("sparse target IR requires f16/bf16 packing")
    element = "f16" if dtype == "float16" else "bf16"
    return f'''module attributes {{gpu.container_module}} {{
  gpu.module @sparse {{
    gpu.func @probe(%a: memref<256x{element}>, %b: memref<512x{element}>,
                    %indices: memref<32xi32>, %out: memref<256xf32>) kernel
        attributes {{gpu.known_block_size = array<i32: 32, 1, 1>}} {{
      %tid = gpu.thread_id x
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %ai = arith.muli %tid, %c8 : index
      %bi = arith.muli %tid, %c16 : index
      %av = vector.load %a[%ai] : memref<256x{element}>, vector<8x{element}>
      %bv = vector.load %b[%bi] : memref<512x{element}>, vector<16x{element}>
      %idx = memref.load %indices[%tid] : memref<32xi32>
      %zero = arith.constant dense<0.0> : vector<8xf32>
      %result = tessera_rocm.swmmac %av, %bv, %zero, %idx {{arch = "gfx1201"}}
          : vector<8x{element}>, vector<16x{element}>, vector<8xf32> -> vector<8xf32>
      vector.store %result, %out[%ai] : memref<256xf32>, vector<8xf32>
      gpu.return
    }}
  }}
}}
'''


def sparse_wmma_schedule_ir(dtype: str) -> str:
    """Emit the bounded Schedule fragment producer before Tile/Target lowering.

    Logical dense/CSR input capture is not implied by this physical packing API.
    """
    return sparse_wmma_target_ir(dtype).replace("tessera_rocm.swmmac", "schedule.sparse_mma")
