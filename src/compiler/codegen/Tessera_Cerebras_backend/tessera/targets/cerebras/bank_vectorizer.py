"""
Bank-aware vector width helper (scaffold).

Assume an 8-bank SRAM. This stub prefers vector widths that avoid issuing
two reads to the same bank in a cycle, by ensuring stride % banks ≠ 0 for
the typical SoA layout we generate for GEMM/attention. Tune against real CSL rules.
"""
from dataclasses import dataclass

@dataclass
class SRAMInfo:
    banks: int = 8
    read_ports: int = 2
    write_ports: int = 1

def pick_vector_width(elements_per_row: int, preferred=(16,8,4,2,1), sram=SRAMInfo()) -> int:
    for w in preferred:
        # Avoid trivial conflicts when the vector width evenly divides the bank count
        if (w % sram.banks) != 0 and (elements_per_row % w) == 0:
            return w
    return 1
