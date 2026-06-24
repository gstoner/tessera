# 12.5.3. DS Stack Operations for Ray Tracing

> RDNA4 ISA — pages 164–165

12.5.3. DS Stack Operations for Ray Tracing
These LDS instructions can be used to manage a per-thread shallow stack in LDS used in ray tracing BVH
traversal. BVH structures consist of box nodes and triangle nodes. A box node has up to four child node
pointers which may all be returned to the shader (to VGPRs) for a given ray (thread). The traversal shader
follows one pointer per ray per iteration, so extra pointers are pushed to a per-thread stack in LDS. Note: the
returned pointers are sorted.

This "short stack" has a limited size beyond which the stack wraps around and overwrites older items. When
the stack is exhausted, the shader switches to a stackless mode where it looks up the parent of the current node
from a table in memory. The shader tracks the last visited address to avoid re-traversing subtrees.

DS_BVH_STACK_PUSH4_POP1_RTN_B32                  vgpr(dst), vgpr(stack_addr), vgpr(lvaddr), vgpr[4](data)
push 4 and pop 1
DS_BVH_STACK_PUSH8_POP1_RTN_B32                  dst, stack_addr, last_node_ptr, data[8], stack_size
push 8 and pop 1
DS_BVH_STACK_PUSH8_POP2_RTN_B64                  dst[2], stack_addr, last_node_ptr, data[8], stack_size
push 8, pop 1 and one tri-pop

Field             Size   Description
OP                8      DS_STORE_STACK
OFFSET0           8      bits[4:0] carry StackSize
OFFSET1           8      bit[1] primitive range enabled
                         bit[0] triangle pair optimization
VDST              9      Destination VGPR for resulting address (e.g. X or top of stack). Must be 0-255.
                         Returns the next "LV addr"

Field               Size        Description
ADDR                8           STACK_VGPR: Both a source and destination VGPR:
                                supplies the LDS stack address and is written back with updated address.
                                stack_addr[31:18] = stack_base[15:2] : stack base address (relative to allocated LDS space).
                                stack_addr[17:16] = stack_size[1:0] : 0=8DWORDs, 1=16, 2=32, 3=64 DWORDs per thread
                                stack_addr[15:0] = stack_index[15:0]. (bits [1:0] must be zero).
DATA0               9           LVADDR: Last Visited Address. Is compared with data values (next field) to determine the next
                                node to visit. Must be 0-255.
DATA1               9           4 VGPRs (X,Y,Z,W). Must be 0-255.
M0                  16          Unused.

Address Fields:

Field                    Bits      Size Type         Description
valid_entries            4:0       5      uint       The number of valid(non-clobbered) entries on the stack
entries_to_tlas          9:5       5      uint       The number of entries that must be popped before a BLAS→TLAS transition
                                                     should happen
ring_addr                14:10     5      uint       The rolling address in the ring buffer where new items are pushed onto the
                                                     stack at or popped from.
stack_base_addr          28:15     14     uint       DWORD offset in LDS where the ring buffer for the lane should be stored at
has_tlas_in_stack        29        1      boolean    0: All valid entries in the stack are from the same BVH
                                                     1: Some valid entries in the stack are from a parent’s BVH
has_overflowed           30        1      boolean    0: Rolling stack has not lost any entries pushed onto the stack
                                                     1: Some of the items pushed on the stack have been lost (for example
                                                     because of clobbering)
blas_to_tlas_pop         31        1      boolean    0: The last pop was from the same BVH as the previous pop
                                                     1: The last pop was from the parent BVH of the previous pop (so the parent
                                                     ray should be restored)
