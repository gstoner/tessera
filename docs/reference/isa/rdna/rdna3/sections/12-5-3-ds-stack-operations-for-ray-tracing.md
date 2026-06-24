# 12.5.3. DS Stack Operations for Ray Tracing

> RDNA3 ISA — pages 138–138

  • ds_permute_b32 : Dst[index[0..31]] = src[0..31]          Where [0..31] is the lane number
  • ds_bpermute_b32 : Dst[0..31] = src[index[0..31]]

The EXEC mask is honored for both reading the source and writing the destination. Index values out of range
wrap around (only index bits [6:2] are used, the other bits of the index are ignored). Reading from disabled
lanes returns zero.

In the instruction word: VDST is the dest VGPR, ADDR is the index VGPR, and DATA0 is the source data VGPR.
Note that index values are in bytes (so multiply by 4), and have the 'offset0' field added to them before use.

12.5.3. DS Stack Operations for Ray Tracing
DS_BVH_STACK_RTN_B32 is an LDS instruction to manage a per-thread shallow stack in LDS used in ray
tracing BVH traversal. BVH structures consist of box nodes and triangle nodes. A box node has up to four child
node pointers that may all be returned to the shader (to VGPRs) for a given ray (thread). A traversal shader
follows one pointer per ray per iteration, and extra pointers can be pushed to a per-thread stack in LDS. Note:
the returned pointers are sorted.

This "short stack" has a limited size beyond that the stack wraps around and overwrites older items. When the
stack is exhausted, the shader should switch to a stackless mode where it looks up the parent of the current
node from a table in memory. The shader program tracks the last visited address to avoid re-traversing
subtrees.

DS_BVH_STACK_RTN_B32 vgpr(dst), vgpr(stack_addr), vgpr(lvaddr), vgpr[4](data)

Field             Size   Description
OP                8      Instruction == DS_STORE_STACK (LDS only)
GDS               1      1 = GDS, 0 = LDS (must be: 0 = LDS)
OFFSET0           8      unused
OFFSET1           8      bits[5:4] carry StackSize (8, 16, 32, 64)
VDST              8      Destination VGPR for resulting address (e.g. X or top of stack)
                         Returns the next "LV addr"
