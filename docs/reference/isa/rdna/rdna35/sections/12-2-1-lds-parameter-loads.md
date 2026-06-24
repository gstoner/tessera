# 12.2.1. LDS Parameter Loads

> RDNA3.5 ISA — pages 131–133

Direct Load
   Loads a single DWORD from LDS and broadcasts the data to a VGPR across all lanes.

Indexed load/store and Atomic ops
   Load/store address comes from a VGPR and data to/from VGPR.
   LDS-ops require up to 3 inputs: 2data+1addr and immediate return VGPR.

Parameter Interpolation Load
   Reads pixel parameters from LDS per quad and loads them into one VGPR.
   Reads all 3 parameters per quad (P1, P1-P0 and P2-P0) and loads them into 3 lanes within the quad (the 4th
   lane receives zero).

The following sections describe these methods.

12.2. Pixel Parameter Interpolation
For pixel waves, vertex attribute data is preloaded into LDS and barycentrics (I, J) are preloaded into VGPRs
before the wave starts. Parameter interpolation can be performed by loading attribute data from LDS into
VGPRs using LDS_PARAM_LOAD and then using V_INTERP instructions to interpolate the value per pixel.

LDS-Parameter loads are used to read vertex parameter data and store them in VGPRs to be used for parameter
interpolation. These instructions operate like memory instructions except they use EXPcnt to track outstanding
reads and decrement EXPCnt when they arrive in VGPRs.

Pixel shaders can be launched before their parameter data has been written into LDS. Once the data is
available in LDS, the wave’s STATUS register "LDS_READY" bit is set to 1. Pixel shader waves stall if an
LDS_DIRECT_LOAD or LDS_PARAM_LOAD is to be issued before LDS_READY is set.

The most common form of interpolation involves weighting vertex parameters by the barycentric coordinates
"I" and "J". A common calculation is:

      Result = P0 + I * P10 + J * P20
           where "P10" is (P1 - P0), and "P20" is (P2 - P0)

Parameter interpolation involves two types of instructions:
  • LDS_PARAM_LOAD : to read packed parameter data from LDS into a VGPR (data packed per quad)
  • V_INTERP_* : VALU FMA instructions that unpack parameter data across lanes in a quad.

12.2.1. LDS Parameter Loads
Parameter Loads are only available in LDS, not in GDS, and only in CU mode (not WGP mode).

LDS_PARAM_LOAD reads three parameters (P0, P10, P20) of one 32-bit attribute or of two 16-bit attributes
from LDS into VGPRs. The are 3 parameters (P0, P10 and P20) are the same for the 4 pixels within a quad.
These values are spread out across VGPR lanes 0, 1 and 2 of each quad. Interpolation is then performed using
FMA with DPP so each lane uses its I or J value with the quad’s shared P0, P10 and P20 values.

                                            Table 57. LDSDIR Instruction Fields
Field              Size       Description
OP                 2          Opcode:
                               0: LDS_PARAM_LOAD
                               1: LDS_DIRECT_LOAD
                               2,3: Reserved
WAITVDST           4          Wait for the number of previously issued still outstanding VALU instructions to be less than
                              or equal to this number. Used to avoid Write-After-Read hazards on VGPRs.
VDST               8          Destination VGPR
ATTR_CHAN          2          Attribute channel: 0=X, 1=Y, 2=Z, 3=W. Unused for LDS_DIRECT_LOAD.
ATTR               6          Attribute number: 0 - 32. Unused for LDS_DIRECT_LOAD.
( M0 )             32         LDS_DIRECT_LOAD:
                               { 13’b0, DataType[2:0], LDS_address[15:0] } //addr in bytes
                              LDS_PARAM_LOAD:
                               { 1’b0, new_prim_mask[15:1], lds_param_offset[15:0] }

M0 is implicitly read for this instruction and must be initialized before these instructions.

new_prim_mask
     a mask that has a bit per quad indicating that this quad begins a new primitive; zero indicates same
     primitive as previous quad. There is an implied "one" for the first quad in the wave (every wave begins a
     new primitive) and so bit[0] is omitted.

lds_param_offset
     The parameter offset indicates the starting address of the parameters in LDS. Space before that can be used
     as temporary wave storage space. Lds_param_offset bits [6:0] must be set to zero.

Example LDS_PARAM_LOAD (new_prim_mask[3:0] = 0110)

LDS_ADDR = lds_base + param_offset + attr#*numPrimsInVector*12DWORDs + prim#*12 + attr_offset
     (attr_offset = 0..11 : 0 = P0.x, 1 = P0.Y, … 11 = P2.W)
     From NewPrimMask h/w derives NumPrimInVec and Prim# (0..15)

If the dest-VGPR is out of range, the load is still performed but EXEC is forced to zero.

LDS_PARAM_LOAD and LDS_DIRECT_LOAD use EXEC per quad (if any pixel is enabled in the quad, data is
written to all 4 pixels/threads in the quad).

12.2.1.1. 16-bit Parameter Data
16-bit parameters are packed in LDS as pairs of attributes in DWORDs: ATTR0.X and ATTR1.X share a DWORD.
There is an alternate packing mode where the parameters are not packed (one 16-bit param in low half of
DWORD). These attributes can be read with the same LDS_PARAM_LOAD instruction, and returns the packed
DWORD with 2 attributes (when they are packed). Interpolation can then be done using specific mixed-
precision FMA opcodes, along with DPP (to select P0, P10 or P20) and OPSEL (to select upper or lower 16-bits).

Barycentrics are 32-bits, not 16 bit.

12.2.1.2. Parameter Load Data Hazard Avoidance
These data dependency rules apply to both parameter and direct loads.

LDS_DIRECT_LOAD and LDS_PARAM_LOAD read data from LDS and write it into VGPRs, and they use EXPcnt
to track when the instruction has completed and written the VGPRs.

It is up to the shader program to ensure that data hazards are avoided. These instructions are issued along a
different path from VALU instructions so it is possible that previous VALU instructions may still be reading
from the VGPR that these LDS instructions are going to write and this could lead to a hazard.

EXPcnt is used to track read-after-write hazards where LDS_PARAM_LOAD writes a value to a VGPR and
another instruction reads it. The shader program uses "s_waitcnt EXPcnt" to wait for results from a
LDS_DIRECT_LOAD or LDS_PARAM_LOAD to be available in VGPRs before consuming it in a subsequent
instruction. The VINTERP instructions have a "wait_EXPcnt" field to assist in avoid this hazard.

These are skipped when EXEC==0 and EXPCnt==0 (like memory ops).

Mixed exports & LDS-direct/param instructions from the same wave might not complete in order (both use
EXPcnt), requiring "s_waitcnt 0" if they are overlapped.

      LDS_PARAM_LOAD V2
      S_WAITCNT EXPcnt 0

A potential Write-After-Read hazard exists if a VALU instruction reads a VGPR and then LDS_PARAM_LOAD
writes that VGPR: It is possible the LDS_PARAM_LOAD overwrites the VALU’s source VGPR before it was read.
The user must prevent this by using the "wait_Vdst" field of the LDS_PARAM_LOAD instruction. This field
indicates the maximum number of uncompleted VALU instructions that may be outstanding when this
LDS_PARAM_LOAD is issued. Use this to ensure any dependent VALU instructions have completed.

Another potential data hazard involves LDS_PARAM_LOAD overwriting a VGPR that has not yet been read as a
source by a previous VMEM (LDS, Texture, Buffer, Flat) instruction. To avoid this hazard, the user must ensure
that the VMEM instruction has read its source VGPRs. This can be achieved by issuing any VALU or export
instruction before the LDS_PARAM_LOAD.
