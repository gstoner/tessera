# 2.2.1. Compute Shaders

> RDNA3.5 ISA — pages 19–19

there are no outstanding VMEM instructions from this wave. It also does not skip either half of a VALU
instruction which writes an SGPR. See Instruction Skipping: EXEC==0 for details on instruction skipping rules.

Hardware operates such that both passes of a wave64 use the state of the wave prior to instruction execution;
the first pass of the wave64 does not affect the input to the second pass.

In addition to the EXEC mask being different between the low and high half, scalar inputs may vary between
the two passes. Both passes use the same constants, but different masks and carry-in/out.

The differences in the second pass are:
  • Input increments: Carry-in, div-fmas and v_cndmask all use the next SGPR (SSRC + 1, or VCC_HI)
  • Output increments: Carry-out, div-scale and v_cmp all write to the next SGPR (SDST + 1, or VCC_HI)
     ◦ v_cmpx writes to EXEC_HI instead of EXEC_LO

The upper 32-bits of EXEC and VCC are ignored for wave32 waves. VCCZ and EXECZ reflect the status of the
lowest 32-bits of VCC and EXEC respectively for wave32 waves.

2.2. Shader Types

2.2.1. Compute Shaders
Compute kernels (shaders) are generic programs that can run on the RDNA3.5 processor, taking data from
memory, processing it, and writing results back to memory. Compute kernels are created by a dispatch, which
causes the RDNA3.5 processors to run the kernel over all of the work-items in a 1D, 2D, or 3D grid of data. The
RDNA3.5 processor walks through this grid and generates waves, which then run the compute kernel. Each
work-item is initialized with its unique address (index) within the grid. Based on this index, the work-item
computes the address of the data it is required to work on and what to do with the results.

2.2.2. Graphics Shaders
The shader supports 3 types of graphics waves: PS, GS, and HS.

Rendering modes (launch behavior):
  • Normal NGG - Geometry Engine (GE) sends info to wave launch hardware to init VGPRs for each element
    (prim) launched; GE fetches index and vertex buffer data and loads to VGPRs
  • Mesh shader - turns GS-launch into a CS-style launch, and wave launch hardware does unrolling into
    elements and generates element indices on the fly. The mesh shader program determines how to use this
    index value.
