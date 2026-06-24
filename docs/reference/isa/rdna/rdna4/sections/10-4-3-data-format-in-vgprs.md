# 10.4.3. Data format in VGPRs

> RDNA4 ISA — pages 135–135

   values for missing components based on T#.dst_sel. Then DMASK is applied and only those components
   selected are returned to the shader.

Stores
   When writing an image object, it is only possible to store an entire element (all components) - not only
   individual components. The components come from consecutive VGPRs and if the surface uses more
   components than the shader supplies, the missing components are filled in by replicating the first VGPR
   into the missing components. E.g. if the surface has 4 components and the shader only supplies X and Y,
   the surfaces is written with XYXX.
   For example if the DMASK=1001, the shader sends Red from VGPR_N and Alpha from VGPR_N+1 to the texture
   unit. If the image object is RGB, the texel is overwritten with Red from the VGPR_N, Green and Blue set to Red
   value, and Alpha from the shader ignored. For D16=1, the DMASK has 1 bit set per 16-bits of data to be written
   from VGPRs to memory. The position of the bits in DMASK is irrelevant, only the number of bits set to 1.

"D16" instructions
   Load and store instructions also come in a "D16" variant. For stores, each 32bit VGPR holds two 16bit data
   elements that are passed to the texture unit which in turn, converts to the texture format before writing to
   memory. For loads, data returned from the texture unit is converted to 16 bits and a pair of data are stored
   in each 32bit VGPR (LSBs first, then MSBs). If there is only one component, the data goes into the lower half
   of the VGPR unless the "HI" instruction variant is used in which case the high-half of the VGPR is loaded
   with data.

Atomics
   Image atomic operations are supported only on 32- and 64-bit-per-pixel surfaces. The surface data format is
   specified in the resource constant. Atomic operations treat the element as a single component of 32- or 64-
   bits. For atomic operations, DMASK is set to the number of VGPRs (DWORDs) to send to the texture unit.
   DMASK legal values for atomic image operations: All other values of DMASK are illegal.

     • 0x1 = 32bit atomics except cmpswap
     • 0x3 = 32bit atomic cmpswap
     • 0x3 = 64bit atomics except cmpswap
     • 0xf = 64bit atomic cmpswap
     • Atomics with Return: Data is read out of the VGPR(s), starting at VDATA, to supply to the atomic
       operation. If the atomic returns a value to VGPRs, that data is returned to those same VGPRs starting at
       VDATA.

The DMASK must be compatible with the resource’s data format.

Denormals in Floats
   Sample ops flush denormals, and loads do not modify denormals.

10.4.3. Data format in VGPRs
Data in VGPRs sent to texture (stores) or returned from texture (loads) is in one of a few standard formats, and
the texture unit converts to/from the memory format.

      FORMAT                           VGPR data format                     If D16==1
      SINT                             signed 32-bit integer                16 bit signed int
      UINT                             unsigned 32-bit integer              16 bit unsigned int
