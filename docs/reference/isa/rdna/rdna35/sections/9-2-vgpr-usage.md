# 9.2. VGPR Usage

> RDNA3.5 ISA — pages 95–96

Opcode                                   Description - all address components for buffer ops are uint
BUFFER_ATOMIC_MIN_U32                    32b , dst = (src < dst) ? src : dst, (unsigned) returns previous value if glc==1
BUFFER_ATOMIC_MIN_U64                    64b , dst = (src < dst) ? src : dst, (unsigned) returns previous value if glc==1
BUFFER_ATOMIC_XOR_B32                    32b , dst ^= src, returns previous value if glc==1
BUFFER_ATOMIC_XOR_B64                    64b , dst ^= src, returns previous value if glc==1
BUFFER_GL0_INV                           invalidate the shader L0 cache (texture cache) associated with this wave.
BUFFER_GL1_INV                           invalidate the GL1 (L1) cache associated with this wave, for this wave’s VMID

  • BUFFER*_FORMAT instructions include a data-format conversion specified in the resource constant (V#).
  • In the table above, "D16" means the data in the VGPR is 16-bits, not the usual 32 bits.
    "D16_HI" means that the upper 16-bits of the VGPR is used instead of "D16" that uses the lower 16 bits.

9.2. VGPR Usage
VGPRs supply address and store-data, and they can be the destination for return data.

Address
   Zero, one or two VGPRs are used, depending on the index-enable (IDXEN) and offset-enable (OFFEN) in the
   instruction word. These are unsigned ints.
   For 64-bit addresses the LSBs are in VGPRn and the MSBs are in VGPRn+1.

                                                     Table 40. Address VGPRs
                                         IDXEN OFFEN VGPRn                 VGPRn+1
                                         0           0       nothing
                                         0           1       uint offset
                                         1           0       uint index
                                         1           1       uint index uint offset

Store Data : N consecutive VGPRs, starting at VDATA. The data format specified in the instruction word’s
opcode and D16 setting determines how many DWORDs the shader provides to store.

Load Data : Same as stores. Data is returned to consecutive VGPRs.

Load Data Format : Load data is 32 or 16 bits, based on the data format in the instruction or resource and D16.
Float or normalized data is returned as floats; integer formats are returned as integers (signed or unsigned,
same type as the memory storage format). Memory loads of data in memory that is 32 or 64 bits do not undergo
any format conversion unless they return as 16-bit due to D16 being set to 1.

Atomics with Return : Data is read out of the VGPR(s) starting at VDATA to supply to the atomic operation. If
the atomic returns a value to VGPRs, that data is returned to those same VGPRs starting at VDATA.

                                   Table 41. Data format in VGPRs and Memory
Instruction                                  Memory Format             VGPR Format                     Notes
BUFFER_LOAD_U8                               ubyte                     V0[31:0] = {24’b0, byte}
BUFFER_LOAD_D16_U8                           ubyte                     V0[15:0] = {8’b0, byte}         writes only 16 bits
BUFFER_LOAD_D16_HI_U8                        ubyte                     V0[31:16] = {8’h0, byte}        writes only 16 bits
BUFFER_LOAD_S8                               sbyte                     V0[31:0] = { 24{sign}, byte}
BUFFER_LOAD_D16_S8                           sbyte                     V0[15:0] {8{sign}, byte}        writes only 16 bits

Instruction                              Memory Format      VGPR Format                      Notes
BUFFER_LOAD_D16_HI_S8                    sbyte              V0[31:16] = {8{sign}, byte}      writes only 16 bits
BUFFER_LOAD_U16                          ushort             V0[31:0] = { 16’b0, short}
BUFFER_LOAD_S16                          sshort             V0[31:0] = { 16{sign}, short}
BUFFER_LOAD_D16_B16                      short              V0[15:0] = short                 writes only 16 bits
BUFFER_LOAD_D16_HI_B16                   short              V0[31:16] = short                writes only 16 bits
BUFFER_LOAD_B32                          DWORD              DWORD
BUFFER_LOAD_FORMAT_X                     FORMAT field       float, uint or sint              data type in VGPR is
                                                            Load X into V0[31:0]             based on FORMAT
BUFFER_LOAD_FORMAT_XY                    FORMAT field       float, uint or sint              field.
                                                            Load X,Y into V0[31:0], V1[31:0] (D16_X and D16_HI_X
BUFFER_LOAD_FORMAT_XYZ                   FORMAT field       float, uint or sint              write only 16 bits)
                                                            Load X,Y,Z into V0[31:0],
                                                            V1[31:0], V2[31:0]
BUFFER_LOAD_FORMAT_XYZW                  FORMAT field       float, uint or sint
                                                            Load X,Y,Z,W into V0[31:0],
                                                            V1[31:0], V2[31:0], v3[31:0]
BUFFER_LOAD_D16_FORMAT_X                 FORMAT field       float, uint or sint
                                                            Load X into in V0[15:0]
BUFFER_LOAD_D16_HI_FORMAT_X              FORMAT field       float, ushort or sshort
                                                            Load X into in V0[31:16]
BUFFER_LOAD_D16_FORMAT_XY                FORMAT field       float, ushort or sshort
                                                            Load X,Y into in V0[15:0],
                                                            V0[31:16]
BUFFER_LOAD_D16_FORMAT_XYZ               FORMAT field       float, ushort or sshort
                                                            Load X,Y,Z into in V0[15:0],
                                                            V0[31:16], V1[15:0]
BUFFER_LOAD_D16_FORMAT_XYZW              FORMAT field       float, ushort or sshort
                                                            Load X,Y,Z,W into in V0[15:0],
                                                            V0[31:16], V1[15:0], V1[31:16]

Where "V0" is the VDATA VGPR; V1 is the VDATA+1 VGPR, etc.

Instruction                              VGPR Format                            Memory       Notes
                                                                                Format
BUFFER_STORE_B8                          byte in [7:0]                          byte
BUFFER_STORE_D16_HI_B8                   byte in [23:16]                        byte
BUFFER_STORE_B16                         short in [15:0]                        short
BUFFER_STORE_D16_HI_B16                  short in [31:16]                       short
BUFFER_STORE_B32                         data in [31:0]                         DWORD
