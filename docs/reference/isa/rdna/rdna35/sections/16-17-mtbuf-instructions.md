# 16.17. MTBUF Instructions

> RDNA3.5 ISA — pages 600–604

16.17. MTBUF Instructions
The bitfield map of the MTBUF format is:

TBUFFER_LOAD_FORMAT_X                                                                                             0

Load 1-component formatted data from a buffer surface, convert the data to 32 bit integral or floating point
format, then store the result into a vector register. The instruction specifies the data format of the surface,
overriding the resource descriptor.

  VDATA[31 : 0].b32 = ConvertFromFormat(MEM[TADDR.X]);
  // Mem access size depends on format

TBUFFER_LOAD_FORMAT_XY                                                                                            1

Load 2-component formatted data from a buffer surface, convert the data to 32 bit integral or floating point
format, then store the result into a vector register. The instruction specifies the data format of the surface,
overriding the resource descriptor.

  VDATA[31 : 0].b32 = ConvertFromFormat(MEM[TADDR.X]);
  // Mem access size depends on format
  VDATA[63 : 32].b32 = ConvertFromFormat(MEM[TADDR.Y])

TBUFFER_LOAD_FORMAT_XYZ                                                                                           2

Load 3-component formatted data from a buffer surface, convert the data to 32 bit integral or floating point
format, then store the result into a vector register. The instruction specifies the data format of the surface,
overriding the resource descriptor.

  VDATA[31 : 0].b32 = ConvertFromFormat(MEM[TADDR.X]);
  // Mem access size depends on format
  VDATA[63 : 32].b32 = ConvertFromFormat(MEM[TADDR.Y]);
  VDATA[95 : 64].b32 = ConvertFromFormat(MEM[TADDR.Z])

TBUFFER_LOAD_FORMAT_XYZW                                                                                          3

Load 4-component formatted data from a buffer surface, convert the data to 32 bit integral or floating point
format, then store the result into a vector register. The instruction specifies the data format of the surface,
overriding the resource descriptor.

  VDATA[31 : 0].b32 = ConvertFromFormat(MEM[TADDR.X]);
  // Mem access size depends on format
  VDATA[63 : 32].b32 = ConvertFromFormat(MEM[TADDR.Y]);
  VDATA[95 : 64].b32 = ConvertFromFormat(MEM[TADDR.Z]);
  VDATA[127 : 96].b32 = ConvertFromFormat(MEM[TADDR.W])

TBUFFER_STORE_FORMAT_X                                                                                            4

Convert 32 bits of data from vector input registers into 1-component formatted data and store the data into a
buffer surface. The instruction specifies the data format of the surface, overriding the resource descriptor.

  MEM[TADDR.X] = ConvertToFormat(VDATA[31 : 0].b32);
  // Mem access size depends on format

TBUFFER_STORE_FORMAT_XY                                                                                           5

Convert 64 bits of data from vector input registers into 2-component formatted data and store the data into a
buffer surface. The instruction specifies the data format of the surface, overriding the resource descriptor.

  MEM[TADDR.X] = ConvertToFormat(VDATA[31 : 0].b32);
  // Mem access size depends on format
  MEM[TADDR.Y] = ConvertToFormat(VDATA[63 : 32].b32)

TBUFFER_STORE_FORMAT_XYZ                                                                                          6

Convert 96 bits of data from vector input registers into 3-component formatted data and store the data into a
buffer surface. The instruction specifies the data format of the surface, overriding the resource descriptor.

  MEM[TADDR.X] = ConvertToFormat(VDATA[31 : 0].b32);
  // Mem access size depends on format
  MEM[TADDR.Y] = ConvertToFormat(VDATA[63 : 32].b32);
  MEM[TADDR.Z] = ConvertToFormat(VDATA[95 : 64].b32)

TBUFFER_STORE_FORMAT_XYZW                                                                                         7

Convert 128 bits of data from vector input registers into 4-component formatted data and store the data into a

buffer surface. The instruction specifies the data format of the surface, overriding the resource descriptor.

  MEM[TADDR.X] = ConvertToFormat(VDATA[31 : 0].b32);
  // Mem access size depends on format
  MEM[TADDR.Y] = ConvertToFormat(VDATA[63 : 32].b32);
  MEM[TADDR.Z] = ConvertToFormat(VDATA[95 : 64].b32);
  MEM[TADDR.W] = ConvertToFormat(VDATA[127 : 96].b32)

TBUFFER_LOAD_D16_FORMAT_X                                                                                        8

Load 1-component formatted data from a buffer surface, convert the data to packed 16 bit integral or floating
point format, then store the result into a vector register. The instruction specifies the data format of the
surface, overriding the resource descriptor.

  VDATA[15 : 0].b16 = 16'B(ConvertFromFormat(MEM[TADDR.X]));
  // Mem access size depends on format
  // VDATA[31:16].b16 is preserved.

TBUFFER_LOAD_D16_FORMAT_XY                                                                                       9

Load 2-component formatted data from a buffer surface, convert the data to packed 16 bit integral or floating
point format, then store the result into a vector register. The instruction specifies the data format of the
surface, overriding the resource descriptor.

  VDATA[15 : 0].b16 = 16'B(ConvertFromFormat(MEM[TADDR.X]));
  // Mem access size depends on format
  VDATA[31 : 16].b16 = 16'B(ConvertFromFormat(MEM[TADDR.Y]))

TBUFFER_LOAD_D16_FORMAT_XYZ                                                                                     10

Load 3-component formatted data from a buffer surface, convert the data to packed 16 bit integral or floating
point format, then store the result into a vector register. The instruction specifies the data format of the
surface, overriding the resource descriptor.

  VDATA[15 : 0].b16 = 16'B(ConvertFromFormat(MEM[TADDR.X]));
  // Mem access size depends on format
  VDATA[31 : 16].b16 = 16'B(ConvertFromFormat(MEM[TADDR.Y]));
  VDATA[47 : 32].b16 = 16'B(ConvertFromFormat(MEM[TADDR.Z]));
  // VDATA[63:48].b16 is preserved.

TBUFFER_LOAD_D16_FORMAT_XYZW                                                                                    11

Load 4-component formatted data from a buffer surface, convert the data to packed 16 bit integral or floating
point format, then store the result into a vector register. The instruction specifies the data format of the
surface, overriding the resource descriptor.

  VDATA[15 : 0].b16 = 16'B(ConvertFromFormat(MEM[TADDR.X]));
  // Mem access size depends on format
  VDATA[31 : 16].b16 = 16'B(ConvertFromFormat(MEM[TADDR.Y]));
  VDATA[47 : 32].b16 = 16'B(ConvertFromFormat(MEM[TADDR.Z]));
  VDATA[63 : 48].b16 = 16'B(ConvertFromFormat(MEM[TADDR.W]))

TBUFFER_STORE_D16_FORMAT_X                                                                                      12

Convert 16 bits of data from vector input registers into 1-component formatted data and store the data into a
buffer surface. The instruction specifies the data format of the surface, overriding the resource descriptor.

  MEM[TADDR.X] = ConvertToFormat(32'B(VDATA[15 : 0].b16));
  // Mem access size depends on format

TBUFFER_STORE_D16_FORMAT_XY                                                                                     13

Convert 32 bits of data from vector input registers into 2-component formatted data and store the data into a
buffer surface. The instruction specifies the data format of the surface, overriding the resource descriptor.

  MEM[TADDR.X] = ConvertToFormat(32'B(VDATA[15 : 0].b16));
  // Mem access size depends on format
  MEM[TADDR.Y] = ConvertToFormat(32'B(VDATA[31 : 16].b16))

TBUFFER_STORE_D16_FORMAT_XYZ                                                                                    14

Convert 48 bits of data from vector input registers into 3-component formatted data and store the data into a
buffer surface. The instruction specifies the data format of the surface, overriding the resource descriptor.

  MEM[TADDR.X] = ConvertToFormat(32'B(VDATA[15 : 0].b16));
  // Mem access size depends on format
  MEM[TADDR.Y] = ConvertToFormat(32'B(VDATA[31 : 16].b16));
  MEM[TADDR.Z] = ConvertToFormat(32'B(VDATA[47 : 32].b16))

TBUFFER_STORE_D16_FORMAT_XYZW                                                                                   15

Convert 64 bits of data from vector input registers into 4-component formatted data and store the data into a
buffer surface. The instruction specifies the data format of the surface, overriding the resource descriptor.

  MEM[TADDR.X] = ConvertToFormat(32'B(VDATA[15 : 0].b16));
  // Mem access size depends on format
  MEM[TADDR.Y] = ConvertToFormat(32'B(VDATA[31 : 16].b16));
  MEM[TADDR.Z] = ConvertToFormat(32'B(VDATA[47 : 32].b16));
  MEM[TADDR.W] = ConvertToFormat(32'B(VDATA[63 : 48].b16))
