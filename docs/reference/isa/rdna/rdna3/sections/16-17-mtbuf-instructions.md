# 16.17. MTBUF Instructions

> RDNA3 ISA — pages 563–566

16.17. MTBUF Instructions
The bitfield map of the MTBUF format is:

TBUFFER_LOAD_FORMAT_X                                            0

Typed buffer load 1 component with format conversion.

  VDATA[31 : 0].b = ConvertFromFormat(MEM[TADDR.X]);
  // Mem access size depends on format

TBUFFER_LOAD_FORMAT_XY                                           1

Typed buffer load 2 components with format conversion.

  VDATA[31 : 0].b = ConvertFromFormat(MEM[TADDR.X]);
  // Mem access size depends on format
  VDATA[63 : 32].b = ConvertFromFormat(MEM[TADDR.Y])

TBUFFER_LOAD_FORMAT_XYZ                                          2

Typed buffer load 3 components with format conversion.

  VDATA[31 : 0].b = ConvertFromFormat(MEM[TADDR.X]);
  // Mem access size depends on format
  VDATA[63 : 32].b = ConvertFromFormat(MEM[TADDR.Y]);
  VDATA[95 : 64].b = ConvertFromFormat(MEM[TADDR.Z])

TBUFFER_LOAD_FORMAT_XYZW                                         3

Typed buffer load 4 components with format conversion.

  VDATA[31 : 0].b = ConvertFromFormat(MEM[TADDR.X]);
  // Mem access size depends on format
  VDATA[63 : 32].b = ConvertFromFormat(MEM[TADDR.Y]);
  VDATA[95 : 64].b = ConvertFromFormat(MEM[TADDR.Z]);

  VDATA[127 : 96].b = ConvertFromFormat(MEM[TADDR.W])

TBUFFER_STORE_FORMAT_X                                            4

Typed buffer store 1 component with format conversion.

  MEM[TADDR.X] = ConvertToFormat(VDATA[31 : 0].b);
  // Mem access size depends on format

TBUFFER_STORE_FORMAT_XY                                           5

Typed buffer store 2 components with format conversion.

  MEM[TADDR.X] = ConvertToFormat(VDATA[31 : 0].b);
  // Mem access size depends on format
  MEM[TADDR.Y] = ConvertToFormat(VDATA[63 : 32].b)

TBUFFER_STORE_FORMAT_XYZ                                          6

Typed buffer store 3 components with format conversion.

  MEM[TADDR.X] = ConvertToFormat(VDATA[31 : 0].b);
  // Mem access size depends on format
  MEM[TADDR.Y] = ConvertToFormat(VDATA[63 : 32].b);
  MEM[TADDR.Z] = ConvertToFormat(VDATA[95 : 64].b)

TBUFFER_STORE_FORMAT_XYZW                                         7

Typed buffer store 4 components with format conversion.

  MEM[TADDR.X] = ConvertToFormat(VDATA[31 : 0].b);
  // Mem access size depends on format
  MEM[TADDR.Y] = ConvertToFormat(VDATA[63 : 32].b);
  MEM[TADDR.Z] = ConvertToFormat(VDATA[95 : 64].b);
  MEM[TADDR.W] = ConvertToFormat(VDATA[127 : 96].b)

TBUFFER_LOAD_D16_FORMAT_X                                         8

Typed buffer load 1 component with format conversion, packed 16-bit components in data register.

  VDATA[15 : 0].b16 = 16'B(ConvertFromFormat(MEM[TADDR.X]));
  // Mem access size depends on format
  // VDATA[31:16].b16 is preserved.

TBUFFER_LOAD_D16_FORMAT_XY                                                                                  9

Typed buffer load 2 components with format conversion, packed 16-bit components in data register.

  VDATA[15 : 0].b16 = 16'B(ConvertFromFormat(MEM[TADDR.X]));
  // Mem access size depends on format
  VDATA[31 : 16].b16 = 16'B(ConvertFromFormat(MEM[TADDR.Y]))

TBUFFER_LOAD_D16_FORMAT_XYZ                                                                                10

Typed buffer load 3 components with format conversion, packed 16-bit components in data register.

  VDATA[15 : 0].b16 = 16'B(ConvertFromFormat(MEM[TADDR.X]));
  // Mem access size depends on format
  VDATA[31 : 16].b16 = 16'B(ConvertFromFormat(MEM[TADDR.Y]));
  VDATA[47 : 32].b16 = 16'B(ConvertFromFormat(MEM[TADDR.Z]));
  // VDATA[63:48].b16 is preserved.

TBUFFER_LOAD_D16_FORMAT_XYZW                                                                               11

Typed buffer load 4 components with format conversion, packed 16-bit components in data register.

  VDATA[15 : 0].b16 = 16'B(ConvertFromFormat(MEM[TADDR.X]));
  // Mem access size depends on format
  VDATA[31 : 16].b16 = 16'B(ConvertFromFormat(MEM[TADDR.Y]));
  VDATA[47 : 32].b16 = 16'B(ConvertFromFormat(MEM[TADDR.Z]));
  VDATA[63 : 48].b16 = 16'B(ConvertFromFormat(MEM[TADDR.W]))

TBUFFER_STORE_D16_FORMAT_X                                                                                 12

Typed buffer store 1 component with format conversion, packed 16-bit components in data register.

  MEM[TADDR.X] = ConvertToFormat(32'B(VDATA[15 : 0].b16));

  // Mem access size depends on format

TBUFFER_STORE_D16_FORMAT_XY                                                                                 13

Typed buffer store 2 components with format conversion, packed 16-bit components in data register.

  MEM[TADDR.X] = ConvertToFormat(32'B(VDATA[15 : 0].b16));
  // Mem access size depends on format
  MEM[TADDR.Y] = ConvertToFormat(32'B(VDATA[31 : 16].b16))

TBUFFER_STORE_D16_FORMAT_XYZ                                                                                14

Typed buffer store 3 components with format conversion, packed 16-bit components in data register.

  MEM[TADDR.X] = ConvertToFormat(32'B(VDATA[15 : 0].b16));
  // Mem access size depends on format
  MEM[TADDR.Y] = ConvertToFormat(32'B(VDATA[31 : 16].b16));
  MEM[TADDR.Z] = ConvertToFormat(32'B(VDATA[47 : 32].b16))

TBUFFER_STORE_D16_FORMAT_XYZW                                                                               15

Typed buffer store 4 components with format conversion, packed 16-bit components in data register.

  MEM[TADDR.X] = ConvertToFormat(32'B(VDATA[15 : 0].b16));
  // Mem access size depends on format
  MEM[TADDR.Y] = ConvertToFormat(32'B(VDATA[31 : 16].b16));
  MEM[TADDR.Z] = ConvertToFormat(32'B(VDATA[47 : 32].b16));
  MEM[TADDR.W] = ConvertToFormat(32'B(VDATA[63 : 48].b16))
