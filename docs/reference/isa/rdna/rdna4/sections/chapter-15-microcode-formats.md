# Chapter 15. Microcode Formats

> RDNA4 ISA — pages 172–173

Chapter 15. Microcode Formats
This section specifies the microcode formats. The definitions can be used to simplify compilation by providing
standard templates and enumeration names for the various instruction formats.

Endian Order - The RDNA4 architecture addresses memory and registers using little-endian byte-ordering and
bit-ordering. Multi-byte values are stored with their least-significant (low-order) byte at the lowest byte
address, and they are illustrated with their least-significant byte at the right side. Byte values are stored with
their least-significant (low-order) bit (LSB) at the lowest bit address, and they are illustrated with their LSB at
the right side.

SALU and VALU instructions may optionally include a 32-bit literal constant, and some VALU instructions may
include a 32-bit DPP control DWORD at the end of the instructions. No instruction may use both DPP and a
literal constant.

The table below summarizes the microcode formats and their widths, not including extra literal or DPP
instruction words. The sections that follow provide details.

                          Table 73. Summary of Microcode Formats
Microcode Formats                                  Reference                   Width (bits)
Scalar ALU and Control Formats
SOP2                                               SOP2                        32
SOP1                                               SOP1
SOPK                                               SOPK
SOPP                                               SOPP
SOPC                                               SOPC
Scalar Memory Format
SMEM                                               SMEM                        64
Vector ALU Format
VOP1                                               VOP1                        32
VOP2                                               VOP2                        32
VOPC                                               VOPC                        32
VOP3                                               VOP3                        64
VOP3SD                                             VOP3SD                      64
VOP3P                                              VOP3P                       64
VOPD                                               VOPD                        64
DPP16                                              DPP16                       32
DPP8                                               DPP8                        32
Vector Parameter Interpolation Format
VINTERP                                            VINTERP                     64
LDS Parameter Load and Direct Load
VDSDIR                                             VDSDIR                      32
Data Share Format
VDS                                                VDS                         64
Vector Memory Buffer Format
VBUFFER                                            VBUFFER                     96
Vector Memory Image Formats
VIMAGE                                             VIMAGE                      96

Microcode Formats                                   Reference                   Width (bits)
VSAMPLE                                             VSAMPLE                     96
Flat, Global and Scratch Formats
VFLAT                                               VFLAT                       96
VGLOBAL                                             VGLOBAL                     96
VSCRATCH                                            VSCRATCH                    96
Export Format
VEXPORT                                             VEXPORT                     64

                Any instruction field marked as "Reserved" must be set to zero.

Instruction Suffixes

Most instructions include a suffix that indicates the data type the instruction handles. This suffix may also
include a number that indicates the size of the data.

For example: "F32" indicates "32-bit floating point data", or "B16" is "16-bit binary data".

  • B = Binary
  • F = Floating point
  • BF = "brain-float" floating point
  • U = Unsigned integer
  • I = Signed integer
  • U = Unsigned integer
  • IU = Signed or Unsigned integer

When more than one data-type specifier occurs in an instruction, the first one is the result type and size, and
the later one(s) is/are input data type and size.
E.g. V_CVT_F32_I32 reads an integer and writes a float.
