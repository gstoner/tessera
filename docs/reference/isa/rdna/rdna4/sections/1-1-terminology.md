# 1.1. Terminology

> RDNA4 ISA — pages 13–14

Chapter 1. Introduction
This document describes the instruction set and shader program accessible state for RDNA4 devices.

The AMD RDNA4 processor implements a parallel micro-architecture that provides a platform for computer
graphics applications and general-purpose data parallel applications.

1.1. Terminology
The following terminology and conventions are used in this document:

                                                  Table 1. Conventions
*                       Any number of alphanumeric characters in the name of a code format, parameter, or instruction.
[1,2)                   A range that includes the left-most value (in this case, 1), but excludes the right-most value (in this
                        case, 2).
[1,2]                   A range that includes both the left-most and right-most values.
{x | y} or {x, y}       One of the multiple options listed. In this case, X or Y. For example, S_AND_{B32, B64} defines
                        two legal instructions: S_AND_B32 and S_AND_B64.
0.0                     A floating-point value.
1011b                   A binary value, in this example a 4-bit value.
'b0010                  A binary value of unspecified size.
32’b0010                A 32-bit binary value. Binary values may include underscores for readability and can be ignored
                        when parsing the value. If fewer than 32 bits are shown, the upper bits are assumed to be zero.
0x1A                    A hexadecimal value.
'h123                   A hexadecimal value.
24’h01A3B4              A 24-bit hexadecimal value.
7:4                     A bit range, from bit 7 to bit 4, inclusive. The high-order bit is shown first. May be enclosed in
[7:4]                   brackets.
italicized word or phrase The first use of a term or concept basic to the understanding of GPU-based computing.

                                                  Table 2. Basic Terms
Term                    Description
RDNA4 Processor         The RDNA4 shader processor is a scalar and vector ALU with memory access designed to run
                        complex programs on behalf of a wave.
Kernel                  A program executed by the shader processor for each work-item submitted to it.
Shader Program          Same meaning as "Kernel". The shader types are:
                        CS (Compute Shader), and for graphics-capable devices, PS (Pixel Shader), GS (Geometry Shader),
                        and HS (Hull Shader).
Dispatch                A dispatch launches a 1D, 2D, or 3D grid of work to the RDNA4 processor array.
Work-group              A work-group is a collection of waves that have the ability to synchronize with each other with
                        barriers; they also can share data through the Local Data Share. Waves in a work-group all run on
                        the same WGP.
Wave                    A collection of 32 or 64 work-items that execute in parallel on a single RDNA4 processor.
Work-item               A single element of work: one element from the dispatch grid , or in graphics a pixel, vertex or
                        primitive.
Thread                  A synonym for "work-item".
Lane                    A synonym for "work-item" typically used only when describing VALU operations.
SA                      Shader Array. A collection of compute units.

Term                     Description
SE                       Shader Engine. A collection of shader arrays.
SGPR                     Scalar General Purpose Registers. 32-bit registers that are shared by work-items in each wave.
VGPR                     Vector General Purpose Registers. 32-bit registers that are private to each work-items in a wave.
LDS                      Local Data Share. A multi-bank scratch memory allocated to waves or work-groups
VMEM                     Vector Memory. Refers to LDS, Texture, Global, Flat and Scratch memory.
SIMD                     Single Instruction Multiple Data. In this document a SIMD refers to the Vector ALU unit that
                         processes instructions for a single wave.
SIMD32                   Single Instruction Multiple Data that operates on 32 work-items in parallel.
CWSR                     Compute Wave Save-Restore (context switch)
Literal Constant         A 32-bit integer or float constant that is placed in the instruction stream.
Scalar ALU (SALU)        The scalar ALU operates on one value per wave and manages all control flow.
Vector ALU (VALU)        The vector ALU maintains Vector GPRs that are unique for each work-item and execute arithmetic
                         operations uniquely on each work-item.
Work-group Processor The basic unit of shader computation hardware, including scalar & vector ALU’s and memory, as
(WGP)                well as LDS and scalar caches. In some contexts this is also referred to as a "Double Compute Unit" or
                     "Compute Unit Pair".
Compute Unit (CU)        One half of a WGP. Contains 2 SIMD32’s that share one path to memory.
Microcode format         The microcode format describes the bit patterns used to encode instructions. Each instruction is
                         32-bits or more, in units of 32-bits.
Instruction              An instruction is the basic unit of the kernel. Instructions include: vector ALU, scalar ALU,
                         memory transfer, and control flow operations.
NGG                      Next Generation Graphics pipeline
Quad                     A quad is a 2x2 group of screen-aligned pixels. This is relevant for sampling texture maps.
Texture Sampler (S#)     A texture sampler is a 128-bit entity that describes how the vector memory system reads and
                         samples (filters) a texture map.
Texture Resource (T#)    A texture resource descriptor describes an image in memory: address, data format, width, height,
                         depth, etc.
Buffer Resource (V#)     A buffer resource descriptor describes a buffer in memory: address, data format, stride, etc.
DPP                      Data Parallel Primitives: VALU instructions that can pass data between work-items
LSB                      Least Significant Bit
MSB                      Most Significant Bit
DWORD                    32-bit data
SHORT                    16-bit data
BYTE                     8-bit data
$                        Cache. Frequently used as a suffix abbreviation, as in "Inst$" to mean "Instruction Cache".
K                        Constant. Occasionally used to refer to constants, constant-buffers or constant-cache ("K$")

                            Table 3. Instruction suffixes have the following definitions:
Format                   Meaning
B32                      binary (untyped data) 32-bit
B64                      binary (untyped data) 64-bit
F16                      floating-point 16-bit (sign + exp5 + mant10)
F32                      floating-point 32-bit (IEEE 754 single-precision float) (sign + exp8 + mant23)
F64                      floating-point 64-bit (IEEE 754 double-precision float) (sign + exp11 + mant52)
BF16                     floating-point 16-bit for machine learning ("bfloat16"). (sign + exp8 + mant7)
F8                       floating-point 8-bit (also called "FP8"): (sign + exp4 + mant3)
BF8                      floating-point 8-bit : (sign + exp5 + mant2)
I8                       signed 8-bit integer
