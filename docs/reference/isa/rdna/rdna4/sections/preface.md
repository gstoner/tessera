# Preface

> RDNA4 ISA — pages 11–11

Preface

About This Document
This document describes the current environment, organization and program state of AMD "RDNA4"
Generation devices. It details the instruction set and the microcode formats native to this family of processors
that are accessible to programmers and compilers.

The document specifies the instructions (including the format of each type of instruction) and the relevant
program state (including how the program state interacts with the instructions). Some instruction fields are
mutually dependent; not all possible settings for all fields are legal. This document specifies the valid
combinations.

The main purposes of this document are to:

 1. Specify the language constructs and behavior, including the organization of each type of instruction in
    both text syntax and binary format
 2. Provide a reference of instruction operation that compiler writers can use to maximize performance of the
    processor

Audience
This document is intended for programmers writing application and system software, including operating
systems, compilers, loaders, linkers, device drivers, and system utilities. It assumes that programmers are
writing compute-intensive parallel applications and assumes an understanding of requisite programming
practices.

Organization
This document begins with an overview of the AMD RDNA4 processors' hardware and programming
environment. Subsequent chapters cover:

 1. Organization of RDNA4 programs
 2. Program state that is maintained
 3. Program flow
 4. Scalar ALU operations
 5. Vector ALU operations
 6. Scalar memory operations
 7. Vector memory operations
 8. Flat memory instructions
 9. Data share operations
10. Exporting the parameters of pixel color and vertex shaders
11. Detailed specification of each microcode format
12. Instruction details, first by the microcode format to which they belong, then in alphabetic order
