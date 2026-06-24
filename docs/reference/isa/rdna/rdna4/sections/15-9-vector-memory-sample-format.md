# 15.9. Vector Memory Sample Format

> RDNA4 ISA — pages 210–211

15.9. Vector Memory Sample Format

  Description         Memory Image-Sample Instructions - sample and gather4 (instructions that do use a
                      sampler)

                                               Table 115. VSAMPLE Fields
Field Name      Bits           Format or Description
DIM             [2:0]          Dimensionality of the resource constant. Set to bits [3:1] of the resource type field.
TFE             [3]            Partially resident texture, texture fault enable.
R128            [4]            Resource constant size: 1 = 128bit, 0 = 256bit
D16             [5]            Data components are 16-bits (instead of the usual 32 bits).
A16             [6]            Address components are 16-bits (instead of the usual 32 bits).
UNRM            [13]           Force address to be un-normalized. User must set to 1 for Image stores & atomics. When
                               set, all address components are 16 bits (packed into 2 per DWORD), except:
                               Texel offsets (3 6bit UINT packed into 1 DWORD)
                               PCF reference (for "_C" instructions)
                               Address components are 16b uint for image ops without sampler; 16b float with sampler.
OP              [21:14]        Opcode. See table below. (combined bits 53 with 18-16 to form opcode)
DMASK           [25:22]        Data VGPR enable mask: 1 .. 4 consecutive VGPRs
                               Defines which components are returned:
                               0=red,1=green,2=blue,3=alpha
                               Enabled components come from consecutive VGPRs.
                               E.G. dmask=1001 : Red is in VGPRn and alpha in VGPRn+1.
                               For D16 writes, DMASK is only used as a word count: each bit represents 16 bits of data to
                               be written starting at the LSB’s of VDATA, then MSBs, then VDATA+1 etc. Bit position is
                               ignored.
ENCODING        [31:26]        'b111001
VDATA           [39:32]        Address of VGPR to supply first component of write data or receive first component of
                               read-data.
LWE             [40]           LOD Warning Enable. When set to 1, a texture fetch may return "LOD_CLAMPED = 1".
RSRC            [49:41]        SGPR to supply V# (resource constant) in 4 or 8 consecutive SGPRs. Must be multiple of 4 in
                               the range 0-120.
SCOPE           [51:50]        Memory Scope
TH              [54:52]        Memory Temporal Hint
SAMP            [63:55]        SGPR to supply S# (sampler constant) in 4 or 8 consecutive SGPRs. Must be multiple of 4 in
                               the range 0-120.
VADDR0          [71:64]        Address of VGPR to supply first component of address.
VADDR1          [79:72]        Address of VGPR to supply second component of address.
VADDR2          [87:80]        Address of VGPR to supply third component of address.
VADDR3          [95:88]        Address of VGPR to supply fourth component of address.

                              Table 116. VSAMPLE Opcodes
Opcode # Name                             Opcode # Name
24         IMAGE_MSAA_LOAD                55           IMAGE_GATHER4_C_LZ_O
27         IMAGE_SAMPLE                   56           IMAGE_GET_LOD
28         IMAGE_SAMPLE_D                 57           IMAGE_SAMPLE_D_G16

Opcode # Name                          Opcode # Name
29         IMAGE_SAMPLE_L              58      IMAGE_SAMPLE_C_D_G16
30         IMAGE_SAMPLE_B              59      IMAGE_SAMPLE_D_O_G16
31         IMAGE_SAMPLE_LZ             60      IMAGE_SAMPLE_C_D_O_G16
32         IMAGE_SAMPLE_C              64      IMAGE_SAMPLE_CL
33         IMAGE_SAMPLE_C_D            65      IMAGE_SAMPLE_D_CL
34         IMAGE_SAMPLE_C_L            66      IMAGE_SAMPLE_B_CL
35         IMAGE_SAMPLE_C_B            67      IMAGE_SAMPLE_C_CL
36         IMAGE_SAMPLE_C_LZ           68      IMAGE_SAMPLE_C_D_CL
37         IMAGE_SAMPLE_O              69      IMAGE_SAMPLE_C_B_CL
38         IMAGE_SAMPLE_D_O            70      IMAGE_SAMPLE_CL_O
39         IMAGE_SAMPLE_L_O            71      IMAGE_SAMPLE_D_CL_O
40         IMAGE_SAMPLE_B_O            72      IMAGE_SAMPLE_B_CL_O
41         IMAGE_SAMPLE_LZ_O           73      IMAGE_SAMPLE_C_CL_O
42         IMAGE_SAMPLE_C_O            74      IMAGE_SAMPLE_C_D_CL_O
43         IMAGE_SAMPLE_C_D_O          75      IMAGE_SAMPLE_C_B_CL_O
44         IMAGE_SAMPLE_C_L_O          84      IMAGE_SAMPLE_C_D_CL_G16
45         IMAGE_SAMPLE_C_B_O          85      IMAGE_SAMPLE_D_CL_O_G16
46         IMAGE_SAMPLE_C_LZ_O         86      IMAGE_SAMPLE_C_D_CL_O_G16
47         IMAGE_GATHER4               95      IMAGE_SAMPLE_D_CL_G16
48         IMAGE_GATHER4_L             96      IMAGE_GATHER4_CL
49         IMAGE_GATHER4_B             97      IMAGE_GATHER4_B_CL
50         IMAGE_GATHER4_LZ            98      IMAGE_GATHER4_C_CL
51         IMAGE_GATHER4_C             99      IMAGE_GATHER4_C_L
52         IMAGE_GATHER4_C_LZ          100     IMAGE_GATHER4_C_B
53         IMAGE_GATHER4_O             101     IMAGE_GATHER4_C_B_CL
54         IMAGE_GATHER4_LZ_O          144     IMAGE_GATHER4H
