# 15.9.2. GLOBAL

> RDNA3 ISA — pages 193–193

Field Name               Bits           Format or Description
SVE                      [55]           Scratch VGPR Enable. 1 = scratch address includes a VGPR to provide an offset; 0 =
                                        no VGPR used.
VDST                     [63:56]        Destination VGPR for data returned from memory to VGPRs.

                                   Table 108. FLAT Opcodes
Opcode # Name                                     Opcode # Name
16         FLAT_LOAD_U8                           56         FLAT_ATOMIC_MIN_I32
17         FLAT_LOAD_I8                           57         FLAT_ATOMIC_MIN_U32
18         FLAT_LOAD_U16                          58         FLAT_ATOMIC_MAX_I32
19         FLAT_LOAD_I16                          59         FLAT_ATOMIC_MAX_U32
20         FLAT_LOAD_B32                          60         FLAT_ATOMIC_AND_B32
21         FLAT_LOAD_B64                          61         FLAT_ATOMIC_OR_B32
22         FLAT_LOAD_B96                          62         FLAT_ATOMIC_XOR_B32
23         FLAT_LOAD_B128                         63         FLAT_ATOMIC_INC_U32
24         FLAT_STORE_B8                          64         FLAT_ATOMIC_DEC_U32
25         FLAT_STORE_B16                         65         FLAT_ATOMIC_SWAP_B64
26         FLAT_STORE_B32                         66         FLAT_ATOMIC_CMPSWAP_B64
27         FLAT_STORE_B64                         67         FLAT_ATOMIC_ADD_U64
28         FLAT_STORE_B96                         68         FLAT_ATOMIC_SUB_U64
29         FLAT_STORE_B128                        69         FLAT_ATOMIC_MIN_I64
30         FLAT_LOAD_D16_U8                       70         FLAT_ATOMIC_MIN_U64
31         FLAT_LOAD_D16_I8                       71         FLAT_ATOMIC_MAX_I64
32         FLAT_LOAD_D16_B16                      72         FLAT_ATOMIC_MAX_U64
33         FLAT_LOAD_D16_HI_U8                    73         FLAT_ATOMIC_AND_B64
34         FLAT_LOAD_D16_HI_I8                    74         FLAT_ATOMIC_OR_B64
35         FLAT_LOAD_D16_HI_B16                   75         FLAT_ATOMIC_XOR_B64
36         FLAT_STORE_D16_HI_B8                   76         FLAT_ATOMIC_INC_U64
37         FLAT_STORE_D16_HI_B16                  77         FLAT_ATOMIC_DEC_U64
51         FLAT_ATOMIC_SWAP_B32                   80         FLAT_ATOMIC_CMPSWAP_F32
52         FLAT_ATOMIC_CMPSWAP_B32                81         FLAT_ATOMIC_MIN_F32
53         FLAT_ATOMIC_ADD_U32                    82         FLAT_ATOMIC_MAX_F32
54         FLAT_ATOMIC_SUB_U32                    86         FLAT_ATOMIC_ADD_F32

15.9.2. GLOBAL
                                       Table 109. GLOBAL Opcodes
Opcode # Name                                          Opcode # Name
16         GLOBAL_LOAD_U8                              55        GLOBAL_ATOMIC_CSUB_U32
17         GLOBAL_LOAD_I8                              56        GLOBAL_ATOMIC_MIN_I32
18         GLOBAL_LOAD_U16                             57        GLOBAL_ATOMIC_MIN_U32
19         GLOBAL_LOAD_I16                             58        GLOBAL_ATOMIC_MAX_I32
20         GLOBAL_LOAD_B32                             59        GLOBAL_ATOMIC_MAX_U32
21         GLOBAL_LOAD_B64                             60        GLOBAL_ATOMIC_AND_B32
22         GLOBAL_LOAD_B96                             61        GLOBAL_ATOMIC_OR_B32
23         GLOBAL_LOAD_B128                            62        GLOBAL_ATOMIC_XOR_B32
24         GLOBAL_STORE_B8                             63        GLOBAL_ATOMIC_INC_U32
